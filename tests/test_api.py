import json
from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.agents.graph import build_graph, memory_checkpointer
from src.agents.state import RouteDecision
from src.api import create_app
from src.tools.openf1 import MockOpenF1Client
from tests.test_graph import FakeIndex, Scripted, _analysis, _tool_call

MONACO_2024 = Path(__file__).parent / "fixtures/openf1/2024_monaco"


def _events(body: str) -> list[tuple[str, dict]]:
    events = []
    for frame in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def _client(router, fetcher, analyst) -> TestClient:
    def make_graph():
        graph = build_graph(
            MockOpenF1Client(MONACO_2024),
            FakeIndex(["30.5"]),
            models={"router": router, "fetcher": fetcher, "analyst": analyst},
            checkpointer=memory_checkpointer(),
            today=date(2026, 9, 29),
        )
        return graph, lambda: None

    return TestClient(create_app(make_graph, make_transcriber=lambda: None))


def test_health():
    with _client(Scripted([]), Scripted([]), Scripted([])) as client:
        assert client.get("/api/health").json() == {"status": "ok"}


def test_chat_streams_thread_steps_and_answer_then_keeps_thread():
    router = Scripted(
        [
            RouteDecision(
                mode="race", year=2024, place="Monaco", focus="f", regulation_queries=["q"]
            ),
            RouteDecision(mode="race", focus="follow-up", regulation_queries=["q"]),
        ]
    )
    fetcher = Scripted(
        [
            AIMessage("", tool_calls=[_tool_call("get_tyre_stints", {"driver": "NOR"}, "1")]),
            AIMessage("done"),
            AIMessage("done"),
        ]
    )
    with _client(router, fetcher, Scripted([_analysis("30.5"), _analysis()])) as client:
        resp = client.post("/api/chat", json={"thread_id": None, "message": "Monaco 2024?"})
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _events(resp.text)

        kinds = [e for e, _ in events]
        assert kinds[0] == "thread" and kinds[-1] == "answer"
        nodes = [d["node"] for e, d in events if e == "step"]
        assert nodes == [
            "route",
            "resolve",
            "fetch",
            "tools",
            "fetch",
            "retrieve",
            "analyze",
            "synthesize",
        ]
        assert dict(events[1 + nodes.index("tools")][1])["detail"] == "ran 1 tool call"

        answer = events[-1][1]
        assert answer["mode"] == "race"
        assert answer["race"]["session_key"] == 9523
        assert answer["regs"] == {"season": 2024, "issue": 6}
        assert answer["citations"][0]["article"] == "30.5"
        assert [c["tool"] for c in answer["tool_calls"]] == [
            "key_race_events",
            "race_summary",
            "get_tyre_stints",
        ]
        assert "[30.5]" in answer["answer"]

        thread_id = events[0][1]["thread_id"]
        follow = _events(
            client.post("/api/chat", json={"thread_id": thread_id, "message": "And Piastri?"}).text
        )
        assert follow[0][1]["thread_id"] == thread_id
        assert follow[-1][1]["race"]["session_key"] == 9523


def test_chat_error_event():
    router = Scripted([])  # pop from empty list -> IndexError inside the graph
    with _client(router, Scripted([]), Scripted([])) as client:
        events = _events(client.post("/api/chat", json={"message": "hi"}).text)
    assert [e for e, _ in events] == ["thread", "error"]
    assert "IndexError" in events[-1][1]["message"]


def test_chat_rejects_empty_message():
    with _client(Scripted([]), Scripted([]), Scripted([])) as client:
        assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_live_replay_streams_monitor_events(monkeypatch):
    from tests.test_monitor import FakeSession, _msg, _race_start

    messages = [
        *_race_start(),
        _msg(90, "LapCount", {"CurrentLap": 20}),
        _msg(91, "TrackStatus", {"Status": "4"}),
    ]
    monkeypatch.setattr("src.api.ArchiveSession", lambda path: FakeSession(messages))
    with _client(Scripted([]), Scripted([]), Scripted([])) as client:
        resp = client.get("/api/live/replay", params={"path": "2099/test/race/", "speed": 1000})
    kinds = [e for e, _ in _events(resp.text)]
    assert {"track_status", "session", "snapshot"} <= set(kinds)
    assert "pit_calls" in kinds and kinds[-1] == "end"
    pit = next(d for e, d in _events(resp.text) if e == "pit_calls")
    assert {c["tla"] for c in pit["report"]["calls"]} == {"NOR", "VER"}
