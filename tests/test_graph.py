"""Graph wiring tests with scripted fake models: no network, no LLM, no tokens."""

from collections.abc import Iterable
from datetime import date
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from src.agents import graph as graph_module
from src.agents.graph import build_graph, memory_checkpointer
from src.agents.state import Analysis, Finding, RouteDecision
from src.rag.chunking import RegChunk
from src.rag.index import RetrievedClause
from src.tools.openf1 import MockOpenF1Client

MONACO_2024 = Path(__file__).parent / "fixtures/openf1/2024_monaco"


class Scripted:
    """Stands in for a chat model: returns queued outputs from invoke(), whether called
    directly, via bind_tools() or via with_structured_output()."""

    def __init__(self, outputs: Iterable):
        self.outputs = list(outputs)
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.outputs.pop(0)

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return self


class FakeIndex:
    def __init__(self, articles: list[str]):
        self.articles = articles
        self.queries: list[tuple] = []

    def search(self, query, season, issue, k=5):
        self.queries.append((query, season, issue))
        return [
            RetrievedClause(
                RegChunk(
                    f"{season}-iss{issue}-{a}-1", a, "TITLE", 1, f"text of {a}", season, issue
                ),
                1.0,
            )
            for a in self.articles
        ]


def _tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _analysis(*articles: str) -> Analysis:
    return Analysis(
        verdict="Verdict.",
        findings=[Finding(claim="Claim.", evidence=["78.403"], articles=list(articles))],
    )


def _build(router, fetcher=None, analyst=None, index=None, checkpointer=None):
    models = {
        "router": router,
        "fetcher": fetcher or Scripted([]),
        "analyst": analyst or Scripted([]),
    }
    return build_graph(
        MockOpenF1Client(MONACO_2024),
        index or FakeIndex(["30.5"]),
        models=models,
        checkpointer=checkpointer,
        today=date(2026, 9, 29),
    )


def _ask(graph, question: str, thread: str = "t"):
    return graph.invoke(
        {"messages": [HumanMessage(question)]}, {"configurable": {"thread_id": thread}}
    )


def test_rules_question_skips_telemetry_and_drops_unretrieved_citations():
    router = Scripted(
        [
            RouteDecision(
                mode="rules", focus="compound rule", regulation_queries=["tyre specifications"]
            )
        ]
    )
    analyst = Scripted([_analysis("B6.3.6", "B9.9.9")])
    index = FakeIndex(["B6.3.6"])
    graph = _build(router, analyst=analyst, index=index, checkpointer=memory_checkpointer())

    state = _ask(graph, "What is the two-compound rule?")

    assert state["reg_context"] == {"season": 2026, "issue": 8}
    assert state["fetched_telemetry_json"] == []
    assert {q[1:] for q in index.queries} == {(2026, 8)}
    answer = state["messages"][-1].content
    assert "[B6.3.6]" in answer and "B9.9.9" not in answer
    assert "2026 Sporting Regulations (Issue 8), Article B6.3.6" in answer
    assert any("dropped unretrieved citations ['B9.9.9']" in s for s in state["evaluation_steps"])


def test_race_question_fetches_events_runs_tools_and_uses_regs_in_force():
    router = Scripted(
        [
            RouteDecision(
                mode="race", year=2024, place="Monaco", focus="NOR tyres", regulation_queries=["q"]
            )
        ]
    )
    fetcher = Scripted(
        [
            AIMessage(
                "",
                tool_calls=[
                    _tool_call("get_tyre_stints", {"driver": "NOR"}, "1"),
                    _tool_call("get_tyre_stints", {"driver": "Nobody"}, "2"),
                    _tool_call("made_up_tool", {}, "3"),
                ],
            ),
            AIMessage("done"),
        ]
    )
    analyst = Scripted([_analysis("30.5")])
    graph = _build(router, fetcher, analyst, checkpointer=memory_checkpointer())

    state = _ask(graph, "What tyres did Norris use at Monaco 2024?")

    assert state["race_context"]["session_key"] == 9523
    assert state["reg_context"] == {"season": 2024, "issue": 6}  # Issue 7 came after the race
    results = state["fetched_telemetry_json"]
    assert [r["tool"] for r in results] == [
        "key_race_events",
        "get_tyre_stints",
        "get_tyre_stints",
        "made_up_tool",
    ]
    assert "RED FLAG" in results[0]["result"]
    assert '"compound":"HARD"' in results[1]["result"]
    assert results[2]["result"].startswith("ERROR: unknown driver")
    assert results[3]["result"].startswith("ERROR: unknown tool")
    # The analyst sees the telemetry it must ground its numbers in.
    assert "HARD" in analyst.calls[0][0].content


def test_fetch_loop_is_capped():
    router = Scripted(
        [RouteDecision(mode="race", year=2024, place="Monaco", focus="f", regulation_queries=["q"])]
    )
    endless = [
        AIMessage("", tool_calls=[_tool_call("list_drivers", {}, str(i))])
        for i in range(graph_module.MAX_FETCH_ROUNDS + 5)
    ]
    fetcher = Scripted(endless)
    graph = _build(router, fetcher, Scripted([_analysis()]))

    state = _ask(graph, "q")

    assert len(fetcher.calls) == graph_module.MAX_FETCH_ROUNDS
    assert state["messages"][-1].content.startswith("Verdict.")


def test_ambiguous_place_returns_options_without_calling_analyst():
    router = Scripted(
        [
            RouteDecision(
                mode="race", year=2024, place="United States", focus="f", regulation_queries=["q"]
            )
        ]
    )
    analyst = Scripted([])
    state = _ask(_build(router, analyst=analyst), "How did Austin go?")

    assert "Miami" in state["messages"][-1].content and "Las Vegas" in state["messages"][-1].content
    assert analyst.calls == []


def test_follow_up_keeps_previous_race():
    decisions = [
        RouteDecision(mode="race", year=2024, place="Monaco", focus="f", regulation_queries=["q"]),
        RouteDecision(mode="race", focus="Leclerc", regulation_queries=["q"]),  # no year/place
    ]
    graph = _build(
        Scripted(decisions),
        Scripted([AIMessage("done"), AIMessage("done")]),
        Scripted([_analysis(), _analysis()]),
        checkpointer=memory_checkpointer(),
    )

    _ask(graph, "Monaco 2024 tyres?")
    state = _ask(graph, "What about Leclerc?")

    assert state["race_context"]["session_key"] == 9523
    assert len([m for m in state["messages"] if isinstance(m, HumanMessage)]) == 2
