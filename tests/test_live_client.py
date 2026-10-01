import asyncio
import base64
import json
import zlib
from datetime import timedelta

import httpx
import websockets

from src.livetiming.archive import ArchiveSession, Message
from src.livetiming.client import RS, LiveTimingClient, Recorder, token_topics_enabled
from src.livetiming.session import SessionRegistry


def _z(obj) -> str:
    """Encode like the feed's .z topics: raw deflate + base64."""
    c = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return base64.b64encode(c.compress(json.dumps(obj).encode()) + c.flush()).decode()


async def _fake_signalr(ws):
    """Minimal SignalR Core hub: handshake, Subscribe completion with state, then feed."""
    assert json.loads((await ws.recv()).rstrip(RS)) == {"protocol": "json", "version": 1}
    await ws.send("{}" + RS)
    invoke = json.loads((await ws.recv()).rstrip(RS))
    assert invoke["target"] == "Subscribe" and "TimingData" in invoke["arguments"][0]
    state = {"TrackStatus": {"Status": "1"}, "LapCount": {"CurrentLap": 3, "TotalLaps": 50}}
    await ws.send(json.dumps({"type": 3, "invocationId": "0", "result": state}) + RS)
    feed = [
        ["TrackStatus", {"Status": "4", "Message": "SCDeployed"}, "2026-10-04T08:00:00Z"],
        [
            "Position.z",
            _z({"Position": [{"Timestamp": "t", "Entries": {}}]}),
            "2026-10-04T08:00:01Z",
        ],
    ]
    frames = [json.dumps({"type": 1, "target": "feed", "arguments": a}) for a in feed]
    await ws.send(RS.join(frames) + RS)  # several messages in one websocket frame
    await ws.send(json.dumps({"type": 6}) + RS)  # ping
    await asyncio.sleep(0.2)


def _negotiate_transport():
    def handler(request):
        if request.method == "OPTIONS":
            return httpx.Response(405, headers={"set-cookie": "AWSALB=abc; Path=/"})
        return httpx.Response(200, json={"negotiateVersion": 1, "connectionToken": "tok"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_client_speaks_signalr_and_decodes_state_feed_and_compressed_topics():
    async def run():
        async with websockets.serve(_fake_signalr, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            client = LiveTimingClient(
                base="http://negotiate.test",
                ws_base=f"ws://127.0.0.1:{port}",
                http=_negotiate_transport(),
            )
            got = []
            async for message in client.messages():
                got.append(message)
                if len(got) == 4:
                    break
            return got

    got = asyncio.run(run())
    assert [(m.topic, m.data) for m in got[:3]] == [
        ("TrackStatus", {"Status": "1"}),
        ("LapCount", {"CurrentLap": 3, "TotalLaps": 50}),
        ("TrackStatus", {"Status": "4", "Message": "SCDeployed"}),
    ]
    assert got[3].topic == "Position.z" and got[3].data["Position"][0]["Timestamp"] == "t"


def test_token_topics_need_token_and_opt_in(monkeypatch):
    monkeypatch.setenv("F1TV_SUBSCRIPTION_TOKEN", "secret")
    monkeypatch.delenv("PITWALL_USE_F1TV_TOKEN", raising=False)
    assert not token_topics_enabled()
    assert "Position.z" not in LiveTimingClient().topics
    monkeypatch.setenv("PITWALL_USE_F1TV_TOKEN", "1")
    assert token_topics_enabled() and "Position.z" in LiveTimingClient().topics


def test_recording_replays_through_the_archive_reader(tmp_path):
    rec = Recorder(tmp_path / "2026-10-04T0700Z")
    rec.write(Message(timedelta(seconds=1.5), "TrackStatus", {"Status": "1"}))
    rec.write(Message(timedelta(seconds=62.25), "TrackStatus", {"Status": "4"}))
    rec.write(Message(timedelta(seconds=3), "LapCount", {"CurrentLap": 1}))
    rec.close()
    messages = list(
        ArchiveSession("2026-10-04T0700Z", cache_dir=tmp_path).messages(("TrackStatus", "LapCount"))
    )
    assert [(m.offset.total_seconds(), m.topic, m.data) for m in messages] == [
        (1.5, "TrackStatus", {"Status": "1"}),
        (3.0, "LapCount", {"CurrentLap": 1}),
        (62.25, "TrackStatus", {"Status": "4"}),
    ]


class FakeLiveClient:
    def __init__(self, messages):
        self._messages = messages

    async def messages(self):
        for m in self._messages:
            await asyncio.sleep(0)
            yield m


def test_live_session_broadcasts_and_late_subscribers_catch_up(tmp_path, monkeypatch):
    from tests.test_api import _race_with_laps

    monkeypatch.setattr("src.livetiming.session.recording_folder", lambda: tmp_path / "rec")

    async def run():
        registry = SessionRegistry()
        session = registry.start_live(lambda: FakeLiveClient(_race_with_laps()), record=True)
        assert registry.start_live(lambda: FakeLiveClient([])) is session  # idempotent
        await session.task
        late = [e async for e in session.subscribe()]
        return session, late

    session, late = asyncio.run(run())
    kinds = [e["type"] for e in late]
    assert kinds[0] == "live" and late[0]["source"] == "live"
    assert {"session", "track_status", "snapshot", "pit_calls"} <= set(kinds)
    assert kinds[-1] == "end" and session.finished is not None
    assert (tmp_path / "rec" / "TimingData.jsonStream").exists()


def test_catch_up_snapshot_reflects_current_state_not_last_published():
    """The feed's initial state arrives in one burst inside the snapshot throttle, so the last
    published snapshot can be empty; a new subscriber must still get the real current state."""
    from tests.test_api import _race_with_laps

    registry = SessionRegistry()
    session = registry.create("live", "test")
    session.publish({"type": "snapshot", "lap": None, "drivers": []})  # stale, empty
    for m in _race_with_laps():
        session.monitor.feed(m)
    events = {e["type"]: e for e in session.catch_up()}
    assert events["snapshot"]["lap"] == 7 and len(events["snapshot"]["drivers"]) == 2
    assert events["track_status"]["status"] == "SAFETY_CAR"


def test_new_session_on_the_feed_starts_a_fresh_live_session(tmp_path, monkeypatch):
    from datetime import timedelta

    from src.livetiming.archive import Message
    from src.livetiming.session import SessionRegistry

    folders = iter([tmp_path / "rec1", tmp_path / "rec2"])
    monkeypatch.setattr("src.livetiming.session.recording_folder", lambda: next(folders))

    def msg(t, topic, data):
        return Message(timedelta(seconds=t), topic, data)

    fp1 = {"Key": 1, "Name": "Practice 1", "Type": "Practice", "Meeting": {"Name": "Test GP"}}
    fp2 = {"Key": 2, "Name": "Practice 2", "Type": "Practice", "Meeting": {"Name": "Test GP"}}
    feeds = iter(
        [
            [
                msg(0, "SessionInfo", fp1),
                msg(1, "DriverList", {"4": {"Tla": "NOR"}}),
                msg(2, "TimingData", {"Lines": {"4": {"Position": "1"}}}),
                msg(3, "SessionInfo", fp2),  # F1 moves on to FP2 on the same connection
            ],
            [msg(0, "SessionInfo", fp2), msg(1, "DriverList", {"1": {"Tla": "VER"}})],
        ]
    )

    async def run():
        registry = SessionRegistry()
        first = registry.start_live(client_factory=lambda: FakeLiveClient(next(feeds)))
        await first.task
        for _ in range(20):  # let the scheduled restart run
            await asyncio.sleep(0)
            if registry.current_live is not first and registry.current_live.task.done():
                break
        second = registry.current_live
        await second.task
        return first, second

    first, second = asyncio.run(run())
    assert second is not first and first.finished
    assert second.monitor.snapshot().session == "Practice 2"
    assert "4" not in second.monitor.state.topics.get("DriverList", {})  # nothing from FP1
    assert second.recording.endswith("rec2")


def test_feed_timestamps_without_a_zone_are_utc():
    from src.livetiming.client import LiveTimingClient

    client = LiveTimingClient()
    feed = {
        "type": 1,
        "target": "feed",
        "arguments": ["TrackStatus", {"Status": "1"}, "2026-10-01T06:18:22.5"],
    }
    aware = {**feed, "arguments": ["TrackStatus", {"Status": "1"}, "2026-10-01T06:18:22.5Z"]}
    [naive_msg] = client._decode(feed)
    [aware_msg] = client._decode(aware)
    assert naive_msg.offset == aware_msg.offset


def test_a_crashed_live_session_restarts(tmp_path, monkeypatch):
    from datetime import timedelta

    from src.livetiming.archive import Message
    from src.livetiming.session import SessionRegistry

    monkeypatch.setattr("src.livetiming.session.recording_folder", lambda: tmp_path / "rec")
    monkeypatch.setattr("src.livetiming.session.RESTART_AFTER_FAILURE_S", 0.0)

    class Crashing:
        async def messages(self):
            yield Message(timedelta(0), "SessionInfo", {"Key": 1, "Name": "Practice 1"})
            raise RuntimeError("unexpected feed data")

    clients = iter([Crashing(), FakeLiveClient([])])

    async def run():
        registry = SessionRegistry()
        first = registry.start_live(client_factory=lambda: next(clients))
        await first.task
        for _ in range(50):
            await asyncio.sleep(0.01)
            if registry.current_live is not first:
                break
        return first, registry.current_live

    first, second = asyncio.run(run())
    assert second is not first and first.finished
