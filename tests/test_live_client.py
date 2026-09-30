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
