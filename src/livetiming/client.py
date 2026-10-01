"""Live F1 timing: a SignalR Core client for livetiming.formula1.com, plus a recorder.

The client yields the same `Message`s as the archive reader, so the live session runs through the
exact pipeline the replays test. Protocol (verified 2026-09-29, no login needed for timing):

  1. OPTIONS  /signalrcore/negotiate  -> load-balancer cookies (AWSALB, AWSALBCORS)
  2. POST     /signalrcore/negotiate?negotiateVersion=1 -> connectionToken
  3. WebSocket wss://livetiming.formula1.com/signalrcore?id=<token> (with the cookies)
  4. send {"protocol":"json","version":1}\\x1e, then invoke "Subscribe" with the topic list
  5. the Subscribe completion carries every topic's current state; then "feed" invocations
     arrive as [topic, data, utc]. `.z` topics are base64 raw-deflate JSON.

Position.z / CarData.z need an F1 TV subscription token. Policy (see CLAUDE.md): the user's token
is for their own local use only, and those topics are only requested when BOTH
F1TV_SUBSCRIPTION_TOKEN is set AND PITWALL_USE_F1TV_TOKEN=1.
"""

import asyncio
import json
import logging
import os
import random
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import websockets

from src.livetiming.archive import Message, decode_compressed

log = logging.getLogger(__name__)

BASE = "https://livetiming.formula1.com/signalrcore"
WS_BASE = "wss://livetiming.formula1.com/signalrcore"
RS = "\x1e"  # SignalR record separator
PUBLIC_TOPICS = (
    "Heartbeat",
    "SessionInfo",
    "SessionStatus",
    "SessionData",
    "DriverList",
    "TimingData",
    "TimingAppData",
    "TimingStats",
    "TrackStatus",
    "LapCount",
    "RaceControlMessages",
    "WeatherData",
    "ExtrapolatedClock",
    "TopThree",
    "TeamRadio",
    "PitLaneTimeCollection",
    "CurrentTyres",
)
TOKEN_TOPICS = ("Position.z", "CarData.z")
DEFAULT_RECORDINGS = Path("data/livetiming/recordings")


def token_topics_enabled() -> bool:
    """Token-gated topics: local, opt-in only (both the token and the switch)."""
    return bool(os.getenv("F1TV_SUBSCRIPTION_TOKEN", "").strip()) and os.getenv(
        "PITWALL_USE_F1TV_TOKEN", ""
    ).strip() in ("1", "true", "yes")


class LiveTimingClient:
    def __init__(
        self,
        topics: tuple[str, ...] | None = None,
        base: str = BASE,
        ws_base: str = WS_BASE,
        http: httpx.AsyncClient | None = None,
        max_backoff_s: float = 60.0,
    ) -> None:
        use_token = token_topics_enabled()
        self.topics = topics or (PUBLIC_TOPICS + (TOKEN_TOPICS if use_token else ()))
        self._token = os.getenv("F1TV_SUBSCRIPTION_TOKEN", "").strip() if use_token else ""
        self._base, self._ws_base = base, ws_base
        self._http = http
        self._max_backoff = max_backoff_s
        self.started = datetime.now(UTC)
        self.connected = False

    async def _negotiate(self) -> tuple[str, str]:
        http = self._http or httpx.AsyncClient(timeout=20)
        try:
            # The OPTIONS call only collects the load balancer's sticky-session cookies; the
            # client's cookie jar carries them into the negotiate, and they're forwarded to the
            # websocket so it lands on the same backend.
            await http.options(f"{self._base}/negotiate?negotiateVersion=1")
            resp = await http.post(f"{self._base}/negotiate?negotiateVersion=1")
            resp.raise_for_status()
            info = resp.json()
            cookies = dict(http.cookies)
            token = info.get("connectionToken") or info["connectionId"]
            return token, "; ".join(f"{k}={v}" for k, v in cookies.items())
        finally:
            if self._http is None:
                await http.aclose()

    def _message(self, topic: str, data, when: datetime | None = None) -> Message:
        if isinstance(data, str):  # compressed .z topic
            data = decode_compressed(data)
        offset = (when or datetime.now(UTC)) - self.started
        return Message(offset, topic, data if isinstance(data, dict) else {"_": data})

    async def _session(self) -> AsyncIterator[Message]:
        token, cookie = await self._negotiate()
        headers = {"Cookie": cookie} if cookie else {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        async with websockets.connect(
            f"{self._ws_base}?id={token}", additional_headers=headers, max_size=None
        ) as ws:
            await ws.send(json.dumps({"protocol": "json", "version": 1}) + RS)
            await ws.recv()  # handshake ack: "{}\x1e"
            await ws.send(
                json.dumps(
                    {
                        "type": 1,
                        "invocationId": "0",
                        "target": "Subscribe",
                        "arguments": [list(self.topics)],
                    }
                )
                + RS
            )
            self.connected = True
            pinger = asyncio.create_task(self._ping(ws))
            try:
                async for raw in ws:
                    for frame in filter(None, str(raw).split(RS)):
                        try:
                            messages = self._decode(json.loads(frame))
                        except Exception:
                            log.exception("skipping a live timing frame that failed to decode")
                            continue
                        for message in messages:
                            yield message
            finally:
                self.connected = False
                pinger.cancel()

    def _decode(self, msg: dict) -> list[Message]:
        kind = msg.get("type")
        if kind == 3:  # Subscribe completion: every topic's current state
            if msg.get("error"):
                log.warning("subscribe error: %s", msg["error"])
            return [self._message(t, d) for t, d in (msg.get("result") or {}).items() if d]
        if kind == 1 and msg.get("target") == "feed":
            topic, data, *rest = msg.get("arguments") or [None, None]
            when = None
            if rest and isinstance(rest[0], str):
                try:
                    when = datetime.fromisoformat(rest[0])
                except ValueError:
                    when = None
                if when is not None and when.tzinfo is None:
                    # Some feed timestamps carry no zone (seen 2026-10-01); the feed is UTC.
                    when = when.replace(tzinfo=UTC)
            return [self._message(topic, data, when)] if topic else []
        if kind == 7:  # close
            log.warning("server closed the connection: %s", msg.get("error"))
        return []

    async def _ping(self, ws) -> None:
        while True:
            await asyncio.sleep(15)
            await ws.send(json.dumps({"type": 6}) + RS)

    async def messages(self) -> AsyncIterator[Message]:
        """Messages forever, reconnecting with jittered exponential backoff. A reconnect
        re-subscribes, which re-sends the full state, so merged state self-heals."""
        backoff = 1.0
        while True:
            try:
                async for message in self._session():
                    backoff = 1.0
                    yield message
                log.warning("live timing connection ended; reconnecting")
            except (OSError, httpx.HTTPError, websockets.WebSocketException) as e:
                log.warning("live timing connection failed (%s); retrying in %.0fs", e, backoff)
            await asyncio.sleep(backoff * (0.5 + random.random()))
            backoff = min(backoff * 2, self._max_backoff)


class Recorder:
    """Write every message to disk in the archive's line format (`HH:MM:SS.mmm{json}`), one file
    per topic, so a recorded live session replays through ArchiveSession-like reading."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, object] = {}

    @staticmethod
    def _stamp(offset: timedelta) -> str:
        total = offset.total_seconds()
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

    def write(self, message: Message) -> None:
        f = self._files.get(message.topic)
        if f is None:
            f = (self.folder / f"{message.topic}.jsonStream").open("a", encoding="utf-8")
            self._files[message.topic] = f
        f.write(
            self._stamp(message.offset) + json.dumps(message.data, separators=(",", ":")) + "\n"
        )
        f.flush()

    def close(self) -> None:
        for f in self._files.values():
            f.close()


def recording_folder(root: Path = DEFAULT_RECORDINGS, now: datetime | None = None) -> Path:
    return root / (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H%M%SZ")
