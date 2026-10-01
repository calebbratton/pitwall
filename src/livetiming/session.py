"""Live race sessions: server-owned race state that outlives any browser tab.

On race day the source is the live F1 feed (a background task on the server); between race
weekends it's an archive replay (a test convenience). Either way the session owns a RaceMonitor,
broadcasts events to any number of subscribers (browser tabs), and the chat / prediction tools
query it by `live_id`. A late subscriber first gets a catch-up of the current state.
"""

import asyncio
import contextlib
import logging
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from src.livetiming.client import LiveTimingClient, Recorder, recording_folder
from src.livetiming.monitor import RaceMonitor, pump

log = logging.getLogger(__name__)
SESSION_TTL_S = 6 * 3600  # keep finished sessions around for post-race questions
QUEUE_SIZE = 2000


@dataclass
class LiveSession:
    live_id: str
    source: str  # "live" | "replay"
    label: str  # e.g. the archive path, or "livetiming.formula1.com"
    monitor: RaceMonitor = field(default_factory=RaceMonitor)
    started: float = field(default_factory=time.time)
    finished: float | None = None
    task: asyncio.Task | None = None
    recording: str | None = None
    _subscribers: set[asyncio.Queue] = field(default_factory=set)
    # Catch-up for late subscribers: latest of each state-like event + recent feed items.
    _latest: dict[str, dict] = field(default_factory=dict)
    _recent: dict[str, deque] = field(
        default_factory=lambda: {
            "pit_calls": deque(maxlen=10),
            "prediction": deque(maxlen=10),
            "race_control": deque(maxlen=60),
            "radio": deque(maxlen=40),
            "radio_transcript": deque(maxlen=40),
            "lap": deque(maxlen=2500),  # every car's laps, for the run timeline
        }
    )

    def hello(self) -> dict[str, Any]:
        return {"type": "live", "live_id": self.live_id, "source": self.source}

    def publish(self, event: dict[str, Any]) -> None:
        kind = event.get("type", "")
        if kind in self._recent:
            self._recent[kind].append(event)
        elif kind != "end":
            self._latest[kind] = event
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # a stalled tab: drop it rather than block the feed
                self._subscribers.discard(queue)

    def catch_up(self) -> list[dict[str, Any]]:
        """Current state for a new subscriber. Snapshot, track status and positions are built
        fresh from the monitor: the last *published* snapshot can predate the state (the feed's
        initial state arrives in one burst inside the snapshot throttle, and between sessions
        nothing new is published)."""
        events = [
            self._latest[k]
            for k in ("session", "track", "weather", "forecast")
            if k in self._latest
        ]
        snap = self.monitor.snapshot()
        if snap.track_status != "UNKNOWN":
            events.append(
                {"type": "track_status", "status": snap.track_status, "lap": snap.current_lap}
            )
        if snap.drivers:
            events.append(self.monitor.snapshot_event())
        if positions := self.monitor.positions_event():
            events.append(positions)
        for kind in ("lap", "race_control", "radio", "radio_transcript", "pit_calls", "prediction"):
            events.extend(self._recent[kind])
        return events

    async def subscribe(self) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._subscribers.add(queue)
        try:
            yield self.hello()
            for event in self.catch_up():
                yield event
            if self.finished is not None:  # joined after the end: nothing more will come
                yield {"type": "end"}
                return
            while True:
                event = await queue.get()
                yield event
                if event.get("type") == "end":
                    return
        finally:
            self._subscribers.discard(queue)

    @property
    def connected(self) -> bool:
        return bool(self.task and not self.task.done())


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, LiveSession] = {}
        self.current_live: LiveSession | None = None

    def create(self, source: str, label: str) -> LiveSession:
        self._expire()
        session = LiveSession(uuid.uuid4().hex[:12], source, label)
        self._sessions[session.live_id] = session
        return session

    def get(self, live_id: str) -> LiveSession | None:
        self._expire()
        return self._sessions.get(live_id)

    def start_live(
        self,
        client_factory: Callable[[], LiveTimingClient] | None = None,
        record: bool = True,
        **pump_kwargs: Any,
    ) -> LiveSession:
        """Start (or return the running) live-feed session as a server background task."""
        if self.current_live and self.current_live.connected:
            return self.current_live
        session = self.create("live", "livetiming.formula1.com")
        recorder = Recorder(recording_folder()) if record else None
        session.recording = str(recorder.folder) if recorder else None
        session.monitor._radio_base = None  # set from SessionInfo once known (see _run_live)
        # Resolved at call time (not as a default argument) so tests can substitute a fake.
        client = (client_factory or LiveTimingClient)()

        def next_session() -> None:
            # F1 moved on to a new session (FP1 -> FP2 -> ... on one connection): start clean
            # (new monitor, recording and subscription, so the feed sends the full new state).
            if self.current_live is session:
                self.start_live(client_factory, record, **pump_kwargs)

        session.task = asyncio.create_task(
            _run_live(
                session,
                client,
                recorder,
                pump_kwargs,
                on_new_session=next_session,
                on_failure=next_session,
            )
        )
        self.current_live = session
        return session

    async def stop_live(self) -> None:
        session = self.current_live
        if session and session.task:
            session.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await session.task

    def _expire(self) -> None:
        now = time.time()
        for live_id, s in list(self._sessions.items()):
            if s.finished and now - s.finished > SESSION_TTL_S:
                del self._sessions[live_id]


RESTART_AFTER_FAILURE_S = 5.0


class _NewSession(Exception):
    """The feed switched to another session (a different SessionInfo Key)."""


async def _run_live(
    session: LiveSession,
    client: LiveTimingClient,
    recorder: Recorder | None,
    pump_kwargs: dict,
    on_new_session: Callable[[], None] | None = None,
    on_failure: Callable[[], None] | None = None,
) -> None:
    session_key = None

    async def source():
        nonlocal session_key
        async for message in client.messages():
            if message.topic == "SessionInfo" and message.data.get("Key") is not None:
                key = message.data["Key"]
                if session_key is not None and key != session_key:
                    raise _NewSession(key)
                session_key = key
            if recorder:
                recorder.write(message)
            if message.topic == "SessionInfo" and not session.monitor._radio_base:
                path = message.data.get("Path")
                if path:  # radio clip paths are relative to the session's static folder
                    session.monitor._radio_base = f"https://livetiming.formula1.com/static/{path}"
            yield message, False

    new_session = failed = False
    try:
        async for event in pump(source(), session.monitor, load_circuit_live=True, **pump_kwargs):
            session.publish(event)
    except asyncio.CancelledError:
        raise
    except _NewSession as e:
        log.info("live feed moved to session %s: starting a new live session", e)
        new_session = True
    except Exception:
        log.exception("live session %s failed; restarting the feed", session.live_id)
        failed = True
    finally:
        session.finished = time.time()
        session.publish({"type": "end"})
        if recorder:
            recorder.close()
    if new_session and on_new_session:
        # After this task has finished, so the registry sees it as no longer connected.
        asyncio.get_running_loop().call_soon(on_new_session)
    elif failed and on_failure:
        # Never leave the server without a live feed: a crash mid-weekend would otherwise mean
        # nothing follows the next session.
        asyncio.get_running_loop().call_later(RESTART_AFTER_FAILURE_S, on_failure)
