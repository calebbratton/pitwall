"""Live race sessions: server-owned race state that outlives any browser tab.

On race day the source is the live F1 feed; between race weekends it's an archive replay (a test
convenience). Either way the session owns a RaceMonitor, and the chat / prediction tools query
that session by `live_id`.
"""

import time
import uuid
from dataclasses import dataclass, field

from src.livetiming.monitor import RaceMonitor

SESSION_TTL_S = 6 * 3600  # keep finished sessions around for post-race questions


@dataclass
class LiveSession:
    live_id: str
    source: str  # "live" | "replay"
    label: str  # e.g. the archive path, or the live meeting
    monitor: RaceMonitor = field(default_factory=RaceMonitor)
    started: float = field(default_factory=time.time)
    finished: float | None = None


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, LiveSession] = {}

    def create(self, source: str, label: str) -> LiveSession:
        self._expire()
        session = LiveSession(uuid.uuid4().hex[:12], source, label)
        self._sessions[session.live_id] = session
        return session

    def get(self, live_id: str) -> LiveSession | None:
        self._expire()
        return self._sessions.get(live_id)

    def _expire(self) -> None:
        now = time.time()
        for live_id, s in list(self._sessions.items()):
            if s.finished and now - s.finished > SESSION_TTL_S:
                del self._sessions[live_id]
