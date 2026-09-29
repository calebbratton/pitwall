"""Turn a stream of feed messages into race events for the UI / terminal.

`RaceMonitor.feed(message)` is synchronous and deterministic (easy to test); `replay()` drives
it from the archive at any speed. The live WebSocket client will drive the same monitor.

Events (dicts with a "type"):
  session       {meeting, session, total_laps}                       once, when known
  track_status  {status, lap}                                        on every change
  race_control  {lap, category, message}                             each new message
  radio         {driver, tla, url, utc, text}                        each new team radio clip
  radio_transcript {url, tla, text}                                   when a clip's text is ready
  pit_calls     {report}                                             when a SC / VSC starts
  snapshot      {lap, total_laps, status, clock, drivers:[...]}      throttled by the driver
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import asdict
from datetime import timedelta
from typing import Any, Protocol

from src.livetiming.archive import STATIC, STRATEGY_TOPICS, ArchiveSession, Message
from src.livetiming.snapshot import TRACK_STATUS, RaceSnapshot, build_snapshot
from src.livetiming.state import TimingState
from src.livetiming.strategy import pit_calls

NEUTRALISED = {"SAFETY_CAR", "VSC"}
REPLAY_TOPICS = (*STRATEGY_TOPICS, "TeamRadio")


def snapshot_event(snap: RaceSnapshot) -> dict[str, Any]:
    return {
        "type": "snapshot",
        "lap": snap.current_lap,
        "total_laps": snap.total_laps,
        "status": snap.track_status,
        "clock": snap.clock,
        "drivers": [
            {
                "position": d.position,
                "tla": d.tla,
                "number": d.number,
                "team": d.team,
                "gap": d.gap_to_leader_s,
                "laps_down": d.laps_down,
                "interval": d.interval_s,
                "compound": d.compound,
                "tyre_age": d.tyre_age_laps,
                "pit_stops": d.pit_stops,
                "in_pit": d.in_pit,
                "retired": d.retired,
            }
            for d in snap.drivers
        ],
    }


class RaceMonitor:
    def __init__(self, radio_base_url: str | None = None) -> None:
        self.state = TimingState()
        self._radio_base = radio_base_url  # clip paths are relative to the session folder
        self._status: str | None = None
        self._session_sent = False
        self._race_control_seen = 0
        self._radio_seen: set[str] = set()

    def snapshot(self) -> RaceSnapshot:
        return build_snapshot(self.state)

    def feed(self, message: Message) -> list[dict[str, Any]]:
        self.state.apply(message.topic, message.data, str(message.offset))
        events: list[dict[str, Any]] = []
        topics = self.state.topics

        if not self._session_sent and "SessionInfo" in topics and "LapCount" in topics:
            snap = self.snapshot()
            events.append(
                {
                    "type": "session",
                    "meeting": snap.meeting,
                    "session": snap.session,
                    "total_laps": snap.total_laps,
                }
            )
            self._session_sent = True

        if message.topic == "TrackStatus":
            status = TRACK_STATUS.get(str(message.data.get("Status")), "UNKNOWN")
            if status != self._status:
                snap = self.snapshot()
                events.append({"type": "track_status", "status": status, "lap": snap.current_lap})
                if status in NEUTRALISED and self._status not in NEUTRALISED:
                    events.append({"type": "pit_calls", "report": asdict(pit_calls(snap))})
                self._status = status

        if message.topic == "RaceControlMessages":
            messages = topics["RaceControlMessages"].get("Messages") or []
            for rc in messages[self._race_control_seen :]:
                if isinstance(rc, dict) and rc.get("Message"):
                    events.append(
                        {
                            "type": "race_control",
                            "lap": rc.get("Lap"),
                            "category": rc.get("Category"),
                            "message": rc["Message"],
                        }
                    )
            self._race_control_seen = len(messages)

        if message.topic == "TeamRadio":
            captures = topics["TeamRadio"].get("Captures") or []
            drivers = topics.get("DriverList", {})
            for clip in captures:
                path = clip.get("Path") if isinstance(clip, dict) else None
                if not path or path in self._radio_seen:
                    continue
                self._radio_seen.add(path)
                number = str(clip.get("RacingNumber"))
                events.append(
                    {
                        "type": "radio",
                        "driver": number,
                        "tla": drivers.get(number, {}).get("Tla", number),
                        "utc": clip.get("Utc"),
                        "url": f"{self._radio_base}{path}" if self._radio_base else path,
                        "text": None,
                    }
                )
        return events

    def driver_names(self, number: str) -> list[str]:
        """The driver's full name, to prime speech recognition with the right spelling."""
        d = self.state.topics.get("DriverList", {}).get(number, {})
        name = " ".join(p for p in (d.get("FirstName"), d.get("LastName")) if p)
        return [name] if name else []


def _lap_count(message: Message) -> int | None:
    if message.topic == "LapCount":
        return message.data.get("CurrentLap")
    return None


class Transcriber(Protocol):
    def cached(self, url: str) -> str | None: ...
    async def transcribe(self, url: str, names: list[str] = ...) -> str | None: ...


async def replay(
    session: ArchiveSession,
    speed: float = 10.0,
    from_lap: int | None = None,
    snapshot_every_s: float = 1.0,
    transcriber: Transcriber | None = None,
    transcript_wait_s: float = 30.0,
) -> AsyncIterator[dict[str, Any]]:
    """Play an archived session as if live. Messages before `from_lap` are applied instantly
    (their events are dropped, except `session`); afterwards the original timing is kept,
    divided by `speed`. A `snapshot` event is emitted at most every `snapshot_every_s` of
    wall-clock time, plus immediately after SC/VSC pit calls.

    With a `transcriber`, each `radio` event carries its cached text or null; uncached clips are
    transcribed in the background and delivered as `radio_transcript` events, so the race feed
    never waits on speech recognition."""
    monitor = RaceMonitor(radio_base_url=f"{STATIC}/{session.path}")
    fast_forward = from_lap is not None
    previous: timedelta | None = None
    loop = asyncio.get_running_loop()
    last_snapshot = 0.0
    transcripts: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    pending: set[asyncio.Task] = set()

    async def transcribe(radio: dict[str, Any]) -> None:
        text = await transcriber.transcribe(radio["url"], monitor.driver_names(radio["driver"]))
        if text is not None:
            await transcripts.put(
                {"type": "radio_transcript", "url": radio["url"], "tla": radio["tla"], "text": text}
            )

    def start_transcription(event: dict[str, Any]) -> None:
        if transcriber is None or event["type"] != "radio":
            return
        cached = transcriber.cached(event["url"])
        if cached is not None:
            event["text"] = cached
            return
        task = asyncio.create_task(transcribe(event))
        pending.add(task)
        task.add_done_callback(pending.discard)

    def drain() -> list[dict[str, Any]]:
        ready = []
        while not transcripts.empty():
            ready.append(transcripts.get_nowait())
        return ready

    # Downloading/parsing the archive is blocking I/O; keep it off the event loop.
    messages = await asyncio.to_thread(lambda: list(session.messages(REPLAY_TOPICS)))
    try:
        for message in messages:
            if fast_forward and (_lap_count(message) or 0) >= from_lap:
                fast_forward = False
                yield snapshot_event(monitor.snapshot())
            if not fast_forward and previous is not None:
                wait = (message.offset - previous).total_seconds() / speed
                if wait > 0:
                    await asyncio.sleep(min(wait, 5.0))  # cap long quiet gaps (e.g. red flags)
            previous = message.offset

            events = monitor.feed(message)
            if fast_forward:
                events = [e for e in events if e["type"] == "session"]
            for event in events:
                start_transcription(event)
                yield event
            for event in drain():
                yield event

            now = loop.time()
            pit_call_made = any(e["type"] == "pit_calls" for e in events)
            if not fast_forward and (pit_call_made or now - last_snapshot >= snapshot_every_s):
                yield snapshot_event(monitor.snapshot())
                last_snapshot = now

        yield snapshot_event(monitor.snapshot())
        if pending:  # let clips from the last laps finish, within reason
            await asyncio.wait(set(pending), timeout=transcript_wait_s)
        for event in drain():
            yield event
        yield {"type": "end"}
    finally:
        for task in pending:
            task.cancel()
