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
  (+ whatever `on_neutralisation` returns, e.g. a `prediction` event: who wins from here)
  snapshot      {lap, total_laps, status, clock, drivers:[...]}      throttled by the driver
  track         {source, points, bounds, rotation?, corners?, marshal_sectors?}  once, first
  positions     {utc, cars: [{number, tla, x, y, status}]}           ~4/s wall clock (replay)
"""

import asyncio
import logging
import statistics
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from src.livetiming.archive import STATIC, STRATEGY_TOPICS, ArchiveSession, Message
from src.livetiming.circuits import Circuit, PitLoss, fetch_circuit
from src.livetiming.rejoin import losses, rejoin_table
from src.livetiming.snapshot import TRACK_STATUS, RaceSnapshot, build_snapshot
from src.livetiming.state import TimingState
from src.livetiming.strategy import pit_calls
from src.livetiming.track import latest_positions, track_outline
from src.weather.forecast import race_rain

NEUTRALISED = {"SAFETY_CAR", "VSC"}
log = logging.getLogger(__name__)
REPLAY_TOPICS = (*STRATEGY_TOPICS, "TeamRadio", "Position.z", "WeatherData")


def snapshot_event(
    snap: RaceSnapshot, pit_loss: PitLoss | None = None, lap_time_s: float | None = None
) -> dict[str, Any]:
    """`rejoin`: per running car, where it would come out if it pitted now (green / SC / VSC);
    `lap_time_s`: a typical current green lap, for drawing the field on one line by gap."""
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
                "team_colour": d.team_colour,
                "gap": d.gap_to_leader_s,
                "best_lap": d.best_lap_s,
                "knocked_out": d.knocked_out,
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
        "session_type": snap.session_type,
        "pit_loss": losses(pit_loss),
        "lap_time_s": lap_time_s,
        # Rejoin projections only in races (practice / qualifying gaps are lap-time deltas).
        "rejoin": rejoin_table(snap, pit_loss) if snap.is_race else {},
    }


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def weather_event(data: dict[str, Any]) -> dict[str, Any] | None:
    """`weather` event from a WeatherData message (published about once a minute)."""
    if not isinstance(data, dict) or "TrackTemp" not in data:
        return None
    return {
        "type": "weather",
        "air_c": _float(data.get("AirTemp")),
        "track_c": _float(data.get("TrackTemp")),
        "humidity": _float(data.get("Humidity")),
        "wind_ms": _float(data.get("WindSpeed")),
        "wind_dir": _float(data.get("WindDirection")),
        "raining": str(data.get("Rainfall", "0")) not in ("0", "", "False", "false"),
    }


def session_start_utc(info: dict[str, Any]) -> datetime | None:
    """SessionInfo StartDate is local circuit time; GmtOffset ("04:00:00" / "-05:00:00") maps it
    to UTC."""
    try:
        local = datetime.fromisoformat(str(info["StartDate"]))
        sign = -1 if str(info.get("GmtOffset", "")).startswith("-") else 1
        h, m, *_ = (int(x) for x in str(info.get("GmtOffset", "0:0")).lstrip("-").split(":"))
        return (local - sign * timedelta(hours=h, minutes=m)).replace(tzinfo=UTC)
    except (KeyError, ValueError):
        return None


def forecast_event(info: dict[str, Any]) -> dict[str, Any] | None:
    """`forecast` event: Open-Meteo rain chance over the session window (live sessions only:
    forecasts don't cover the past)."""
    start = session_start_utc(info)
    location = (info.get("Meeting") or {}).get("Location") or ""
    if start is None or not location:
        return None
    try:
        outlook = race_rain(location, start)
    except Exception as e:  # noqa: BLE001 — no forecast is fine; never break the feed
        log.info("no rain forecast for %s: %s", location, e)
        return None
    return {
        "type": "forecast",
        "location": location,
        "p_rain": outlook.p_rain,
        "mm": outlook.mm,
        "hourly": [{"utc": t, "p": p, "mm": mm} for t, p, mm in outlook.hourly],
    }


@dataclass(frozen=True)
class LapRecord:
    """One completed lap of one car, as seen live."""

    lap: int
    time_s: float | None
    compound: str | None
    tyre_age: int | None  # laps on the set at the end of this lap
    stint: int
    pit_in: bool
    pit_out: bool
    neutralised: bool  # an SC / VSC / red flag was out at some point during the lap


def _lap_seconds(value: str | None) -> float | None:
    """ "1:48.619" -> 108.619."""
    if not value:
        return None
    try:
        minutes, _, seconds = value.rpartition(":")
        return (int(minutes) * 60 if minutes else 0) + float(seconds)
    except ValueError:
        return None


class RaceMonitor:
    def __init__(self, radio_base_url: str | None = None, pit_loss: PitLoss | None = None) -> None:
        self.state = TimingState()
        self._radio_base = radio_base_url  # clip paths are relative to the session folder
        self.pit_loss = pit_loss  # measured for this circuit when known; else estimated
        self._status: str | None = None
        self._session_sent = False
        self._race_control_seen = 0
        self._radio_seen: set[str] = set()
        self._positions: tuple[str, dict] | None = None
        # Lap history (for in-race pace, degradation and tyre life): per car number.
        self.laps: dict[str, list[LapRecord]] = {}
        self._neutralised_since_lap: dict[str, bool] = {}
        self._in_pit_this_lap: dict[str, bool] = {}

    def positions_event(self) -> dict[str, Any] | None:
        """Newest car coordinates, or None before the first Position.z message."""
        if not self._positions:
            return None
        timestamp, entries = self._positions
        drivers = self.state.topics.get("DriverList", {})
        cars = [
            {
                "number": number,
                "tla": drivers.get(number, {}).get("Tla", number),
                "x": car.get("X"),
                "y": car.get("Y"),
                "status": car.get("Status"),
            }
            for number, car in entries.items()
            if isinstance(car, dict)
        ]
        return {"type": "positions", "utc": timestamp, "cars": cars}

    def snapshot(self) -> RaceSnapshot:
        return build_snapshot(self.state)

    def reference_lap_s(self) -> float | None:
        """Median of each running car's latest clean green lap: the current lap time."""
        latest = []
        for laps in self.laps.values():
            for lap in reversed(laps[-3:]):
                if lap.time_s and not (lap.neutralised or lap.pit_in or lap.pit_out):
                    latest.append(lap.time_s)
                    break
        return round(statistics.median(latest), 3) if len(latest) >= 5 else None

    def snapshot_event(self) -> dict[str, Any]:
        return snapshot_event(self.snapshot(), self.pit_loss, self.reference_lap_s())

    def starting_grid(self) -> dict[int, int]:
        """Official starting grid (car number -> slot, penalties applied) from TimingAppData
        GridPos: published when the race feed opens, about an hour before the start. Empty
        until most of the field has one, or outside a race."""
        if "race" not in str(self.state.topics.get("SessionInfo", {}).get("Name", "")).lower():
            return {}
        grid = {}
        for number, line in self.state.topics.get("TimingAppData", {}).get("Lines", {}).items():
            pos = line.get("GridPos") if isinstance(line, dict) else None
            if str(pos or "").isdigit() and int(pos) > 0 and str(number).isdigit():
                grid[int(number)] = int(pos)
        return grid if len(grid) >= 10 else {}

    def feed(self, message: Message) -> list[dict[str, Any]]:
        if message.topic == "Position.z":
            # High-frequency and huge: kept out of TimingState; the caller throttles emission.
            self._positions = latest_positions(message) or self._positions
            return []
        self.state.apply(message.topic, message.data, str(message.offset))
        events: list[dict[str, Any]] = []
        topics = self.state.topics
        if message.topic == "TrackStatus":
            status = TRACK_STATUS.get(str(message.data.get("Status")), "UNKNOWN")
            if status in ("SAFETY_CAR", "VSC", "VSC_ENDING", "RED_FLAG"):
                for car in self._neutralised_since_lap:
                    self._neutralised_since_lap[car] = True
                self._neutralised_now = True
            else:
                self._neutralised_now = False
        if message.topic == "TimingData":
            events.extend(self._record_laps(message.data))

        # Races wait for LapCount (the header carries the race distance); practice and
        # qualifying never publish it.
        info_type = topics.get("SessionInfo", {}).get("Type")
        if (
            not self._session_sent
            and "SessionInfo" in topics
            and ("LapCount" in topics or (info_type and info_type != "Race"))
        ):
            snap = self.snapshot()
            events.append(
                {
                    "type": "session",
                    "meeting": snap.meeting,
                    "session": snap.session,
                    "session_type": snap.session_type,
                    "total_laps": snap.total_laps,
                }
            )
            self._session_sent = True

        if message.topic == "TrackStatus":
            status = TRACK_STATUS.get(str(message.data.get("Status")), "UNKNOWN")
            if status != self._status:
                snap = self.snapshot()
                events.append({"type": "track_status", "status": status, "lap": snap.current_lap})
                # Pit calls only mean something in a race (practice runs VSC tests).
                if status in NEUTRALISED and self._status not in NEUTRALISED and snap.is_race:
                    report = pit_calls(snap, self.pit_loss)
                    events.append({"type": "pit_calls", "report": asdict(report)})
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

        if message.topic == "WeatherData" and (event := weather_event(message.data)):
            event["lap"] = self.snapshot().current_lap
            events.append(event)

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

    def _record_laps(self, update: dict) -> list[dict[str, Any]]:
        """Append each car's newly completed lap to `laps`; returns `lap` events."""
        events: list[dict[str, Any]] = []
        drivers = self.state.topics.get("DriverList", {})
        lines = self.state.topics.get("TimingData", {}).get("Lines", {})
        app = self.state.topics.get("TimingAppData", {}).get("Lines", {})
        for car, change in (update.get("Lines") or {}).items():
            if not isinstance(change, dict):
                continue
            line = lines.get(car, {})
            if change.get("InPit") or change.get("PitOut"):
                self._in_pit_this_lap[car] = True
            if "NumberOfLaps" not in change:
                continue
            stints = [s for s in (app.get(car, {}).get("Stints") or []) if s.get("Compound")]
            current = stints[-1] if stints else {}
            pitted = self._in_pit_this_lap.pop(car, False)
            last = line.get("LastLapTime") or {}
            history = self.laps.get(car) or []
            # A stop touches two laps (in the pit lane at the end of one and the start of the
            # next): the second of two pit laps in a row is the out-lap.
            out_lap = bool(line.get("PitOut")) or (pitted and bool(history) and history[-1].pit_in)
            record = LapRecord(
                lap=int(change["NumberOfLaps"]),
                time_s=_lap_seconds((line.get("LastLapTime") or {}).get("Value")),
                compound=current.get("Compound"),
                tyre_age=current.get("TotalLaps"),
                stint=len(stints),
                # lap 1: cars on the grid read as "in pit" before the start
                pit_in=(bool(line.get("InPit")) or pitted)
                and not out_lap
                and int(change["NumberOfLaps"]) > 1,
                pit_out=out_lap,
                neutralised=self._neutralised_since_lap.get(car, False)
                or getattr(self, "_neutralised_now", False),
            )
            self.laps.setdefault(car, []).append(record)
            self._neutralised_since_lap[car] = getattr(self, "_neutralised_now", False)
            if record.lap < 1:  # the grid state before the start, not a lap
                continue
            events.append(
                {
                    "type": "lap",
                    "number": car,
                    "tla": drivers.get(car, {}).get("Tla", car),
                    "lap": record.lap,
                    "time": round(record.time_s, 3) if record.time_s else None,
                    "compound": record.compound,
                    "tyre_age": record.tyre_age,
                    "pit_in": record.pit_in,
                    "pit_out": record.pit_out,
                    "neutralised": record.neutralised,
                    "personal_best": bool(last.get("PersonalFastest")),
                    "overall_best": bool(last.get("OverallFastest")),
                }
            )
        return events

    def driver_names(self, number: str) -> list[str]:
        """The driver's full name, to prime speech recognition with the right spelling."""
        d = self.state.topics.get("DriverList", {}).get(number, {})
        name = " ".join(p for p in (d.get("FirstName"), d.get("LastName")) if p)
        return [name] if name else []


def _circuit_for(messages: list[Message]) -> Circuit | None:
    """MultiViewer circuit data for the session's circuit key and season."""
    info = next((m.data for m in messages if m.topic == "SessionInfo"), None)
    try:
        key = int(info["Meeting"]["Circuit"]["Key"])
        year = int(str(info["StartDate"])[:4])
    except (TypeError, KeyError, ValueError):
        return None
    return fetch_circuit(key, year)


def _lap_count(message: Message) -> int | None:
    if message.topic == "LapCount":
        return message.data.get("CurrentLap")
    return None


class Transcriber(Protocol):
    def cached(self, url: str) -> str | None: ...
    async def transcribe(self, url: str, names: list[str] = ...) -> str | None: ...


async def pump(
    source: AsyncIterator[tuple[Message, bool]],
    monitor: RaceMonitor,
    snapshot_every_s: float = 1.0,
    transcriber: Transcriber | None = None,
    transcript_wait_s: float = 30.0,
    positions_every_s: float = 0.25,
    on_neutralisation: Callable[[RaceMonitor], dict[str, Any] | None] | None = None,
    load_circuit_live: bool = False,
    on_grid: Callable[[RaceMonitor], dict[str, Any] | None] | None = None,
    alerts: bool = False,
    on_q3: Callable[[RaceMonitor], dict[str, Any] | None] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Turn a stream of (message, quiet) into race events — shared by the live feed and the
    replay, so live runs the code the replays test.

    `quiet` messages are applied without emitting events (replay fast-forward), except
    `session`; when quiet ends a snapshot (and car positions) are sent straight away.
    A `snapshot` is emitted at most every `snapshot_every_s` of wall-clock time, plus right
    after SC/VSC pit calls. With a `transcriber`, `radio` events carry cached text or null and
    uncached clips are transcribed in the background (`radio_transcript` events), so the race
    feed never waits on speech recognition. `load_circuit_live` fetches circuit data (track map,
    measured pit loss) once SessionInfo arrives — the replay precomputes it instead.
    `on_grid` runs once when the official starting grid is known (before lap 1), e.g. for a
    pre-race prediction from the real grid. `on_q3` runs once when qualifying reaches Q3 (the
    pre-Q3 prediction). `alerts`: run the strategy alert engine (alerts.py)
    after each completed lap and at SC/VSC calls; it emits `alert` events."""
    loop = asyncio.get_running_loop()
    last_snapshot = last_positions = 0.0
    was_quiet = False
    circuit_done = not load_circuit_live
    grid_done = on_grid is None
    q3_done = on_q3 is None
    engine = None
    if alerts:
        from src.livetiming.alerts import AlertEngine
        from src.livetiming.director import Director

        engine = AlertEngine()
        director = Director()
    last_lap = None
    transcripts: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    pending: set[asyncio.Task] = set()

    async def write_alert(alert: dict[str, Any]) -> None:
        from src.agents.alert_writer import phrase

        text = await asyncio.to_thread(phrase, alert)
        if text:
            await transcripts.put({**alert, "text": text})

    def start_writing(alert: dict[str, Any]) -> None:
        from src.agents.alert_writer import enabled

        if enabled():
            task = asyncio.create_task(write_alert(alert))
            pending.add(task)
            task.add_done_callback(pending.discard)

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

    try:
        async for message, quiet in source:
            if was_quiet and not quiet:
                yield monitor.snapshot_event()
                if positions := monitor.positions_event():  # dots appear straight away
                    yield positions
            was_quiet = quiet

            events = monitor.feed(message)
            if not circuit_done and message.topic == "SessionInfo":
                circuit_done = True
                circuit = await asyncio.to_thread(_circuit_for, [message])
                if circuit:
                    monitor.pit_loss = circuit.pit_loss
                    events.insert(0, circuit.track_event())
                if forecast := await asyncio.to_thread(forecast_event, message.data):
                    events.append(forecast)
            if quiet:
                # Lap history still flows during fast-forward: the run timeline shows every lap.
                events = [e for e in events if e["type"] in ("session", "track", "lap")]
            for event in events:
                start_transcription(event)
                yield event
                if event["type"] == "pit_calls" and on_neutralisation and not quiet:
                    extra = await asyncio.to_thread(on_neutralisation, monitor)
                    if extra:
                        yield extra
            for event in drain():
                yield event
            if engine is not None and not quiet:
                for e in events:
                    if e["type"] == "pit_calls":
                        report = pit_calls(monitor.snapshot(), monitor.pit_loss)
                        for alert in engine.on_pit_calls(report):
                            start_writing(alert)
                            yield alert
                if message.topic == "LapCount":
                    lap = monitor.snapshot().current_lap
                    if lap and lap != last_lap:
                        last_lap = lap
                        loss = (
                            (monitor.pit_loss.green, monitor.pit_loss.safety_car)
                            if monitor.pit_loss
                            else (22.0, 13.5)
                        )
                        try:
                            new = await asyncio.to_thread(engine.on_lap, monitor, loss)
                        except Exception:
                            log.exception("alert engine failed on lap %s", lap)
                            new = []
                        for alert in new:
                            start_writing(alert)
                            yield alert
                        try:
                            picks = await asyncio.to_thread(director.on_lap, monitor)
                        except Exception:
                            log.exception("director failed on lap %s", lap)
                            picks = []
                        for event in picks:
                            if event["type"] == "alert":
                                start_writing(event)
                            yield event
            if (
                not q3_done
                and not quiet
                and message.topic == "TimingData"
                and str(message.data.get("SessionPart")) == "3"
                and monitor.snapshot().session == "Qualifying"
            ):
                q3_done = True
                if extra := await asyncio.to_thread(on_q3, monitor):
                    yield extra
            if (
                not grid_done
                and message.topic in ("TimingAppData", "LapCount")
                and monitor.starting_grid()
            ):
                grid_done = True
                before_start = (monitor.snapshot().current_lap or 0) <= 1
                if before_start and (extra := await asyncio.to_thread(on_grid, monitor)):
                    yield extra

            now = loop.time()
            if (
                message.topic == "Position.z"
                and not quiet
                and now - last_positions >= positions_every_s
                and (positions := monitor.positions_event())
            ):
                yield positions
                last_positions = now
            pit_call_made = any(e["type"] == "pit_calls" for e in events)
            if not quiet and (pit_call_made or now - last_snapshot >= snapshot_every_s):
                yield monitor.snapshot_event()
                last_snapshot = now

        yield monitor.snapshot_event()
        if pending:  # let clips from the last laps finish, within reason
            await asyncio.wait(set(pending), timeout=transcript_wait_s)
        for event in drain():
            yield event
        yield {"type": "end"}
    finally:
        for task in pending:
            task.cancel()


async def replay(
    session: ArchiveSession,
    speed: float = 10.0,
    from_lap: int | None = None,
    snapshot_every_s: float = 1.0,
    transcriber: Transcriber | None = None,
    transcript_wait_s: float = 30.0,
    positions_every_s: float = 0.25,
    circuit_loader: Callable[[list[Message]], Circuit | None] | None = None,
    monitor: RaceMonitor | None = None,
    on_neutralisation: Callable[[RaceMonitor], dict[str, Any] | None] | None = None,
    on_grid: Callable[[RaceMonitor], dict[str, Any] | None] | None = None,
    alerts: bool = False,
    on_q3: Callable[[RaceMonitor], dict[str, Any] | None] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Play an archived session as if live (a test source for the live tooling). Messages before
    `from_lap` are applied instantly; afterwards the original timing is kept, divided by
    `speed`. Event semantics are `pump`'s."""
    # A caller-owned monitor lets a live session answer questions about the race state while
    # this generator feeds it.
    if monitor is None:
        monitor = RaceMonitor()
    monitor._radio_base = f"{STATIC}/{session.path}"
    load_circuit = circuit_loader or _circuit_for

    # Downloading/parsing the archive is blocking I/O; keep it off the event loop.
    messages = await asyncio.to_thread(lambda: list(session.messages(REPLAY_TOPICS)))
    circuit = await asyncio.to_thread(load_circuit, messages)
    if circuit:
        monitor.pit_loss = circuit.pit_loss
        yield circuit.track_event()
    elif outline := await asyncio.to_thread(track_outline, messages):
        yield {"type": "track", "source": "traced", **outline}

    async def timed() -> AsyncIterator[tuple[Message, bool]]:
        fast_forward = from_lap is not None
        previous: timedelta | None = None
        for message in messages:
            if fast_forward and (_lap_count(message) or 0) >= from_lap:
                fast_forward = False
            if not fast_forward and previous is not None:
                wait = (message.offset - previous).total_seconds() / speed
                if wait > 0:
                    await asyncio.sleep(min(wait, 5.0))  # cap long quiet gaps (red flags)
            previous = message.offset
            yield message, fast_forward

    async for event in pump(
        timed(),
        monitor,
        snapshot_every_s=snapshot_every_s,
        transcriber=transcriber,
        transcript_wait_s=transcript_wait_s,
        positions_every_s=positions_every_s,
        on_neutralisation=on_neutralisation,
        on_grid=on_grid,
        alerts=alerts,
        on_q3=on_q3,
    ):
        yield event
