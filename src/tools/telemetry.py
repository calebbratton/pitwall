"""LLM-facing telemetry tools, bound to one race session.

Design constraints (Groq free tier: 8K tokens/minute per model):
- Outputs are compact JSON; lap-by-lap requests are capped, and `get_pace_summary` does the
  number-crunching in Python so the model reasons over a few statistics, not 78 rows.
- Tool arguments are validated, never trusted: models sometimes leak reasoning into arguments
  ("Liddle? Actually Lando Norris"). Invalid input returns an error message listing valid
  options so the model can retry, rather than raising.
"""

import json
import statistics
import unicodedata
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.tools import BaseTool, tool

from src.tools.models import Driver, Lap, RaceControlMessage, Session
from src.tools.openf1 import OpenF1Client

MAX_LAPS_PER_CALL = 20
MAX_RACE_CONTROL_MESSAGES = 40
# Laps slower than this multiple of the median are treated as non-representative
# (safety car, VSC, red flag, in-laps, traffic incidents) and excluded from pace statistics.
SLOW_LAP_FACTOR = 1.07


def _json(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"))


def _fold(s: str) -> str:
    decomposed = unicodedata.normalize("NFKD", s.strip().casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


Window = tuple[datetime, datetime, str]  # (start, end, kind)
# How long after "SAFETY CAR IN THIS LAP" / "VSC ENDING" laps are still unrepresentative
# (the SC's in-lap and the restart lap), as multiples of a typical lap / seconds.
SC_TAIL_LAPS = 1.5
VSC_TAIL_S = 15.0


def _dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def neutralised_windows(messages: list[RaceControlMessage], laps: list[Lap]) -> list[Window]:
    """Time windows when the race wasn't at racing speed: safety car, VSC, red flag (+ the
    restart). Laps overlapping them don't represent pace or tyre wear."""
    timed = [lap.lap_duration for lap in laps if lap.lap_duration]
    typical = statistics.median(timed) if timed else 90.0
    lap_starts = sorted(_dt(lap.date_start) for lap in laps if lap.date_start)
    windows: list[Window] = []
    open_since: dict[str, datetime] = {}
    for m in sorted(messages, key=lambda m: m.date):
        text, when = m.message.upper(), _dt(m.date)
        if "VIRTUAL SAFETY CAR DEPLOYED" in text:
            open_since["VSC"] = when
        elif "VIRTUAL SAFETY CAR ENDING" in text and "VSC" in open_since:
            windows.append((open_since.pop("VSC"), when + timedelta(seconds=VSC_TAIL_S), "VSC"))
        elif "SAFETY CAR DEPLOYED" in text:
            open_since["SC"] = when
        elif "SAFETY CAR IN THIS LAP" in text and "SC" in open_since:
            tail = timedelta(seconds=typical * SC_TAIL_LAPS)
            windows.append((open_since.pop("SC"), when + tail, "SC"))
        elif m.flag == "RED":
            # Until the end of the first lap after the restart (a standing/rolling start).
            restart = next((t for t in lap_starts if t > when), None)
            end = restart + timedelta(seconds=typical * 1.5) if restart else when
            windows.append((when, end, "RED"))
    for kind, since in open_since.items():  # never ended (e.g. finished under SC)
        windows.append((since, datetime.max.replace(tzinfo=UTC), kind))
    return windows


def _overlaps(lap: Lap, windows: list[Window]) -> bool:
    if not lap.date_start or not lap.lap_duration:
        return False
    start = _dt(lap.date_start)
    end = start + timedelta(seconds=lap.lap_duration)
    return any(start < w_end and end > w_start for w_start, w_end, _ in windows)


def representative_laps(laps: list[Lap], windows: list[Window] = ()) -> list[Lap]:
    """Timed green-flag laps: drops lap 1, pit-out laps, laps overlapping a SC/VSC/red-flag
    window, then laps > SLOW_LAP_FACTOR x the median of what's left (in-laps, traffic)."""
    timed = [
        lap
        for lap in laps
        if lap.lap_duration
        and lap.lap_number > 1
        and not lap.is_pit_out_lap
        and not _overlaps(lap, windows)
    ]
    if not timed:
        return []
    median = statistics.median(lap.lap_duration for lap in timed)
    return [lap for lap in timed if lap.lap_duration <= median * SLOW_LAP_FACTOR]


def pace_summary(laps: list[Lap], windows: list[Window] = ()) -> dict[str, Any]:
    clean = representative_laps(laps, windows)
    summary: dict[str, Any] = {
        "laps_requested": len(laps),
        "laps_used": len(clean),
        "laps_excluded": sorted({lap.lap_number for lap in laps} - {l.lap_number for l in clean}),
    }
    if not clean:
        return summary
    times = [lap.lap_duration for lap in clean]
    summary |= {
        "mean_s": round(statistics.fmean(times), 3),
        "median_s": round(statistics.median(times), 3),
        "best_s": min(times),
        "best_lap": min(clean, key=lambda lap: lap.lap_duration).lap_number,
    }
    if len(clean) >= 3:
        fit = statistics.linear_regression([lap.lap_number for lap in clean], times)
        # Positive = getting slower per lap. Includes fuel burn-off (makes cars faster), so it
        # understates true tyre degradation.
        summary["trend_s_per_lap"] = round(fit.slope, 4)
    return summary


_KEY_EVENT_WORDS = ("RED FLAG", "SAFETY CAR", "SUSPENDED", "RESUME", "CHEQUERED")


def key_race_events(client: OpenF1Client, session_key: int) -> str:
    """Red flags, safety cars, VSCs, suspensions and the finish: the events that reshape
    strategy. Fetched deterministically for every race question rather than left to the LLM."""
    events = [
        [m.lap_number, m.message]
        for m in client.get_race_control(session_key)
        if m.flag in ("RED", "CHEQUERED")
        or m.category == "SafetyCar"
        or any(word in m.message.upper() for word in _KEY_EVENT_WORDS)
    ]
    return _json(events)


def race_summary(client: OpenF1Client, session: Session) -> str:
    """Whole-field strategy table in ~2.5K chars: enough to compare every team's strategy
    without any lap-by-lap tool calls (which blow the free-tier token budget)."""
    sk = session.session_key
    drivers = {d.driver_number: d for d in client.get_drivers(sk)}
    results = {r.driver_number: r for r in client.get_results(sk)}
    grid: dict[int, int] = {}
    for p in sorted(client.get_positions(sk), key=lambda p: p.date):
        grid.setdefault(p.driver_number, p.position)
    stints: dict[int, list] = {}
    for st in client.get_stints(sk):
        stints.setdefault(st.driver_number, []).append(st)
    all_laps = client.get_all_laps(sk)
    windows = neutralised_windows(client.get_race_control(sk), all_laps)
    laps: dict[int, list[Lap]] = {}
    for lap in all_laps:
        laps.setdefault(lap.driver_number, []).append(lap)
    pit_time: dict[int, float] = {}
    for stop in client.get_pit_stops(sk):
        pit_time[stop.driver_number] = pit_time.get(stop.driver_number, 0) + (
            stop.lane_duration or 0
        )

    rows = []
    order = sorted(
        drivers, key=lambda n: results[n].position if n in results and results[n].position else 99
    )
    for n in order:
        d, r = drivers[n], results.get(n)
        finish = (
            None
            if not r
            else ("DNF" if r.dnf else "DNS" if r.dns else "DSQ" if r.dsq else r.position)
        )
        plan, pace = [], []
        for st in stints.get(n, []):
            if not st.compound or st.lap_start is None or st.lap_end is None:
                continue
            length = st.lap_end - st.lap_start + 1
            plan.append(f"{st.compound[0]}{length}")
            in_stint = [
                lap for lap in laps.get(n, []) if st.lap_start <= lap.lap_number <= st.lap_end
            ]
            clean = representative_laps(in_stint, windows)
            pace.append(
                round(statistics.median(l.lap_duration for l in clean), 2) if clean else None
            )
        start = grid.get(n)
        gained = start - finish if isinstance(finish, int) and start else None
        rows.append(
            [
                d.name_acronym,
                d.team_name,
                start,
                finish,
                gained,
                (r.points or 0) if r else 0,
                max(len(plan) - 1, 0),
                "-".join(plan),
                pace,
                round(pit_time.get(n, 0), 1),
            ]
        )
    return _json(
        {
            "columns": [
                "driver",
                "team",
                "grid",
                "finish",
                "places_gained",
                "points",
                "stops",
                "stints (compound initial + laps)",
                "median clean lap per stint (s)",
                "total pit lane time (s)",
            ],
            "rows": rows,
        }
    )


def build_telemetry_tools(client: OpenF1Client, session: Session) -> list[BaseTool]:
    sk = session.session_key
    drivers: list[Driver] = client.get_drivers(sk)
    windows_cache: list[list[Window]] = []

    def windows() -> list[Window]:
        if not windows_cache:
            windows_cache.append(
                neutralised_windows(client.get_race_control(sk), client.get_all_laps(sk))
            )
        return windows_cache[0]

    def resolve(driver: str) -> Driver | str:
        key = _fold(str(driver))
        for d in drivers:
            names = {str(d.driver_number), d.name_acronym, d.full_name, d.full_name.split()[-1]}
            if key in {_fold(n) for n in names}:
                return d
        valid = ", ".join(f"{d.name_acronym} (#{d.driver_number})" for d in drivers)
        return f"ERROR: unknown driver {driver!r}. Use a number, acronym or surname: {valid}"

    @tool
    def list_drivers(team: str | None = None) -> str:
        """List drivers in this race (number, acronym, name, team). Optionally filter by team
        name, e.g. "McLaren"."""
        rows = [d for d in drivers if not team or _fold(team) in _fold(d.team_name or "")]
        if not rows:
            teams = sorted({d.team_name for d in drivers if d.team_name})
            return f"ERROR: no team matching {team!r}. Teams: {', '.join(teams)}"
        return _json([[d.driver_number, d.name_acronym, d.full_name, d.team_name] for d in rows])

    @tool
    def get_tyre_stints(driver: str | None = None) -> str:
        """Tyre stints: compound, first and last lap, tyre age at stint start. One driver
        (number, acronym or surname) or all drivers if omitted."""
        number = None
        if driver:
            d = resolve(driver)
            if isinstance(d, str):
                return d
            number = d.driver_number
        stints = client.get_stints(sk, number)
        acronyms = {d.driver_number: d.name_acronym for d in drivers}
        return _json(
            [
                {
                    "driver": acronyms.get(s.driver_number, s.driver_number),
                    "stint": s.stint_number,
                    "compound": s.compound,
                    "laps": [s.lap_start, s.lap_end],
                    "tyre_age_at_start": s.tyre_age_at_start,
                }
                for s in stints
            ]
        )

    @tool
    def get_lap_times(driver: str, lap_start: int, lap_end: int) -> str:
        """Lap-by-lap times in seconds for one driver, at most 20 laps per call.
        Rows are [lap, lap_time_s, is_pit_out_lap]. Prefer get_pace_summary for long ranges."""
        d = resolve(driver)
        if isinstance(d, str):
            return d
        laps = client.get_laps(sk, d.driver_number, lap_start, lap_end)
        if len(laps) > MAX_LAPS_PER_CALL:
            # Don't waste a tool round on an error: answer with the summary instead.
            note = f"{len(laps)} laps exceeds {MAX_LAPS_PER_CALL}; returning pace summary instead"
            return _json(
                {"note": note, "driver": d.name_acronym, "laps": [lap_start, lap_end]}
                | pace_summary(laps, windows())
            )
        return _json([[lap.lap_number, lap.lap_duration, lap.is_pit_out_lap] for lap in laps])

    @tool
    def get_pace_summary(driver: str, lap_start: int, lap_end: int) -> str:
        """Pace statistics for one driver over a lap range: mean, median, best lap, and lap-time
        trend (seconds per lap; positive = slowing, i.e. degradation). Excludes lap 1, pit-out
        laps, safety car / VSC / red flag laps, and laps >7% slower than the median."""
        d = resolve(driver)
        if isinstance(d, str):
            return d
        laps = client.get_laps(sk, d.driver_number, lap_start, lap_end)
        return _json(
            {"driver": d.name_acronym, "laps": [lap_start, lap_end]} | pace_summary(laps, windows())
        )

    @tool
    def get_race_control(
        lap_start: int | None = None, lap_end: int | None = None, category: str | None = None
    ) -> str:
        """Race control messages: flags, safety car, VSC, red flags, penalties, pit exit status.
        Optional lap range and category ("Flag", "SafetyCar", "Other"). Blue flags omitted."""
        msgs = [
            m
            for m in client.get_race_control(sk, lap_start, lap_end, category)
            if m.flag != "BLUE" and m.category != "Drs"
        ]
        rows = [[m.lap_number, m.category, m.flag, m.message] for m in msgs]
        if len(rows) > MAX_RACE_CONTROL_MESSAGES:
            note = f"TRUNCATED: {len(rows)} messages; narrow the lap range or category."
            return _json(rows[:MAX_RACE_CONTROL_MESSAGES] + [note])
        return _json(rows)

    return [list_drivers, get_tyre_stints, get_lap_times, get_pace_summary, get_race_control]
