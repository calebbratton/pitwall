"""Pre-race prediction from the official starting grid, for the live race feed.

When the race feed opens (about an hour before lights out) it publishes each car's grid slot,
penalties included. `grid_prediction_event` runs the pre-race simulator from that grid and the
weekend's practice/qualifying (fetching them into the warehouse first if they're missing) and
returns a `prediction` event in the same shape as the in-race ones, with lap 0.
"""

import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

from src.livetiming.snapshot import _lap_s
from src.sim import prediction_log
from src.sim.calibrate import DEFAULT_CALIBRATION, apply
from src.sim.inputs import build_inputs, weekend_sessions
from src.sim.race import simulate
from src.sim.wet import simulate_weather
from src.warehouse.queries import connect

log = logging.getLogger(__name__)

WEEKEND_SESSIONS = (
    "Practice 1",
    "Practice 2",
    "Practice 3",
    "Sprint Qualifying",
    "Sprint",
    "Qualifying",
)


def _has_qualifying(year: int, place: str) -> bool:
    try:
        con = connect()
        try:
            return "Qualifying" in weekend_sessions(con, year, place)[1]
        finally:
            con.close()
    except Exception:  # noqa: BLE001 — no warehouse / unknown weekend
        return False


def ensure_weekend(year: int, place: str) -> bool:
    """Make sure this weekend's sessions are in the warehouse: ingest any missing ones from
    OpenF1 (free, ~30 min after each session ends) and rebuild. True if qualifying is there."""
    if _has_qualifying(year, place):
        return True
    from src.warehouse.build import build
    from src.warehouse.ingest import ingest

    new = ingest([year], list(WEEKEND_SESSIONS), settle=timedelta(minutes=30))
    if new:
        build()
    return _has_qualifying(year, place)


def race_laps(year: int, location: str) -> int | None:
    """Race distance before the race feed exists: reference/race_laps_<year>.json, else the
    previous season's race at the same location."""
    path = Path(f"reference/race_laps_{year}.json")
    if path.exists():
        table = {k.casefold(): v for k, v in json.loads(path.read_text()).items() if k[0] != "_"}
        if location.casefold() in table:
            return int(table[location.casefold()])
    try:
        con = connect()
        try:
            row = con.execute(
                """SELECT max(l.lap_number) FROM laps l JOIN races r USING (session_key)
                   WHERE r.session_name = 'Race' AND r.year = ? AND r.location ILIKE ?""",
                [year - 1, location],
            ).fetchone()
        finally:
            con.close()
        return int(row[0]) if row and row[0] else None
    except Exception:  # noqa: BLE001
        return None


def grid_prediction(
    year: int,
    place: str,
    laps: int,
    grid: dict[int, int] | None = None,
    sims: int = 5000,
    calibrated: bool = True,
    p_rain: float | None = None,
    quali: dict[int, float] | None = None,
    pole_s: float | None = None,
) -> dict[str, Any]:
    con = connect()
    try:
        inputs = build_inputs(con, year, place, laps=laps, grid=grid, quali=quali, pole_s=pole_s)
    finally:
        con.close()
    prediction = (
        simulate_weather(inputs, p_rain, sims=sims) if p_rain else simulate(inputs, sims=sims)
    )
    table = prediction.table()
    if calibrated:
        table = apply(table, DEFAULT_CALIBRATION, sims)
    slot = {d.tla: d.grid for d in inputs.drivers}
    notes = [
        "before the start: from qualifying pace and the official grid"
        if grid or inputs.race_session_key is not None  # a finished race: its real start order
        else "before the start: grid = qualifying order (no penalties known)",
        *inputs.notes,
        "every car finishes and no SC/VSC luck; probabilities calibrated on 2026 results",
    ]
    if p_rain:
        notes.append(f"{p_rain:.0%} chance of rain: that share of simulations run wet")
    return {
        "lap": 0,
        "laps_remaining": laps,
        "sims": sims,
        "table": [
            {
                "tla": row["tla"],
                "position": slot.get(row["tla"]),
                "expected": row["expected"],
                "p_win": row["p_win"],
                "p_podium": row["p_podium"],
                "p_points": row["p_points"],
                "compound": None,
                "compound_c": None,
                "tyre_age": None,
                "stopped": False,
                "tyre_risk": None,
                "tyre_note": "",
            }
            for row in sorted(table, key=lambda r: slot.get(r["tla"], 99))
        ],
        "notes": notes,
    }


def grid_prediction_event(monitor) -> dict[str, Any] | None:
    """`on_grid` hook for the live pump: a lap-0 `prediction` event, or None if the weekend's
    qualifying isn't available."""
    snapshot = monitor.snapshot()
    if not (snapshot.year and snapshot.location and snapshot.total_laps):
        return None
    # The feed's location name can differ from OpenF1's (e.g. a circuit vs a city name); the
    # country name is the fallback.
    meeting = monitor.state.topics.get("SessionInfo", {}).get("Meeting", {})
    country = (meeting.get("Country") or {}).get("Name") or ""
    try:
        place = next(
            (p for p in (snapshot.location, country) if p and ensure_weekend(snapshot.year, p)),
            None,
        )
        if place is None:
            log.warning("no qualifying data for %s %s", snapshot.location, snapshot.year)
            return None
        # No automatic rain mixing: the wet setting showed no out-of-sample gain (src/sim/wet.py).
        # The forecast is shown on its own (weather bar).
        prediction = grid_prediction(
            snapshot.year, place, snapshot.total_laps, grid=monitor.starting_grid()
        )
    except Exception:
        log.exception("pre-race prediction failed")
        return None
    prediction_log.save(snapshot.year, snapshot.location, "grid", prediction)
    return {"type": "prediction", "stage": "grid", "status": snapshot.track_status, **prediction}


def live_qualifying(monitor) -> tuple[dict[int, float], dict[int, int], float] | None:
    """(gap to the fastest lap so far, provisional grid slot, fastest lap) from live
    qualifying timing: best lap over the segments run so far, positions as shown."""
    lines = monitor.state.topics.get("TimingData", {}).get("Lines", {})
    best: dict[int, float] = {}
    grid: dict[int, int] = {}
    for number, line in lines.items():
        if not str(number).isdigit() or not isinstance(line, dict):
            continue
        times = [
            _lap_s((seg or {}).get("Value"))
            for seg in (line.get("BestLapTimes") or [])
            if isinstance(seg, dict)
        ] + [_lap_s((line.get("BestLapTime") or {}).get("Value"))]
        times = [t for t in times if t]
        if times:
            best[int(number)] = min(times)
        if str(line.get("Position", "")).isdigit():
            grid[int(number)] = int(line["Position"])
    if len(best) < 10 or len(grid) < 10:
        return None
    pole = min(best.values())
    # >7% off: aborted laps, not pace (as for archived qualifying)
    deltas = {n: t - pole for n, t in best.items() if t <= pole * 1.07}
    return deltas, grid, pole


def pre_q3_prediction_event(monitor) -> dict[str, Any] | None:
    """`on_q3` hook: the prediction made as Q3 starts (grid not final), logged once."""
    snapshot = monitor.snapshot()
    if snapshot.session != "Qualifying" or not (snapshot.year and snapshot.location):
        return None
    live = live_qualifying(monitor)
    laps = race_laps(snapshot.year, snapshot.location)
    if live is None or laps is None:
        log.warning("pre-Q3 prediction skipped: timing or race distance unknown")
        return None
    deltas, grid, pole = live
    meeting = monitor.state.topics.get("SessionInfo", {}).get("Meeting", {})
    country = (meeting.get("Country") or {}).get("Name") or ""
    try:
        ensure_weekend(snapshot.year, snapshot.location)  # practice sessions into the warehouse
        place = snapshot.location
        try:
            prediction = grid_prediction(
                snapshot.year, place, laps, grid=grid, quali=deltas, pole_s=pole
            )
        except LookupError:
            place = country
            prediction = grid_prediction(
                snapshot.year, place, laps, grid=grid, quali=deltas, pole_s=pole
            )
    except Exception:
        log.exception("pre-Q3 prediction failed")
        return None
    prediction["notes"] = [
        "before Q3: from Q1/Q2 times and the provisional order; the grid isn't final",
        *[n for n in prediction["notes"] if not n.startswith("before the start")],
    ]
    prediction_log.save(snapshot.year, snapshot.location, "pre-Q3", prediction)
    return {"type": "prediction", "stage": "pre-Q3", "status": snapshot.track_status, **prediction}
