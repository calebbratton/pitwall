"""Pre-race prediction from the official starting grid, for the live race feed.

When the race feed opens (about an hour before lights out) it publishes each car's grid slot,
penalties included. `grid_prediction_event` runs the pre-race simulator from that grid and the
weekend's practice/qualifying (fetching them into the warehouse first if they're missing) and
returns a `prediction` event in the same shape as the in-race ones, with lap 0.
"""

import logging
from datetime import timedelta
from typing import Any

from src.livetiming.monitor import forecast_event
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


def grid_prediction(
    year: int,
    place: str,
    laps: int,
    grid: dict[int, int] | None = None,
    sims: int = 5000,
    calibrated: bool = True,
    p_rain: float | None = None,
) -> dict[str, Any]:
    con = connect()
    try:
        inputs = build_inputs(con, year, place, laps=laps, grid=grid)
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
        if grid
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
    try:
        if not ensure_weekend(snapshot.year, snapshot.location):
            log.warning("no qualifying data for %s %s", snapshot.location, snapshot.year)
            return None
        forecast = forecast_event(monitor.state.topics.get("SessionInfo", {}))
        prediction = grid_prediction(
            snapshot.year,
            snapshot.location,
            snapshot.total_laps,
            grid=monitor.starting_grid(),
            p_rain=forecast["p_rain"] if forecast else None,
        )
    except Exception:
        log.exception("pre-race prediction failed")
        return None
    return {"type": "prediction", "status": snapshot.track_status, **prediction}
