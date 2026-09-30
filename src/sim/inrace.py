"""Who wins from here: simulate the rest of a race from its current state.

Built for live races (the replay is only a test source). From the live monitor we take the
running order and gaps, each car's tyres (compound, age, stops made, second compound still owed)
and its lap history, then simulate the remaining laps many times:

  pace       each car's recent clean race laps (tyre-age corrected), relative to the field
  tyres      degradation measured in this race so far (defaults until there's enough data);
             beyond a compound's long-stint length (2026: 90th percentile of stints) lap times
             fall off increasingly — the "can they make it to the end?" question
  strategy   cars that still owe the second compound must stop; cars whose tyres can't reach
             the flag stop before they fall off; others stay out
  SC now     the field is compressed and the race resumes from the queue
"""

import statistics
from dataclasses import dataclass, field

import numpy as np

from src.livetiming.monitor import LapRecord, RaceMonitor
from src.livetiming.snapshot import RaceSnapshot
from src.models.compounds import c_number
from src.models.tyres import CleanLap, fit_tyre_model
from src.sim.race import SimParams, traffic_step

DEFAULT_DEG = {"SOFT": 0.09, "MEDIUM": 0.06, "HARD": 0.045}
# Long-stint length per compound: 90th percentile of 2026 race stints that ended in a stop.
DEFAULT_LIFE = {"SOFT": 30, "MEDIUM": 30, "HARD": 35, "INTERMEDIATE": 30, "WET": 40}
CLIFF_S_PER_LAP2 = 0.08  # extra time per lap, per lap beyond the long-stint length
RECENT_LAPS = 8
MIN_LAPS_FOR_PREDICTION = 3


@dataclass(frozen=True)
class CarState:
    number: str
    tla: str
    position: int
    gap_s: float  # to the leader; lapped cars get a large gap
    compound: str | None
    tyre_age: int
    stops: int
    owes_compound: bool
    pace_s: float  # recent race pace vs the field median (s/lap, lower = faster)
    compound_c: str | None = None  # Pirelli compound for this weekend, e.g. "C4"


@dataclass(frozen=True)
class RaceState:
    lap: int
    laps_remaining: int
    status: str
    cars: list[CarState]
    deg: dict[str, float]
    life: dict[str, int]
    pit_loss_green: float
    pit_loss_sc: float
    notes: list[str] = field(default_factory=list)


class NotEnoughData(ValueError):
    pass


def _clean(record: LapRecord) -> bool:
    return (
        record.time_s is not None
        and record.lap > 1
        and not record.neutralised
        and not record.pit_in
        and not record.pit_out
        and record.compound is not None
        and record.tyre_age is not None
    )


def race_state(
    monitor: RaceMonitor,
    snapshot: RaceSnapshot,
    life: dict[str, int] | None = None,
    pit_loss: tuple[float, float] = (22.0, 13.5),
) -> RaceState:
    if snapshot.current_lap is None or snapshot.total_laps is None:
        raise NotEnoughData("lap count not known yet")
    if snapshot.current_lap < MIN_LAPS_FOR_PREDICTION:
        raise NotEnoughData(f"only {snapshot.current_lap} laps run; need a few laps of pace data")
    notes: list[str] = []

    # Degradation from this race's clean laps, per compound (defaults until enough stints).
    clean = [
        CleanLap(car, r.stint, r.compound, r.lap, r.tyre_age, r.time_s)
        for car, records in monitor.laps.items()
        for r in records
        if _clean(r)
    ]
    # In-race we want the *effective* lap-time trend on each tyre as observed in this race
    # (wear minus track evolution), not wear alone: that's what the rest of the race will look
    # like. It can be negative when evolution outpaces wear (Baku 2026) — floor it at 0 rather
    # than falling back to a generic default, which would over-correct long stints.
    deg = dict(DEFAULT_DEG)
    fitted = fit_tyre_model(clean, method="stint", fuel_gain=0.0).compounds if clean else {}
    for compound, fit in fitted.items():
        if fit.n_stints >= 3:
            deg[compound] = min(max(fit.deg_s_per_lap, 0.0), 0.3)
    defaulted = [c for c in DEFAULT_DEG if c not in fitted or fitted[c].n_stints < 3]
    if defaulted:
        notes.append(f"default degradation for {', '.join(defaulted)} (too few stints so far)")

    # Recent pace per car, tyre-age corrected, relative to the field median.
    pace: dict[str, float] = {}
    for car, records in monitor.laps.items():
        recent = [r for r in records if _clean(r)][-RECENT_LAPS:]
        if len(recent) >= 3:
            pace[car] = statistics.median(
                r.time_s - deg.get(r.compound, 0.06) * r.tyre_age for r in recent
            )
    field_median = statistics.median(pace.values()) if pace else 0.0

    cars = []
    for d in snapshot.drivers:
        if d.retired or d.position is None:
            continue
        gap = d.gap_to_leader_s
        if gap is None:
            gap = 999.0 + 100 * d.laps_down if d.position > 1 else 0.0
        cars.append(
            CarState(
                number=d.number,
                tla=d.tla,
                position=d.position,
                gap_s=gap,
                compound=d.compound,
                tyre_age=d.tyre_age_laps or 0,
                stops=d.pit_stops,
                owes_compound=d.needs_second_compound,
                pace_s=pace.get(d.number, field_median) - field_median,
                compound_c=c_number(snapshot.year or 0, snapshot.location, d.compound),
            )
        )
    cars.sort(key=lambda c: c.position)
    return RaceState(
        lap=snapshot.current_lap,
        laps_remaining=max(snapshot.total_laps - snapshot.current_lap, 0),
        status=snapshot.track_status,
        cars=cars,
        deg=deg,
        life=life or dict(DEFAULT_LIFE),
        pit_loss_green=pit_loss[0],
        pit_loss_sc=pit_loss[1],
        notes=notes,
    )


def tyre_outlook(car: CarState, state: RaceState) -> tuple[str, str]:
    """(risk, note) for running to the flag on the current set."""
    life = state.life.get(car.compound or "", 30)
    needed = car.tyre_age + state.laps_remaining
    if car.owes_compound:
        return "high", f"still owes the second compound: must stop ({car.tyre_age} laps on these)"
    if needed <= life:
        return "low", f"{car.tyre_age} laps old; reaches the flag at {needed}, within a long stint"
    over = needed - life
    risk = "medium" if over <= 0.2 * life else "high"
    return risk, (
        f"{car.tyre_age} laps old; would reach {needed} laps at the flag, {over} past the "
        f"longest typical 2026 {(car.compound or '').lower()} stint ({life})"
    )


def simulate_from(
    state: RaceState, params: SimParams | None = None, sims: int = 2000, seed: int = 0
) -> dict:
    p = params or SimParams()
    rng = np.random.default_rng(seed)
    cars, n, remaining = state.cars, len(state.cars), state.laps_remaining
    rows = np.arange(sims)[:, None]
    pace = np.array([c.pace_s for c in cars])[None, :]
    life = np.array([state.life.get(c.compound or "", 30) for c in cars], dtype=float)[None, :]
    deg_now = np.array([state.deg.get(c.compound or "", 0.06) for c in cars])[None, :]
    age = np.tile(np.array([c.tyre_age for c in cars], dtype=float), (sims, 1))

    if state.status in ("SAFETY_CAR", "RED_FLAG"):
        T = np.tile(np.arange(n) * 0.8, (sims, 1))  # queue behind the SC / on the grid
    else:
        T = np.tile(np.array([c.gap_s for c in cars]), (sims, 1))
    T = T + rng.normal(0, p.lap_noise, (sims, n))

    # Remaining stop (at most one): owed compound, or tyres that can't reach the flag.
    must = np.array([c.owes_compound for c in cars])[None, :]
    wont_last = (age[:1] + remaining) > life + 3
    needs_stop = np.broadcast_to(must | wont_last, (sims, n))
    latest = np.clip(life - age[:1], 1, max(remaining - 1, 1))
    stop_lap = np.where(needs_stop, 1 + np.floor(rng.random((sims, n)) * latest).astype(int), -1)
    new_deg = np.array(
        [state.deg.get("HARD" if c.compound != "HARD" else "MEDIUM", 0.05) for c in cars]
    )[None, :]
    new_life = np.array(
        [state.life.get("HARD" if c.compound != "HARD" else "MEDIUM", 30) for c in cars],
        dtype=float,
    )[None, :]
    deg = np.tile(deg_now, (sims, 1))
    life_now = np.tile(life, (sims, 1))
    running = np.ones((sims, n), dtype=bool)

    for lap in range(1, remaining + 1):
        order = np.argsort(T, axis=1)
        cliff = CLIFF_S_PER_LAP2 * np.maximum(age - life_now, 0) ** 1.5
        lap_time = pace + deg * age + cliff + rng.normal(0, p.lap_noise, (sims, n))
        pit_now = stop_lap == lap
        loss = (
            state.pit_loss_sc
            if (lap == 1 and state.status == "SAFETY_CAR")
            else state.pit_loss_green
        )
        lap_time = lap_time + np.where(pit_now, loss, 0.0)
        age = np.where(pit_now, 0, age + 1)
        deg = np.where(pit_now, new_deg, deg)
        life_now = np.where(pit_now, new_life, life_now)
        new_T = T + lap_time
        traffic_step(new_T, lap_time, order, running, np.ones(sims, dtype=bool), p, rng)
        T = new_T

    positions = np.empty((sims, n), dtype=int)
    positions[rows, np.argsort(T, axis=1)] = np.arange(1, n + 1)[None, :]
    table = []
    for i, c in enumerate(cars):
        risk, note = tyre_outlook(c, state)
        table.append(
            {
                "tla": c.tla,
                "position": c.position,
                "expected": round(float(positions[:, i].mean()), 2),
                "p_win": round(float((positions[:, i] == 1).mean()), 3),
                "p_podium": round(float((positions[:, i] <= 3).mean()), 3),
                "compound": c.compound,
                "compound_c": c.compound_c,
                "tyre_age": c.tyre_age,
                "stopped": c.stops > 0,
                "tyre_risk": risk,
                "tyre_note": note,
            }
        )
    return {
        "lap": state.lap,
        "laps_remaining": state.laps_remaining,
        "status": state.status,
        "sims": sims,
        "table": table,
        "notes": state.notes,
    }
