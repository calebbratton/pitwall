"""Was pitting the right call? Replays a finished race to just before a driver's stop and
simulates the alternatives with the tyre numbers measured in that race.

  - State: the live pipeline (RaceMonitor) fed from the archive up to the lap the driver pitted.
  - Measured, not assumed: the driver's degradation on the old set (their own last laps), the
    fresh-tyre gain they actually got (their first clean laps after the stop vs their last before),
    compound offsets and degradation from the whole race, the circuit's measured pit loss.
  - Everyone else stops when they really did (hindsight about rivals), so the question is only
    this driver's call.
  - Branches: pit on that lap (what happened), stay out 1..N more laps, or no more stops when the
    tyres could make the flag and no compound is owed. Same random numbers for every branch.

Usage: python -m src.sim.pit_review --year 2026 --meeting Azerbaijan --driver RUS --lap 31
"""

import argparse
import itertools
import statistics
from dataclasses import dataclass, replace

from src.livetiming.archive import ArchiveSession, find_session, list_sessions
from src.livetiming.monitor import STRATEGY_TOPICS, LapRecord, RaceMonitor, _circuit_for
from src.models.tyre_curves import season_curves
from src.models.tyres import CleanLap, fit_tyre_model
from src.sim.inrace import StopPlan, race_state, simulate_from

MEASURE_LAPS = 3  # clean laps either side of the stop for the measured fresh-tyre gain
MAX_DEFER = 5  # "stay out" branches: 1..MAX_DEFER more laps


@dataclass(frozen=True)
class Branch:
    label: str  # "pit lap 31 (actual)", "stay out 2 laps", "no stop"
    expected: float  # expected finishing position
    p_win: float
    p_podium: float
    p_better_than_actual: float  # share of simulations finishing ahead of the actual branch
    p_worse_than_actual: float


@dataclass(frozen=True)
class PitReview:
    driver: str
    lap: int
    status: str  # track status when the decision was made
    old_compound: str | None
    new_compound: str | None
    old_tyre_age: int
    measured: dict[str, float]  # the numbers the review used, for the answer
    branches: list[Branch]
    verdict: str
    notes: list[str]


def _clean_times(records: list[LapRecord]) -> list[LapRecord]:
    return [
        r
        for r in records
        if r.time_s is not None and not (r.pit_in or r.pit_out or r.neutralised) and r.lap > 1
    ]


def _stops(records: list[LapRecord]) -> list[tuple[int, str | None]]:
    """(in-lap, compound fitted) for each stop: the out-lap's compound."""
    out = []
    for prev, cur in itertools.pairwise(records):
        if prev.pit_in and cur.stint > prev.stint:
            out.append((prev.lap, cur.compound))
    return out


def measured_pit_loss(laps: dict[str, list[LapRecord]]) -> tuple[float, int] | None:
    """Green-flag pit loss measured in the race: per stop, (in-lap + out-lap) minus twice the
    car's median clean lap in the surrounding laps; median over stops. (median, stops used)."""
    losses = []
    for recs in laps.values():
        by_lap = {r.lap: r for r in recs}
        clean = _clean_times(recs)
        for r in recs:
            out = by_lap.get(r.lap + 1)
            if not (r.pit_in and out and out.pit_out) or r.neutralised or out.neutralised:
                continue
            if r.time_s is None or out.time_s is None:
                continue
            around = [c.time_s for c in clean if 0 < abs(c.lap - r.lap) <= 5]
            if len(around) >= 3:
                losses.append(r.time_s + out.time_s - 2 * statistics.median(around))
    losses = [x for x in losses if 10 < x < 45]
    return (round(statistics.median(losses), 1), len(losses)) if len(losses) >= 3 else None


def archive_race(year: int, place: str) -> ArchiveSession:
    """The archived race for a meeting name ("Azerbaijan") or a location ("Madrid" is filed as
    the Spanish Grand Prix): by archive folder name, else by race date from the warehouse."""
    try:
        return ArchiveSession(find_session(year, place))
    except LookupError:
        from src.sim.inputs import weekend_sessions
        from src.warehouse.queries import connect

        con = connect()
        try:
            _, sessions = weekend_sessions(con, year, place)
            day = con.execute(
                "SELECT strftime(date_start, '%Y-%m-%d') FROM races WHERE session_key = ?",
                [sessions["Race"]],
            ).fetchone()[0]
        finally:
            con.close()
        match = [r for r in list_sessions(year) if r["date"] == day]
        if not match:
            raise
        return ArchiveSession(match[0]["path"])


def review(year: int, meeting: str, driver: str, lap: int, sims: int = 4000) -> PitReview:
    """Review `driver`'s stop at the end of `lap` (their in-lap)."""
    session = archive_race(year, meeting)
    messages = list(session.messages(STRATEGY_TOPICS))
    circuit = _circuit_for(messages)
    loss = (
        (circuit.pit_loss.green, circuit.pit_loss.safety_car)
        if circuit and circuit.pit_loss
        else (22.0, 13.5)
    )

    # Whole race first: the real stops of every car and the driver's laps around the stop.
    full = RaceMonitor()
    for m in messages:
        full.feed(m)
    pit_note = f"pit loss from circuit data ({loss[0]:.1f} s green)" if circuit else ""
    if (measured := measured_pit_loss(full.laps)) is not None:
        green, n_stops = measured
        # Keep the circuit's SC/green ratio (a stop under SC costs less: the field is slow).
        ratio = loss[1] / loss[0]
        loss = (green, round(green * ratio, 1))
        pit_note = f"pit loss measured from {n_stops} green-flag stops in this race: {green:.1f} s"
    snap_end = full.snapshot()
    number = snap_end.driver(driver).number
    tla = snap_end.driver(driver).tla
    stops = {car: _stops(recs) for car, recs in full.laps.items()}
    mine = [s for s in stops.get(number, []) if s[0] == lap]
    if not mine:
        known = [s[0] for s in stops.get(number, [])]
        raise ValueError(f"{tla} didn't pit at the end of lap {lap}; their stops: {known}")
    new_compound = mine[0][1]

    # State at the decision: just before the driver enters the pit lane on their in-lap (so a
    # safety car deployed earlier in that lap is part of the situation).
    monitor = RaceMonitor(pit_loss=circuit.pit_loss if circuit else None)
    for m in messages:
        recs = monitor.laps.get(number) or []
        entering = (
            m.topic == "TimingData"
            and ((m.data.get("Lines") or {}).get(number) or {}).get("InPit") is True
        )
        if entering and recs and recs[-1].lap >= lap - 1:
            break
        monitor.feed(m)
    snap = monitor.snapshot()
    curves = season_curves(year)
    state = race_state(monitor, snap, pit_loss=loss, curves=curves)
    decision_lap = state.lap  # leader's lap count at the decision

    # Hindsight tyre numbers from the whole race (the "real tyre delta"), fuel-corrected: the
    # in-race fit leaves fuel burn in (right for "what happens next"), which makes late stints
    # on the softer compound look far faster than the tyre is.
    clean = [
        CleanLap(car_no, r.stint, r.compound, r.lap, r.tyre_age, r.time_s)
        for car_no, recs in full.laps.items()
        for r in _clean_times(recs)
        if r.compound and r.tyre_age is not None
    ]
    fits = fit_tyre_model(clean, method="stint").compounds
    deg = dict(state.deg)
    offset = {}
    for compound, fit in fits.items():
        if fit.n_stints >= 3:
            deg[compound] = min(max(fit.deg_s_per_lap, 0.0), 0.3)
        if fit.offset_s is not None:
            offset[compound] = fit.offset_s
    if offset:
        for compound in fits:
            offset.setdefault(compound, 0.0)
    state = replace(state, deg=deg, offset=offset or state.offset)
    my_laps = full.laps[number]
    before = [r for r in _clean_times(my_laps) if r.lap < lap][-MEASURE_LAPS:]
    after = [r for r in _clean_times(my_laps) if r.lap > lap + 1][:MEASURE_LAPS]
    measured = {
        "pit_loss_green_s": loss[0],
        "pit_loss_sc_s": loss[1],
    }
    car = next(c for c in state.cars if c.number == number)
    if before:
        measured["old_tyre_lap_s"] = round(statistics.median(r.time_s for r in before), 3)
        measured["old_tyre_deg_s_per_lap"] = round(state.deg.get(car.compound or "", 0.06), 3)
    if before and after:
        measured["fresh_tyre_gain_s_per_lap"] = round(
            statistics.median(r.time_s for r in before)
            - statistics.median(r.time_s for r in after),
            3,
        )
    if new_compound and car.compound:
        measured["compound_offset_s"] = round(
            state.offset.get(new_compound, 0.0) - state.offset.get(car.compound, 0.0), 3
        )

    # Rivals: their real next stop after the decision (laps from now), or none.
    def rival_plan(car_number: str) -> StopPlan:
        upcoming = [(in_lap, c) for in_lap, c in stops.get(car_number, []) if in_lap >= lap]
        if not upcoming:
            return StopPlan(None)
        in_lap, compound = upcoming[0]
        return StopPlan(max(in_lap - (lap - 1), 1), compound)

    base_plan = {c.number: rival_plan(c.number) for c in state.cars if c.number != number}
    remaining = state.laps_remaining
    options: list[tuple[str, StopPlan]] = [(f"pit lap {lap} (actual)", StopPlan(1, new_compound))]
    for k in range(1, MAX_DEFER + 1):
        if k + 1 <= remaining:
            options.append(
                (f"stay out {k} more lap{'s' if k > 1 else ''}", StopPlan(1 + k, new_compound))
            )
    can_finish = car.tyre_age + remaining <= (
        car.life_laps or state.life.get(car.compound or "", 30)
    )
    if can_finish and not car.owes_compound:
        options.append(("no stop (run to the flag)", StopPlan(None)))

    idx = [c.number for c in state.cars].index(number)
    results = []
    for label, my_plan in options:
        out = simulate_from(
            state, sims=sims, seed=7, plan={**base_plan, number: my_plan}, return_positions=True
        )
        results.append((label, out))
    actual_pos = results[0][1]["positions"][:, idx]
    branches = []
    for label, out in results:
        pos = out["positions"][:, idx]
        branches.append(
            Branch(
                label=label,
                expected=round(float(pos.mean()), 2),
                p_win=round(float((pos == 1).mean()), 3),
                p_podium=round(float((pos <= 3).mean()), 3),
                p_better_than_actual=round(float((pos < actual_pos).mean()), 3),
                p_worse_than_actual=round(float((pos > actual_pos).mean()), 3),
            )
        )
    best = min(branches, key=lambda b: b.expected)
    actual = branches[0]
    margin = actual.expected - best.expected
    if best is actual or margin < 0.15:
        verdict = f"Pitting on lap {lap} was the right call (or as good as any alternative)."
    else:
        verdict = (
            f"{best.label.capitalize()} would likely have been better: expected P{best.expected:.1f}"
            f" vs P{actual.expected:.1f}, ahead of the actual result in "
            f"{best.p_better_than_actual:.0%} of simulations."
        )
    return PitReview(
        driver=tla,
        lap=lap,
        status=state.status,
        old_compound=car.compound,
        new_compound=new_compound,
        old_tyre_age=car.tyre_age,
        measured=measured,
        branches=branches,
        verdict=verdict,
        notes=[
            f"state at the leader's lap {decision_lap}, {remaining} laps to go, track {state.status}",
            *([pit_note] if pit_note else []),
            "rivals stop when they really did (hindsight); tyre numbers measured over the whole race",
            *state.notes,
        ],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--meeting", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--lap", type=int, required=True, help="the driver's in-lap")
    args = ap.parse_args()
    r = review(args.year, args.meeting, args.driver, args.lap)
    print(
        f"{r.driver} pitted at the end of lap {r.lap} ({r.status}): {r.old_compound} "
        f"({r.old_tyre_age} laps) -> {r.new_compound}"
    )
    print("measured:", r.measured)
    print(f"{'option':28} {'exp. pos':>8} {'win':>6} {'podium':>7} {'better':>7} {'worse':>6}")
    for b in r.branches:
        print(
            f"{b.label:28} {b.expected:8.2f} {b.p_win:6.0%} {b.p_podium:7.0%} "
            f"{b.p_better_than_actual:7.0%} {b.p_worse_than_actual:6.0%}"
        )
    print("\n" + r.verdict)
    for n in r.notes:
        print("-", n)


if __name__ == "__main__":
    main()
