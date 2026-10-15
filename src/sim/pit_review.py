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

What if they had pitted on a lap they didn't (`what_if_stop`, e.g. under a VSC)? Same machinery:
the state as the driver reaches the pit entry on that lap, then two branches - what they really
did next (their real next stop, or none) and a stop on that lap.

Usage: python -m src.sim.pit_review --year 2026 --meeting Azerbaijan --driver RUS --lap 31
       python -m src.sim.pit_review --year 2026 --meeting Madrid --driver NOR --lap 14 --what-if
"""

import argparse
import itertools
import statistics
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date

import numpy as np

from src.livetiming.archive import ArchiveSession, Message, find_session, list_sessions
from src.livetiming.circuits import Circuit
from src.livetiming.monitor import STRATEGY_TOPICS, LapRecord, RaceMonitor, _circuit_for
from src.models.tyre_curves import season_curves
from src.models.tyres import CleanLap, fit_tyre_model, stint_slopes
from src.sim.inrace import RaceState, StopPlan, race_state, simulate_from

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
    ledger: dict | None = None  # what-ifs: the stop's cost vs the fresh-tyre gain, in seconds


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


def archive_race(year: int, place: str, day: str | None = None) -> ArchiveSession:
    """The archived race for a meeting name ("Azerbaijan") or a location ("Madrid" is filed as
    the Spanish Grand Prix): by archive folder name, else by race date - `day` (YYYY-MM-DD, e.g.
    from the OpenF1 session) when given, else from the warehouse."""
    try:
        return ArchiveSession(find_session(year, place))
    except LookupError:
        if day is None:
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
        # The archive dates races in local time, OpenF1 and the warehouse in UTC: a night race
        # can land on the next day (Las Vegas). Races are a week apart, so one day either side
        # is unambiguous.
        target = date.fromisoformat(day)
        match = [
            r
            for r in list_sessions(year)
            if r["date"] and abs((date.fromisoformat(r["date"]) - target).days) <= 1
        ]
        if len(match) != 1:
            raise
        return ArchiveSession(match[0]["path"])


@dataclass
class _Race:
    """A finished race fed through the live pipeline once, for one driver."""

    messages: list[Message]
    year: int
    circuit: Circuit | None
    loss: tuple[float, float]  # (green, SC) pit loss
    pit_note: str
    full: RaceMonitor  # the whole race
    number: str
    tla: str
    stops: dict[str, list[tuple[int, str | None]]]  # car -> (in-lap, compound fitted)


def _load(messages: list[Message], year: int, driver: str) -> _Race:
    circuit = _circuit_for(messages)
    loss = (
        (circuit.pit_loss.green, circuit.pit_loss.safety_car)
        if circuit and circuit.pit_loss
        else (22.0, 13.5)
    )
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
    me = full.snapshot().driver(driver)
    return _Race(
        messages=messages,
        year=year,
        circuit=circuit,
        loss=loss,
        pit_note=pit_note,
        full=full,
        number=me.number,
        tla=me.tla,
        stops={car: _stops(recs) for car, recs in full.laps.items()},
    )


def _finish(race: _Race) -> dict[str, int]:
    """Where every classified car really finished."""
    return {d.number: d.position for d in race.full.snapshot().drivers if d.position}


def _tlas(race: _Race) -> dict[str, str]:
    return {d.number: d.tla for d in race.full.snapshot().drivers}


def _state_until(race: _Race, stop_before: Callable[[Message, list[LapRecord]], bool]) -> RaceState:
    """Race state just before the first message `stop_before` flags (given the driver's laps so
    far), with hindsight tyre numbers from the whole race."""
    monitor = RaceMonitor(pit_loss=race.circuit.pit_loss if race.circuit else None)
    for m in race.messages:
        if stop_before(m, monitor.laps.get(race.number) or []):
            break
        monitor.feed(m)
    state = race_state(
        monitor, monitor.snapshot(), pit_loss=race.loss, curves=season_curves(race.year)
    )
    # Hindsight tyre numbers from the whole race (the "real tyre delta"), fuel-corrected: the
    # in-race fit leaves fuel burn in (right for "what happens next"), which makes late stints
    # on the softer compound look far faster than the tyre is.
    clean = [
        CleanLap(car_no, r.stint, r.compound, r.lap, r.tyre_age, r.time_s)
        for car_no, recs in race.full.laps.items()
        for r in _clean_times(recs)
        if r.compound and r.tyre_age is not None
    ]
    fits = fit_tyre_model(clean, method="stint").compounds if clean else {}
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
    return replace(state, deg=deg, offset=offset or state.offset)


def _rival_plans(race: _Race, state: RaceState, lap: int) -> dict[str, StopPlan]:
    """Every other car's real next stop after the decision (laps from now), or none."""

    def plan(car_number: str) -> StopPlan:
        upcoming = [(in_lap, c) for in_lap, c in race.stops.get(car_number, []) if in_lap >= lap]
        if not upcoming:
            return StopPlan(None)
        in_lap, compound = upcoming[0]
        return StopPlan(max(in_lap - (lap - 1), 1), compound)

    return {c.number: plan(c.number) for c in state.cars if c.number != race.number}


def _branches(
    race: _Race,
    state: RaceState,
    base_plan: dict[str, StopPlan],
    options: list[tuple[str, StopPlan]],
    sims: int,
) -> tuple[list[Branch], np.ndarray]:
    """Simulate each option for the driver (same random numbers for every branch); the first
    option is the reference the others are compared with. Also returns the reference option's
    finishing positions for every car (sims x cars, in `state.cars` order)."""
    idx = [c.number for c in state.cars].index(race.number)
    results, reference_positions = [], None
    for label, my_plan in options:
        out = simulate_from(
            state,
            sims=sims,
            seed=7,
            plan={**base_plan, race.number: my_plan},
            return_positions=True,
        )
        results.append((label, out["positions"][:, idx]))
        if reference_positions is None:
            reference_positions = out["positions"]
    reference = results[0][1]
    return [
        Branch(
            label=label,
            expected=round(float(pos.mean()), 2),
            p_win=round(float((pos == 1).mean()), 3),
            p_podium=round(float((pos <= 3).mean()), 3),
            p_better_than_actual=round(float((pos < reference).mean()), 3),
            p_worse_than_actual=round(float((pos > reference).mean()), 3),
        )
        for label, pos in results
    ], reference_positions


def _measured(race: _Race, state: RaceState, lap: int, new_compound: str | None) -> dict:
    """The numbers the review used: pit loss, the driver's old-tyre pace and wear, and (for a
    stop that happened) the fresh-tyre gain they actually got."""
    measured = {"pit_loss_green_s": race.loss[0], "pit_loss_sc_s": race.loss[1]}
    car = next(c for c in state.cars if c.number == race.number)
    my_laps = race.full.laps[race.number]
    before = [r for r in _clean_times(my_laps) if r.lap < lap][-MEASURE_LAPS:]
    after = [r for r in _clean_times(my_laps) if r.lap > lap + 1][:MEASURE_LAPS]
    if before:
        measured["old_tyre_lap_s"] = round(statistics.median(r.time_s for r in before), 3)
        measured["old_tyre_deg_s_per_lap"] = round(state.deg.get(car.compound or "", 0.06), 3)
    if before and after and any(s[0] == lap for s in race.stops.get(race.number, [])):
        measured["fresh_tyre_gain_s_per_lap"] = round(
            statistics.median(r.time_s for r in before)
            - statistics.median(r.time_s for r in after),
            3,
        )
    if new_compound and car.compound:
        measured["compound_offset_s"] = round(
            state.offset.get(new_compound, 0.0) - state.offset.get(car.compound, 0.0), 3
        )
    return measured


def neutralised_laps(laps: list[LapRecord]) -> list[int]:
    """The car's laps with a SC / VSC / red flag out at some point: the cheap-stop windows."""
    return [r.lap for r in laps if r.neutralised and r.lap > 1]


def review(
    year: int, meeting: str, driver: str, lap: int, sims: int = 4000, day: str | None = None
) -> PitReview:
    """Review `driver`'s stop at the end of `lap` (their in-lap). `day`: the race date, when
    `meeting` might not name the archive folder (see `archive_race`)."""
    messages = list(archive_race(year, meeting, day).messages(STRATEGY_TOPICS))
    return review_race(messages, year, driver, lap, sims)


def baseline_check(
    number: str,
    cars: list[str],
    positions: np.ndarray,
    real: dict[str, int],
    tla: dict[str, str],
) -> str | None:
    """What the driver really did, simulated (`positions`: sims x cars, in `cars` order) next to
    where they really finished. A place or more apart: the head-to-heads the simulation gets
    wrong (it keeps the driver ahead of / behind a car in 80%+ of runs, the real result was the
    other way). Both options share that bias, so the comparison between them is the answer."""
    if number not in real:
        return None
    me = cars.index(number)
    expected = float(positions[:, me].mean())
    name = tla.get(number, number)
    note = f"what {name} really did simulates to P{expected:.1f}; {name} really finished P{real[number]}"
    if abs(expected - real[number]) < 1:
        return note
    wrong = []
    for j, car in enumerate(cars):
        if j == me or car not in real:
            continue
        ahead = float((positions[:, me] < positions[:, j]).mean())
        other = tla.get(car, car)
        if real[car] < real[number] and ahead >= 0.8:
            wrong.append(f"ahead of {other} in {ahead:.0%} of runs, but {other} finished ahead")
        elif real[car] > real[number] and ahead <= 0.2:
            wrong.append(f"behind {other} in {1 - ahead:.0%} of runs, but {other} finished behind")
    if wrong:
        note += f" (the simulation has {name} " + "; ".join(wrong) + ")"
    return note + ": compare the options with each other rather than with the real result"


CLOSE_CALL = 0.15  # expected-position difference below which options are a toss-up


def review_verdict(lap: int, branches: list[Branch]) -> str:
    """The verdict on a real stop (branches[0]) against the alternatives, worded by the margin:
    the stop is only "the right call" when no alternative scored better."""
    actual = branches[0]
    best = min(branches, key=lambda b: (b.expected, b is not actual))
    margin = actual.expected - best.expected
    if best is actual:
        return f"Pitting on lap {lap} was the right call: no alternative scored better."
    if margin < CLOSE_CALL:
        return (
            f"Close call: {best.label} scores slightly better (expected P{best.expected:.2f} vs "
            f"P{actual.expected:.2f} for pitting on lap {lap}; ahead of the actual result in "
            f"{best.p_better_than_actual:.0%} of simulations, behind in "
            f"{best.p_worse_than_actual:.0%}): {margin:.2f} places apart."
        )
    return (
        f"{best.label.capitalize()} would likely have been better: expected P{best.expected:.1f}"
        f" vs P{actual.expected:.1f}, ahead of the actual result in "
        f"{best.p_better_than_actual:.0%} of simulations."
    )


def review_race(
    messages: list[Message], year: int, driver: str, lap: int, sims: int = 4000
) -> PitReview:
    race = _load(messages, year, driver)
    number, tla, stops, loss = race.number, race.tla, race.stops, race.loss
    mine = [s for s in stops.get(number, []) if s[0] == lap]
    if not mine:
        known = [s[0] for s in stops.get(number, [])]
        raise ValueError(f"{tla} didn't pit at the end of lap {lap}; their stops: {known}")
    new_compound = mine[0][1]

    # State at the decision: just before the driver enters the pit lane on their in-lap (so a
    # safety car deployed earlier in that lap is part of the situation).
    def entering(m: Message, recs: list[LapRecord]) -> bool:
        in_pit = (
            m.topic == "TimingData"
            and ((m.data.get("Lines") or {}).get(number) or {}).get("InPit") is True
        )
        return in_pit and bool(recs) and recs[-1].lap >= lap - 1

    state = _state_until(race, entering)
    measured = _measured(race, state, lap, new_compound)
    car = next(c for c in state.cars if c.number == number)
    base_plan = _rival_plans(race, state, lap)
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
    branches, reference = _branches(race, state, base_plan, options, sims)
    check = baseline_check(
        number, [c.number for c in state.cars], reference, _finish(race), _tlas(race)
    )

    # Context: rivals near the driver who stopped under a SC/VSC just before this decision - the
    # cheap stop the driver didn't get (Madrid 2026: everyone but Norris pitted under the VSC).
    context = []
    my_gap = car.gap_s
    for other in state.cars:
        if other.number == number or abs(other.gap_s - my_gap) > loss[0] + 5:
            continue
        for in_lap, _ in stops.get(other.number, []):
            rec = next((r for r in race.full.laps.get(other.number, []) if r.lap == in_lap), None)
            if rec and rec.neutralised and lap - 3 <= in_lap < lap:
                context.append(other.tla)
    if context:
        saving = loss[0] - loss[1]
        context_note = (
            f"{', '.join(context)} stopped under a SC/VSC in the 3 laps before {tla}'s stop "
            f"(about {saving:.0f} s cheaper than {tla}'s green-flag stop): the cost came from "
            "the neutralisation's timing, not from this decision"
        )
    else:
        context_note = ""
    verdict = review_verdict(lap, branches)
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
            f"state at the leader's lap {state.lap}, {remaining} laps to go, track {state.status}",
            *([race.pit_note] if race.pit_note else []),
            *([context_note] if context_note else []),
            *([check] if check else []),
            "rivals stop when they really did (hindsight); tyre numbers measured over the whole race",
            *state.notes,
        ],
    )


LEDGER_BEFORE_LAPS = 6  # clean laps before the stop window that set each pair's pace gap


def _lap_times(recs: list[LapRecord]) -> dict[int, LapRecord]:
    return {r.lap: r for r in recs if r.time_s is not None}


def time_ledger(
    race: _Race, lap: int, compound: str | None, deg: dict[str, float] | None = None
) -> dict | None:
    """Back-of-the-envelope check on a what-if stop, from the laps really driven: what a stop
    on `lap` cost the cars that made one then, and how much faster they were afterwards
    relative to the driver (who stayed out) than before. Net > 0: the stop would have gained
    the driver that many seconds by the flag (ignoring traffic and track position). If the
    driver really stopped once more later, the hypothetical stop replaces it: that stop's cost
    is saved, and the laps after it count too (where the driver's tyres were the fresher)."""
    number = race.number
    mine = _lap_times(race.full.laps.get(number) or [])
    later = [in_lap for in_lap, _ in race.stops.get(number, []) if in_lap > lap]
    if len(later) > 1:
        return None  # the hypothetical stop would replace one of several: not a simple sum
    end = max(mine, default=lap)
    window = {lap - 1, lap, lap + 1}
    refs = [
        (car, in_lap, fitted)
        for car, stops in race.stops.items()
        for in_lap, fitted in stops
        if car != number and in_lap in window
    ]
    # Stops on the lap itself first: a lap either side can be under another track status (a
    # green stop the lap before a VSC costs nearly twice as much).
    refs = [r for r in refs if r[1] == lap] or refs
    same = [r for r in refs if compound and r[2] == compound]
    refs = same or refs
    if not refs:
        return None
    tla = {d.number: d.tla for d in race.full.snapshot().drivers}

    def clean(r: LapRecord | None) -> bool:
        return r is not None and not (r.pit_in or r.pit_out or r.neutralised)

    # Each car's own wear on the stint it was on, from its clean laps before the decision (a
    # compound's race-wide rate needn't fit one car's stint); the race-wide rate where a stint
    # is too short to fit.
    own_wear = stint_slopes(
        [
            CleanLap(car, r.stint, r.compound, r.lap, r.tyre_age, r.time_s)
            for car, recs in race.full.laps.items()
            for r in _clean_times(recs)
            if r.lap < lap and r.compound and r.tyre_age is not None
        ]
    )

    def wear(car: str, r: LapRecord) -> float:
        slope = own_wear.get((car, r.stint))
        if slope is None:
            slope = (deg or {}).get(r.compound or "", 0.0)
        return min(max(slope, 0.0), 0.3) * (r.tyre_age or 0)

    def stop_cost(car: str, in_lap: int, exclude: set[str]) -> float | None:
        """In-lap + out-lap vs the median of cars that didn't stop on those laps."""
        theirs = _lap_times(race.full.laps.get(car) or [])
        if in_lap not in theirs or in_lap + 1 not in theirs:
            return None
        others = [
            laps[in_lap].time_s + laps[in_lap + 1].time_s
            for other, recs in race.full.laps.items()
            if other not in exclude
            and in_lap in (laps := _lap_times(recs))
            and in_lap + 1 in laps
            and not any(laps[n].pit_in or laps[n].pit_out for n in (in_lap, in_lap + 1))
        ]
        if not others:
            return None
        return theirs[in_lap].time_s + theirs[in_lap + 1].time_s - statistics.median(others)

    stopping = {car for car, _, _ in refs}
    # A real later stop is one the hypothetical one replaces: its cost is saved.
    saved = stop_cost(number, later[0], {number}) if later else 0.0
    if saved is None:
        return None
    costs, gains, used = [], [], []
    for car, in_lap, _ in refs:
        theirs = _lap_times(race.full.laps.get(car) or [])
        # Their pace gap on equal tyres: the lap-time gap before the stop, less the part from
        # tyre age (measured degradation x age difference) - had the driver stopped too, they'd
        # have been on tyres as old as the other car's.
        before = [
            mine[n].time_s - theirs[n].time_s - wear(number, mine[n]) + wear(car, theirs[n])
            for n in range(lap - LEDGER_BEFORE_LAPS, lap)
            if clean(mine.get(n)) and clean(theirs.get(n))
        ]
        base = statistics.median(before) if len(before) >= 3 else None
        # The stop's cost: their in- and out-lap, at the driver's pace, vs the driver's own laps
        # (who stayed out, near them on track). A SC/VSC ends at different times around the
        # lap, so the field's median laps aren't what staying out was worth here.
        own = [mine.get(n) for n in (in_lap, in_lap + 1)]
        if (
            base is not None
            and {in_lap, in_lap + 1} <= theirs.keys()
            and all(r is not None and not (r.pit_in or r.pit_out) for r in own)
        ):
            costs.append(
                theirs[in_lap].time_s
                + theirs[in_lap + 1].time_s
                + 2 * base
                - sum(r.time_s for r in own)
            )
        elif base is None and (cost := stop_cost(car, in_lap, stopping)) is not None:
            costs.append(cost)
        after = [
            (mine[n].time_s - theirs[n].time_s)
            for n in range(in_lap + 2, end + 1)
            if clean(mine.get(n)) and clean(theirs.get(n))
        ]
        if base is not None and after:
            excess = [d - base for d in after]
            laps_run = max(end - (in_lap + 1), len(after))
            # Laps without a clean comparison (traffic, flags) count at the average gain.
            gains.append(sum(excess) + statistics.fmean(excess) * (laps_run - len(after)))
            used.append(tla.get(car, car))
    if not costs or not gains:
        return None
    cost = statistics.median(costs)
    gain = statistics.median(gains)
    net = gain - cost + saved

    # Against the real finishing gaps: who the net time would have put the driver ahead of.
    final = race.full.snapshot()
    me = final.driver(number)
    passed, lost_to = [], []
    if me.gap_to_leader_s is not None:
        for d in final.drivers:
            if d.number == number or d.gap_to_leader_s is None or d.laps_down:
                continue
            gap = me.gap_to_leader_s - d.gap_to_leader_s  # > 0: they finished ahead
            if 0 < gap < net:
                passed.append(f"{d.tla} ({gap:.1f} s ahead at the flag)")
            elif net < gap < 0:
                lost_to.append(f"{d.tla} ({-gap:.1f} s behind at the flag)")
    return {
        "compared_with": used,
        "stop_cost_s": round(cost, 1),
        "fresh_tyre_gain_s": round(gain, 1),
        **({f"saved_real_stop_lap_{later[0]}_s": round(saved, 1)} if later else {}),
        "net_s": round(net, 1),
        "laps_to_flag": end - lap,
        "would_have_finished_ahead_of": passed,
        "would_have_lost_to": lost_to,
    }


def what_if_stop(
    year: int,
    meeting: str,
    driver: str,
    lap: int,
    compound: str | None = None,
    sims: int = 4000,
    day: str | None = None,
) -> PitReview:
    """What if `driver` had pitted at the end of `lap` (a lap they didn't pit on, e.g. under a
    SC/VSC) instead of what they really did? `day`: as for `review`."""
    messages = list(archive_race(year, meeting, day).messages(STRATEGY_TOPICS))
    return what_if_race(messages, year, driver, lap, compound, sims)


def what_if_race(
    messages: list[Message],
    year: int,
    driver: str,
    lap: int,
    compound: str | None = None,
    sims: int = 4000,
) -> PitReview:
    race = _load(messages, year, driver)
    number, tla = race.number, race.tla
    my_stops = race.stops.get(number, [])
    my_laps = race.full.laps.get(number) or []
    if any(in_lap == lap for in_lap, _ in my_stops):
        raise ValueError(f"{tla} did pit at the end of lap {lap}: review that stop instead")
    last_lap = my_laps[-1].lap if my_laps else 0
    if not 2 <= lap < last_lap:
        raise ValueError(f"{tla} ran laps 1-{last_lap}; pick a lap between 2 and {last_lap - 1}")

    # State at the decision: just before the driver completes `lap`, i.e. as they reach the pit
    # entry (a SC/VSC deployed earlier in that lap is part of the situation).
    def completing(m: Message, _recs: list[LapRecord]) -> bool:
        line = (m.data.get("Lines") or {}).get(number) if m.topic == "TimingData" else None
        return isinstance(line, dict) and str(line.get("NumberOfLaps")) == str(lap)

    state = _state_until(race, completing)
    car = next((c for c in state.cars if c.number == number), None)
    if car is None:
        raise ValueError(f"{tla} wasn't running at lap {lap}")

    # What really happened next: the driver's next real stop (and the compound they fitted).
    later = [(in_lap, c) for in_lap, c in my_stops if in_lap > lap]
    if later:
        real_lap, real_compound = later[0]
        actual = (f"actual: pit lap {real_lap}", StopPlan(real_lap - (lap - 1), real_compound))
    else:
        real_compound = None
        actual = ("actual: no more stops", StopPlan(None))
    new = (compound or real_compound or ("HARD" if car.compound != "HARD" else "MEDIUM")).upper()
    if car.owes_compound and new == car.compound:
        raise ValueError(
            f"{tla} still owed a second compound at lap {lap}: fitting {new} again would break "
            "the tyre rule; pick another compound"
        )
    status = {"SAFETY_CAR": "under the SC", "VSC": "under the VSC"}.get(state.status, "")
    # The direct answer from the laps really driven; its measured stop cost feeds the simulation.
    ledger = time_ledger(race, lap, new, state.deg)
    if ledger:
        state = replace(state, pit_loss_now=ledger["stop_cost_s"])
    hypo_label = f"pit lap {lap}{f' {status}' if status else ''} for {new}"
    branches, reference = _branches(
        race, state, _rival_plans(race, state, lap), [actual, (hypo_label, StopPlan(1, new))], sims
    )
    check = baseline_check(
        number, [c.number for c in state.cars], reference, _finish(race), _tlas(race)
    )
    real, hypo = branches
    if abs(real.expected - hypo.expected) < 0.15:
        sim = (
            f"little difference to the expected result (P{hypo.expected:.1f} vs "
            f"P{real.expected:.1f} for what {tla} really did)"
        )
    else:
        sim = (
            f"expected P{hypo.expected:.1f} vs P{real.expected:.1f} for what {tla} really did "
            f"({actual[0][8:]}); ahead in {hypo.p_better_than_actual:.0%} of simulations, "
            f"behind in {hypo.p_worse_than_actual:.0%}"
        )
    if ledger:
        net = ledger["net_s"]
        verdict = (
            f"Pitting on lap {lap} would have {'gained' if net > 0 else 'lost'} {tla} about "
            f"{abs(net):.0f} s by the flag: the stop cost {ledger['stop_cost_s']:.1f} s (measured "
            f"from {', '.join(ledger['compared_with'])}), fresh tyres were worth "
            f"{ledger['fresh_tyre_gain_s']:.1f} s"
            + (f", and it replaces the real lap-{later[0][0]} stop" if later else "")
            + "."
        )
        if ledger["would_have_finished_ahead_of"]:
            ahead = ", ".join(ledger["would_have_finished_ahead_of"])
            verdict += f" Enough to finish ahead of {ahead}."
        if ledger["would_have_lost_to"]:
            verdict += f" It would have dropped behind {', '.join(ledger['would_have_lost_to'])}."
        verdict += f" Simulated with traffic and track position: {sim}."
    else:
        verdict = f"Pitting on lap {lap}, simulated: {sim}."
    notes = [
        f"state at the leader's lap {state.lap}, {state.laps_remaining} laps to go",
        *(
            [f"the track was {state.status} as {tla} reached the pit entry, so a normal stop"]
            if not status
            else []
        ),
        *([race.pit_note] if race.pit_note else []),
        check or "",
        *(
            [f"{tla} really stopped {len(later)} more times; the simulation models one more stop"]
            if len(later) > 1
            else []
        ),
        "rivals stop when they really did (hindsight); tyre numbers measured over the whole race",
        *state.notes,
    ]
    return PitReview(
        driver=tla,
        lap=lap,
        status=state.status,
        old_compound=car.compound,
        new_compound=new,
        old_tyre_age=car.tyre_age,
        measured=_measured(race, state, lap, new),
        branches=branches,
        verdict=verdict,
        notes=[n for n in notes if n],
        ledger=ledger,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--meeting", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--lap", type=int, required=True, help="the driver's in-lap")
    ap.add_argument(
        "--what-if", action="store_true", help="a lap they didn't pit on: what if they had?"
    )
    ap.add_argument("--compound", help="--what-if: the compound fitted (default: their next one)")
    args = ap.parse_args()
    if args.what_if:
        r = what_if_stop(args.year, args.meeting, args.driver, args.lap, args.compound)
    else:
        r = review(args.year, args.meeting, args.driver, args.lap)
    print(
        f"{r.driver} {'had they pitted' if args.what_if else 'pitted'} at the end of lap {r.lap} "
        f"({r.status}): {r.old_compound} ({r.old_tyre_age} laps) -> {r.new_compound}"
    )
    print("measured:", r.measured)
    print(f"{'option':28} {'exp. pos':>8} {'win':>6} {'podium':>7} {'better':>7} {'worse':>6}")
    for b in r.branches:
        print(
            f"{b.label:28} {b.expected:8.2f} {b.p_win:6.0%} {b.p_podium:7.0%} "
            f"{b.p_better_than_actual:7.0%} {b.p_worse_than_actual:6.0%}"
        )
    if r.ledger:
        print("time ledger:", r.ledger)
    print("\n" + r.verdict)
    for n in r.notes:
        print("-", n)


if __name__ == "__main__":
    main()
