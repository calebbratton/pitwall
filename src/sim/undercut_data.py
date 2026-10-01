"""Undercut attempts from archived races (2023 onwards): the dataset for a learned model.

For each race the live pipeline (RaceMonitor) is replayed; at the start of every green lap,
every close pair (chaser within MAX_GAP_S behind the car ahead, both still owing their
mandatory compound, i.e. before the first stop) is a candidate. An *attempt* is a candidate
whose chaser pitted at the end of that lap while the car ahead didn't. Outcome: was the chaser
ahead RESOLVE_LAPS laps after the car ahead's own stop. Also recorded: how many laps the car ahead
took to respond, which replaces the simulator's "covers next lap" assumption.

Features use only what was known at the time (gaps, tyres, laps left, recent pace, circuit pit
loss); no luck adjustment is needed because the target is a two-car exchange around the stops.

Usage: python -m src.sim.undercut_data [--years 2023 2024 2025 2026]  (cached to data/warehouse)
"""

import argparse
import itertools
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from src.livetiming.archive import ArchiveSession, list_sessions
from src.livetiming.monitor import STRATEGY_TOPICS, LapRecord, RaceMonitor, _circuit_for

MAX_GAP_S = 4.0
RESOLVE_LAPS = 2
CACHE = Path("data/warehouse/undercut_attempts.jsonl")
COMPOUND_RANK = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}


@dataclass(frozen=True)
class Attempt:
    year: int
    race: str
    lap: int
    total_laps: int
    chaser: str
    ahead: str
    gap_s: float
    chaser_age: int
    ahead_age: int
    chaser_compound: str
    ahead_compound: str
    new_compound: str | None  # what the chaser fitted
    pace_delta_s: float | None  # chaser's recent clean-lap median minus the car ahead's
    pit_loss_s: float
    response_laps: int | None  # car ahead's stop minus the chaser's (None = never stopped)
    worked: bool | None  # chaser ahead after both stopped (None if unresolved)


def _recent(records: list[LapRecord], n: int = 5) -> float | None:
    clean = [
        r.time_s
        for r in records
        if r.time_s and not (r.pit_in or r.pit_out or r.neutralised) and r.lap > 1
    ][-n:]
    return statistics.median(clean) if len(clean) >= 3 else None


def _first_stop(records: list[LapRecord]) -> tuple[int, str | None] | None:
    for prev, cur in itertools.pairwise(records):
        if prev.pit_in and cur.stint > prev.stint:
            return prev.lap, cur.compound
    return None


def race_attempts(year: int, race: dict) -> list[Attempt]:
    messages = list(ArchiveSession(race["path"]).messages(STRATEGY_TOPICS))
    circuit = _circuit_for(messages)
    pit_loss = circuit.pit_loss.green if circuit and circuit.pit_loss else 22.0
    monitor = RaceMonitor()
    order_at: dict[int, dict[str, int]] = {}
    candidates: dict[int, list[dict]] = {}  # leader lap -> close pairs at its start
    last = None
    for m in messages:
        monitor.feed(m)
        if m.topic != "LapCount":
            continue
        snap = monitor.snapshot()
        lap = snap.current_lap
        if not lap or lap == last:
            continue
        last = lap
        order_at[lap] = {d.number: d.position for d in snap.drivers if d.position}
        if snap.track_status != "GREEN" or not snap.total_laps:
            continue
        running = [
            d
            for d in snap.drivers
            if not d.retired and not d.in_pit and d.gap_to_leader_s is not None
        ]
        running.sort(key=lambda d: d.position or 99)
        pairs = []
        for ahead, chaser in itertools.pairwise(running):
            gap = (chaser.gap_to_leader_s or 0) - (ahead.gap_to_leader_s or 0)
            if not (0 <= gap <= MAX_GAP_S):
                continue
            if not (chaser.needs_second_compound and ahead.needs_second_compound):
                continue
            if chaser.pit_stops or ahead.pit_stops:
                continue
            ca, aa = (
                _recent(monitor.laps.get(chaser.number, [])),
                _recent(monitor.laps.get(ahead.number, [])),
            )
            pairs.append(
                {
                    "chaser": chaser,
                    "ahead": ahead,
                    "gap": gap,
                    "pace_delta": None if ca is None or aa is None else ca - aa,
                    "total": snap.total_laps,
                }
            )
        candidates[lap] = pairs

    # Resolve with the whole race known.
    out = []
    final_lap = max(order_at) if order_at else 0
    for lap, pairs in candidates.items():
        for p in pairs:
            c, a = p["chaser"], p["ahead"]
            c_stop = _first_stop(monitor.laps.get(c.number, []))
            a_stop = _first_stop(monitor.laps.get(a.number, []))
            # Attempt: the chaser's in-lap is the lap just started (leader's lap `lap`) and the
            # car ahead hadn't stopped by then. Chaser's own lap count can lag the leader's.
            if not c_stop or not (lap - 1 <= c_stop[0] <= lap + 1):
                continue
            if a_stop and a_stop[0] <= c_stop[0]:
                continue
            response = a_stop[0] - c_stop[0] if a_stop else None
            worked = None
            if a_stop:
                judge = min(a_stop[0] + RESOLVE_LAPS, final_lap)
                pos = order_at.get(judge) or {}
                if c.number in pos and a.number in pos:
                    worked = pos[c.number] < pos[a.number]
            out.append(
                Attempt(
                    year=year,
                    race=race["meeting"],
                    lap=lap,
                    total_laps=p["total"],
                    chaser=c.tla,
                    ahead=a.tla,
                    gap_s=round(p["gap"], 3),
                    chaser_age=c.tyre_age_laps or 0,
                    ahead_age=a.tyre_age_laps or 0,
                    chaser_compound=c.compound or "",
                    ahead_compound=a.compound or "",
                    new_compound=c_stop[1],
                    pace_delta_s=None if p["pace_delta"] is None else round(p["pace_delta"], 3),
                    pit_loss_s=pit_loss,
                    response_laps=response,
                    worked=worked,
                )
            )
    # One attempt per (chaser, ahead): the lap closest to the stop.
    best: dict[tuple[str, str], Attempt] = {}
    for at in out:
        key = (at.chaser, at.ahead)
        if key not in best or at.lap > best[key].lap:
            best[key] = at
    return list(best.values())


def build(years: list[int], refresh: bool = False) -> list[Attempt]:
    if CACHE.exists() and not refresh:
        rows = [Attempt(**json.loads(line)) for line in CACHE.read_text().splitlines()]
        if {r.year for r in rows} >= set(years):
            return [r for r in rows if r.year in years]
    rows: list[Attempt] = []
    for year in years:
        for race in list_sessions(year):
            try:
                found = race_attempts(year, race)
            except Exception as e:  # noqa: BLE001 — a broken archive race shouldn't stop the set
                print(f"  skip {year} {race['meeting']}: {type(e).__name__}: {e}")
                continue
            rows += found
            print(f"  {year} {race['meeting']:28} {len(found)} attempts", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text("".join(json.dumps(asdict(r)) + "\n" for r in rows))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    rows = build(args.years, args.refresh)
    resolved = [r for r in rows if r.worked is not None]
    print(
        f"\n{len(rows)} attempts, {len(resolved)} resolved, worked {sum(r.worked for r in resolved)}"
    )
    for y in sorted({r.year for r in rows}):
        ry = [r for r in resolved if r.year == y]
        if ry:
            print(f"  {y}: {len(ry)} resolved, worked {sum(r.worked for r in ry) / len(ry):.0%}")
    resp = [r.response_laps for r in rows if r.response_laps is not None]
    never = sum(r.response_laps is None for r in rows)
    if resp:
        print(
            "car ahead's response (laps after the chaser's stop):",
            {k: resp.count(k) for k in sorted(set(resp))[:10]},
            f"never: {never}",
        )


if __name__ == "__main__":
    main()
