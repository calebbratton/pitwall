"""Backtest and calibrate the in-race "who wins from here" predictor on a season's races.

Each race is replayed through the live pipeline (RaceMonitor). The race state is captured at
every SC / VSC deployment and at green-flag checkpoints (1/3 and 2/3 distance). Each capture is
scored against the final classification with the luck of LATER neutralisations removed (the one
underway at a capture is part of the situation being predicted — see luck.py `after_lap`).

Calibration uses leave-one-race-out: settings are chosen on the other races, scored on the held
out one. Checkpoints are cached in data/warehouse/inrace_checkpoints_<year>.pkl.

Usage: python -m src.sim.inrace_backtest [--year 2026] [--sims 400]
"""

import argparse
import itertools
import math
import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from src.livetiming.archive import ArchiveSession, list_sessions
from src.livetiming.monitor import STRATEGY_TOPICS, RaceMonitor, _circuit_for
from src.livetiming.snapshot import TRACK_STATUS
from src.sim.inrace import InRaceParams, NotEnoughData, RaceState, race_state, simulate_from
from src.sim.luck import neutralisation_luck
from src.warehouse.queries import connect

CACHE = Path("data/warehouse")
GRID = {
    "pace_sigma": [0.0, 0.15, 0.3, 0.5],
    "restart_noise": [0.0, 0.5, 1.0, 2.0],
    "take_neutralised_stop": [0.75, 0.9],
}


@dataclass(frozen=True)
class Checkpoint:
    race: str
    kind: str  # "SC" | "VSC" | "GREEN"
    state: RaceState
    target: dict[str, int]  # car number -> luck-adjusted finishing rank (finishers only)


def _race_keys(con, year: int) -> dict[str, int]:
    """Race start date -> warehouse race session_key."""
    rows = con.execute(
        """SELECT strftime(date_start, '%Y-%m-%d'), session_key FROM races
           WHERE year = ? AND session_name = 'Race'""",
        [year],
    ).fetchall()
    return dict(rows)


def capture(year: int, age_curves: bool = False, hazards: bool = False) -> list[Checkpoint]:
    """`age_curves`: age tyres with the season's tyre-age curves, fitted without the race being
    captured (no leakage)."""
    from src.models.tyre_curves import fit_curves

    con = connect()
    keys = _race_keys(con, year)
    checkpoints: list[Checkpoint] = []
    for race in list_sessions(year):
        race_sk = keys.get(race["date"])
        if race_sk is None:
            continue
        session = ArchiveSession(race["path"])
        messages = list(session.messages(STRATEGY_TOPICS))
        circuit = _circuit_for(messages)
        loss = (
            (circuit.pit_loss.green, circuit.pit_loss.safety_car)
            if circuit and circuit.pit_loss
            else (22.0, 13.5)
        )
        curves = fit_curves(con, year, exclude=(race_sk,)) if age_curves else None
        blend = None
        if hazards:
            from src.sim.pit_hazard import LIVE_FEATURES, Blend, columns, load

            rows = load()
            location = con.execute(
                "SELECT location FROM races WHERE session_key = ?", [race_sk]
            ).fetchone()[0]
            keep = np.array([rows.races[i] != (year, location) for i in rows.race])
            blend = Blend(rows.X[keep][:, columns(LIVE_FEATURES)], rows.y[keep])
        monitor = RaceMonitor()
        status = None
        green_marks: set[int] = set()
        taken: list[tuple[str, RaceState]] = []
        for message in messages:
            monitor.feed(message)
            kind = None
            if message.topic == "TrackStatus":
                new = TRACK_STATUS.get(str(message.data.get("Status")))
                if new in ("SAFETY_CAR", "VSC") and new != status:
                    kind = "SC" if new == "SAFETY_CAR" else "VSC"
                status = new
            elif message.topic == "LapCount" and status == "GREEN":
                lap, total = message.data.get("CurrentLap"), monitor.snapshot().total_laps
                if lap and total:
                    for mark in (round(total / 3), round(2 * total / 3)):
                        if lap == mark and mark not in green_marks:
                            green_marks.add(mark)
                            kind = "GREEN"
            if kind:
                try:
                    state = race_state(
                        monitor,
                        monitor.snapshot(),
                        pit_loss=loss,
                        curves=curves,
                        age_curves=age_curves,
                    )
                    if blend is not None:
                        from dataclasses import replace as _replace

                        from src.sim.pit_hazard import future_hazards

                        hz = future_hazards(monitor.snapshot(), monitor, blend)
                        state = _replace(state, stop_hazard={k: v.tolist() for k, v in hz.items()})
                    taken.append((kind, state))
                except NotEnoughData:
                    pass
        for kind, state in taken:
            report = neutralisation_luck(con, race_sk, loss[0], after_lap=state.lap)
            target = {str(d): i + 1 for i, d in enumerate(report.adjusted_order)}
            checkpoints.append(Checkpoint(race["meeting"], kind, state, target))
        print(f"  {race['meeting']:32} {len(taken)} checkpoints")
    return checkpoints


def load_or_capture(
    year: int, refresh: bool = False, age_curves: bool = False, hazards: bool = False
) -> list[Checkpoint]:
    suffix = ("_curves" if age_curves else "") + ("_hazard" if hazards else "")
    path = CACHE / f"inrace_checkpoints_{year}{suffix}.pkl"
    if path.exists() and not refresh:
        return pickle.loads(path.read_bytes())
    checkpoints = capture(year, age_curves, hazards)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(checkpoints))
    return checkpoints


@dataclass(frozen=True)
class CheckpointScore:
    winner_prob: float  # probability given to the (luck-adjusted) winner
    brier: float  # mean over cars of (p_win - won)^2
    spearman: float
    p_win: list[float]  # per car, for the calibration table
    won: list[int]


def score_checkpoint(cp: Checkpoint, calib: InRaceParams, sims: int) -> CheckpointScore | None:
    out = simulate_from(cp.state, sims=sims, seed=11, calib=calib)
    rows = [
        (r, cp.target.get(car.number)) for r, car in zip(out["table"], cp.state.cars, strict=True)
    ]
    rows = [(r, t) for r, t in rows if t is not None]
    if len(rows) < 5:
        return None
    ranks = np.array([t for _, t in rows], dtype=float)
    ranks = np.argsort(np.argsort(ranks)) + 1  # re-rank among the cars still running
    expected = np.array([r["expected"] for r, _ in rows])
    pred_rank = np.argsort(np.argsort(expected)) + 1
    p_win = [r["p_win"] for r, _ in rows]
    won = [int(k == 1) for k in ranks]
    return CheckpointScore(
        winner_prob=p_win[won.index(1)],
        brier=float(np.mean([(p - w) ** 2 for p, w in zip(p_win, won, strict=True)])),
        spearman=float(np.corrcoef(pred_rank, ranks)[0, 1]),
        p_win=p_win,
        won=won,
    )


def summarise(scores: list[CheckpointScore]) -> dict[str, float]:
    return {
        "log_loss": float(np.mean([-math.log(max(s.winner_prob, 1e-3)) for s in scores])),
        "brier": float(np.mean([s.brier for s in scores])),
        "spearman": float(np.mean([s.spearman for s in scores])),
        "n": len(scores),
    }


def calibration_table(scores: list[CheckpointScore]) -> list[tuple[str, int, float, float]]:
    """(bin, count, mean predicted P(win), observed win rate) — the reliability diagram."""
    p = np.array([x for s in scores for x in s.p_win])
    w = np.array([x for s in scores for x in s.won])
    bins = [(0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]
    table = []
    for lo, hi in bins:
        mask = (p >= lo) & (p < hi)
        if mask.any():
            table.append(
                (
                    f"{lo:.2f}-{min(hi, 1):.2f}",
                    int(mask.sum()),
                    float(p[mask].mean()),
                    float(w[mask].mean()),
                )
            )
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--refresh", action="store_true", help="re-capture checkpoints")
    ap.add_argument("--age-curves", action="store_true", help="season tyre-age curves (LORO)")
    ap.add_argument("--hazards", action="store_true", help="rival pit-timing model (LORO)")
    args = ap.parse_args()

    checkpoints = load_or_capture(args.year, args.refresh, args.age_curves, args.hazards)
    races = sorted({cp.race for cp in checkpoints})
    print(
        f"{len(checkpoints)} checkpoints in {len(races)} races: "
        + ", ".join(
            f"{k} {sum(cp.kind == k for cp in checkpoints)}" for k in ("SC", "VSC", "GREEN")
        )
    )

    combos = [dict(zip(GRID, v, strict=True)) for v in itertools.product(*GRID.values())]
    # scores[c][i] for combo c and checkpoint i
    scores = [
        [score_checkpoint(cp, replace(InRaceParams(), **combo), args.sims) for cp in checkpoints]
        for combo in combos
    ]

    def log_loss(combo_idx: int, keep) -> float:
        vals = [s for s, cp in zip(scores[combo_idx], checkpoints, strict=True) if s and keep(cp)]
        return summarise(vals)["log_loss"] if vals else math.inf

    baseline_idx = combos.index(
        {"pace_sigma": 0.0, "restart_noise": 0.0, "take_neutralised_stop": 0.9}
    )
    baseline = [s for s in scores[baseline_idx] if s]
    held_out: list[CheckpointScore] = []
    chosen: dict[str, int] = {}
    for race in races:
        best = min(range(len(combos)), key=lambda c, r=race: log_loss(c, lambda cp: cp.race != r))
        chosen[str(combos[best])] = chosen.get(str(combos[best]), 0) + 1
        held_out += [
            s for s, cp in zip(scores[best], checkpoints, strict=True) if s and cp.race == race
        ]

    for name, vals in (("current (no uncertainty)", baseline), ("calibrated (LORO)", held_out)):
        m = summarise(vals)
        print(
            f"{name:26} winner log-loss {m['log_loss']:.3f}  Brier {m['brier']:.4f}  "
            f"rank corr {m['spearman']:+.3f}  (n={m['n']})"
        )
    print("chosen per fold:", chosen)
    best_all = min(range(len(combos)), key=lambda c: log_loss(c, lambda cp: True))
    print("best on all races:", combos[best_all])
    for name, vals in (("current", baseline), ("calibrated", held_out)):
        print(f"\nreliability ({name}): bin, n, mean predicted P(win), observed win rate")
        for row in calibration_table(vals):
            print(f"  {row[0]:10} {row[1]:5} {row[2]:6.3f} {row[3]:6.3f}")


if __name__ == "__main__":
    main()
