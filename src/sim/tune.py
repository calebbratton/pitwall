"""Tune simulator parameters with leave-one-race-out cross-validation.

For each held-out race, parameters are chosen on the other races only, then scored on the
held-out one — so the reported numbers are honest out-of-sample performance.

Usage: python -m src.sim.tune [--year 2026] [--sims 300]
"""

import argparse
import itertools
from dataclasses import replace

import numpy as np

from src.sim.backtest import actual_order, score
from src.sim.inputs import build_inputs
from src.sim.race import SimParams, simulate
from src.warehouse.queries import connect

GRID = {
    "quali_weight": [0.5, 0.7, 0.85, 1.0],
    "pass_threshold": [0.5, 0.8, 1.2],
    "start_noise": [0.3, 0.6, 1.0],
    "lap_noise": [0.2, 0.35],
}


def objective(s) -> float:
    """Higher is better: rank correlation, plus calibration on the winner (log-loss)."""
    return s.spearman - 0.1 * (s.winner_logloss or 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--sims", type=int, default=300)
    args = ap.parse_args()

    con = connect()
    locations = [
        r[0]
        for r in con.execute(
            """SELECT location FROM races WHERE year = ? AND session_name = 'Race'
               ORDER BY date_start""",
            [args.year],
        ).fetchall()
    ]
    races = []
    for loc in locations:
        inputs = build_inputs(con, args.year, loc)
        races.append((loc, inputs, actual_order(con, inputs)))

    combos = [dict(zip(GRID, values, strict=True)) for values in itertools.product(*GRID.values())]
    # table[c][r] = score of combo c on race r
    table = []
    for combo in combos:
        params = replace(SimParams(), **combo)
        row = []
        for _, inputs, actual in races:
            pred = simulate(inputs, params, sims=args.sims, seed=1)
            row.append(score(pred.expected_position(), actual, pred.probability(1)))
        table.append(row)
    print(f"evaluated {len(combos)} settings x {len(races)} races")

    held_out = []
    chosen = []
    for r in range(len(races)):
        train = [
            np.mean([objective(table[c][i]) for i in range(len(races)) if i != r])
            for c in range(len(combos))
        ]
        best = int(np.argmax(train))
        chosen.append(combos[best])
        held_out.append(table[best][r])

    ll = [s.winner_logloss for s in held_out if s.winner_logloss is not None]
    print(
        f"leave-one-race-out: ρ {np.mean([s.spearman for s in held_out]):+.3f}  "
        f"mae {np.mean([s.mae for s in held_out]):.2f}  "
        f"winners {sum(s.winner_hit for s in held_out)}/{len(held_out)}  "
        f"podium {np.mean([s.podium_overlap for s in held_out]):.2f}/3  "
        f"winner log-loss {np.mean(ll):.2f}"
    )
    overall = int(np.argmax([np.mean([objective(s) for s in row]) for row in table]))
    print("chosen per fold:", {str(k): v for k, v in _counts(chosen).items()})
    print("best on all races:", combos[overall])


def _counts(chosen: list[dict]) -> dict:
    counts: dict = {}
    for c in chosen:
        key = tuple(sorted(c.items()))
        counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    main()
