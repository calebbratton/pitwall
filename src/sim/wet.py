"""Wet races: a separate simulator setting, and forecast-weighted predictions.

In the rain, qualifying pace predicts much less (2025-26: rank corr 0.65 wet vs 0.83 dry): car
advantages shrink, mistakes and strategy calls (when to fit intermediates) shuffle the order.
The wet setting is the same simulator with pace gaps scaled down and more lap / start noise,
tuned leave-one-race-out on the wet races (>= 10 cars on intermediates or wets) of 2023 onwards,
scored like the dry tuning (luck-adjusted results: SC/VSC timing and DNFs removed).

With a rain probability p (from the forecast), a prediction runs p of its simulations wet and
the rest dry — the mixture is what the probabilities should reflect before anyone knows.

Usage: python -m src.sim.wet [--years 2023 2024 2025 2026]
"""

import argparse
import itertools
from dataclasses import replace

import numpy as np

from src.sim.backtest import actual_order, score
from src.sim.inputs import WeekendInputs, build_inputs
from src.sim.race import Prediction, SimParams, simulate
from src.warehouse.queries import connect

WET_GRID = {
    "race_pace_scale": [0.3, 0.5, 0.7, 0.9],
    "lap_noise": [0.35, 0.9, 1.5, 2.5],
    "start_noise": [2.0, 4.0],
}
# Tuned on the 8 wet races of 2023-25 (python -m src.sim.wet, 2026-09-30): best is smaller pace
# gaps (race_pace_scale 0.3), but leave-one-race-out it does NOT beat the dry setting (rank corr
# 0.714 vs 0.709, winner log-loss 2.08 vs 2.00): wet races weren't much less predictable than
# dry ones. So forecasts don't change predictions automatically; `predict --rain` is a what-if.
WET_PARAMS = replace(SimParams(), race_pace_scale=0.3)
MIN_WET_CARS = 10


def wet_races(con, years: list[int]) -> list[tuple[int, str]]:
    """(year, location) of races where most of the field ran intermediates or wets."""
    return con.execute(
        f"""SELECT r.year, r.location FROM races r JOIN raw_stints s USING (session_key)
            WHERE r.session_name = 'Race' AND r.year IN ({",".join("?" * len(years))})
              AND s.compound IN ('INTERMEDIATE', 'WET')
            GROUP BY 1, 2, r.date_start HAVING count(DISTINCT s.driver_number) >= ?
            ORDER BY r.date_start""",
        [*years, MIN_WET_CARS],
    ).fetchall()


def simulate_weather(
    inputs: WeekendInputs,
    p_rain: float,
    sims: int = 5000,
    dry: SimParams | None = None,
    wet: SimParams = WET_PARAMS,
    seed: int = 0,
) -> Prediction:
    """Forecast-weighted prediction: round(p_rain × sims) wet simulations, the rest dry."""
    n_wet = round(min(max(p_rain, 0.0), 1.0) * sims)
    parts = []
    if sims - n_wet:
        parts.append(simulate(inputs, dry, sims=sims - n_wet, seed=seed).positions)
    if n_wet:
        parts.append(simulate(inputs, wet, sims=n_wet, seed=seed + 1).positions)
    return Prediction([d.tla for d in inputs.drivers], np.vstack(parts))


def objective(s) -> float:
    """Same as the dry tuner: rank correlation plus winner calibration."""
    return s.spearman - 0.1 * (s.winner_logloss or 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    ap.add_argument("--sims", type=int, default=400)
    args = ap.parse_args()
    con = connect()
    races = []
    for year, location in wet_races(con, args.years):
        try:
            inputs = build_inputs(con, year, location)
        except Exception as e:  # noqa: BLE001 — incomplete weekend
            print(f"  skip {year} {location}: {e}")
            continue
        races.append((f"{year} {location}", inputs, actual_order(con, inputs)))
    print(f"{len(races)} wet races: {', '.join(r[0] for r in races)}")

    combos = [dict(zip(WET_GRID, v, strict=True)) for v in itertools.product(*WET_GRID.values())]
    table = [
        [
            score(
                (
                    p := simulate(inputs, replace(SimParams(), **c), sims=args.sims, seed=1)
                ).expected_position(),
                actual,
                p.probability(1),
            )
            for _, inputs, actual in races
        ]
        for c in combos
    ]
    dry = combos.index({"race_pace_scale": 0.9, "lap_noise": 0.35, "start_noise": 2.0})
    held_out, chosen = [], []
    for r in range(len(races)):
        best = max(
            range(len(combos)),
            key=lambda c, r=r: np.mean(
                [objective(table[c][i]) for i in range(len(races)) if i != r]
            ),
        )
        chosen.append(str(combos[best]))
        held_out.append(table[best][r])
    for name, scores in (("dry settings", table[dry]), ("wet (LORO)", held_out)):
        ll = [s.winner_logloss for s in scores if s.winner_logloss is not None]
        print(
            f"{name:14} rank corr {np.mean([s.spearman for s in scores]):+.3f}  "
            f"winners {sum(s.winner_hit for s in scores)}/{len(scores)}  "
            f"winner log-loss {np.mean(ll):.2f}"
        )
    print("chosen per fold:", {c: chosen.count(c) for c in set(chosen)})
    best_all = max(range(len(combos)), key=lambda c: np.mean([objective(s) for s in table[c]]))
    print("best on all wet races:", combos[best_all])
    for name, inputs, actual in races:
        s = table[best_all][races.index((name, inputs, actual))]
        print(f"  {name:28} rank corr {s.spearman:+.2f}")


if __name__ == "__main__":
    main()
