"""Probability calibration for the pre-race simulator: does "30% to win" win 30% of the time?

The simulator's win / podium / points probabilities are counts over Monte Carlo runs, so they
are only as honest as its noise model. This fits one temperature per horizon on the season's
races (leave-one-race-out, so the reported numbers are out of sample) and reports reliability
tables before and after:

  win      multi-class temperature:  p_i ∝ p_i^(1/T)           (sums to 1 over the field)
  podium   binary temperature on the logit, rescaled to sum to 3 (points: to 10)

One parameter per horizon because a season has ~15 races (15 winners): isotonic regression
would memorise them. T > 1 means the simulator is overconfident and gets flattened.

Scored against luck-adjusted results with DNFs removed, like the backtest.

Usage: python -m src.sim.calibrate [--year 2026] [--sims 2000]
"""

import argparse
import math
from dataclasses import dataclass

import numpy as np

from src.sim.backtest import actual_order
from src.sim.inputs import build_inputs
from src.sim.race import simulate
from src.warehouse.queries import connect

TEMPERATURES = [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 4.0, 5.0, 6.0, 8.0]
HORIZONS = {"win": 1, "podium": 3, "points": 10}


@dataclass(frozen=True)
class Calibration:
    """Fitted temperatures (see the module docstring); 1.0 = the raw simulator."""

    win: float = 1.0
    podium: float = 1.0
    points: float = 1.0


# Fitted on 2026 (15 races, luck-adjusted, leave-one-race-out): winner log-loss 1.57 → 1.53,
# podium log-loss 0.286 → 0.245, points 0.736 → 0.426. The raw simulator is overconfident,
# most of all deep in the field (cars it gave < 5% for points scored 24% of the time).
DEFAULT_CALIBRATION = Calibration(win=1.8, podium=2.2, points=4.0)


def _floor(p: np.ndarray, sims: int) -> np.ndarray:
    # A car that never won in N runs gets half a win: log-loss needs p > 0.
    return np.clip(p, 0.5 / sims, 1 - 0.5 / sims)


def scale(p: np.ndarray, horizon: str, temperature: float, sims: int) -> np.ndarray:
    """Apply a temperature to one race's per-car probabilities for `horizon`."""
    p = _floor(np.asarray(p, dtype=float), sims)
    k = HORIZONS[horizon]
    if k == 1:
        logits = np.log(p) / temperature
        out = np.exp(logits - logits.max())
        return out / out.sum()
    if len(p) <= k:
        return np.ones_like(p)  # everyone makes the top k
    logits = np.log(p / (1 - p)) / temperature
    # Keep the expected count (3 podium places, 10 points places) by a shift on the logit.
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logits + mid)))).sum() > k:
            hi = mid
        else:
            lo = mid
    return 1 / (1 + np.exp(-(logits + lo)))


def apply(table: list[dict], calib: Calibration, sims: int) -> list[dict]:
    """Calibrated copy of a `Prediction.table()`."""
    out = [dict(row) for row in table]
    for horizon in HORIZONS:
        key = f"p_{horizon}"
        scaled = scale([r[key] for r in out], horizon, getattr(calib, horizon), sims)
        for row, p in zip(out, scaled, strict=True):
            row[key] = round(float(p), 3)
    return out


@dataclass(frozen=True)
class RaceProbs:
    location: str
    p: dict[str, np.ndarray]  # horizon -> per-car raw probability (finishers only)
    hit: dict[str, np.ndarray]  # horizon -> 0/1 outcome per car


def collect(year: int, sims: int) -> list[RaceProbs]:
    con = connect()
    locations = [
        r[0]
        for r in con.execute(
            """SELECT location FROM races WHERE year = ? AND session_name = 'Race'
               ORDER BY date_start""",
            [year],
        ).fetchall()
    ]
    out = []
    for location in locations:
        inputs = build_inputs(con, year, location)
        actual = actual_order(con, inputs)
        pred = simulate(inputs, sims=sims, seed=3)
        finished = ~np.isnan(actual)
        p, hit = {}, {}
        for horizon, k in HORIZONS.items():
            # Only finishers are scored; renormalise the win distribution over them since the
            # simulator (DNFs off) already assumes every car finishes.
            raw = pred.probability(k)[finished]
            p[horizon] = raw / raw.sum() * min(k, finished.sum()) if raw.sum() else raw
            hit[horizon] = (actual[finished] <= k).astype(int)
        out.append(RaceProbs(location, p, hit))
    return out


def race_loss(race: RaceProbs, horizon: str, temperature: float, sims: int) -> float:
    q = scale(race.p[horizon], horizon, temperature, sims)
    y = race.hit[horizon]
    if horizon == "win":
        return -math.log(max(float(q[int(np.argmax(y))]), 1e-4))
    q = np.clip(q, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def reliability(ps: np.ndarray, ys: np.ndarray, bins) -> list[tuple[str, int, float, float]]:
    rows = []
    for lo, hi in bins:
        mask = (ps >= lo) & (ps < hi)
        if mask.any():
            rows.append(
                (f"{lo:.2f}-{min(hi, 1):.2f}", int(mask.sum()), ps[mask].mean(), ys[mask].mean())
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--sims", type=int, default=2000)
    args = ap.parse_args()

    races = collect(args.year, args.sims)
    print(f"{len(races)} races, {args.sims} simulations each\n")
    bins = [(0, 0.05), (0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    fitted = {}
    for horizon in HORIZONS:
        held_out_raw, held_out_cal, chosen = [], [], []
        cal_ps, cal_ys = [], []
        for i, race in enumerate(races):
            train = [r for j, r in enumerate(races) if j != i]
            t = min(
                TEMPERATURES,
                key=lambda t: np.mean([race_loss(r, horizon, t, args.sims) for r in train]),
            )
            chosen.append(t)
            held_out_raw.append(race_loss(race, horizon, 1.0, args.sims))
            held_out_cal.append(race_loss(race, horizon, t, args.sims))
            cal_ps.append(scale(race.p[horizon], horizon, t, args.sims))
            cal_ys.append(race.hit[horizon])
        fitted[horizon] = min(
            TEMPERATURES,
            key=lambda t: np.mean([race_loss(r, horizon, t, args.sims) for r in races]),
        )
        loss_name = "winner log-loss" if horizon == "win" else "log-loss per car"
        print(
            f"{horizon:7} {loss_name}: raw {np.mean(held_out_raw):.3f} → "
            f"calibrated (LORO) {np.mean(held_out_cal):.3f}; "
            f"T per fold {sorted(set(chosen))}, T on all races {fitted[horizon]}"
        )
        raw_p = np.concatenate([r.p[horizon] for r in races])
        ys = np.concatenate([r.hit[horizon] for r in races])
        for name, ps, yy in (
            ("raw", raw_p, ys),
            ("calibrated", np.concatenate(cal_ps), np.concatenate(cal_ys)),
        ):
            print(f"  reliability ({name}): bin, n, mean predicted, observed rate")
            for row in reliability(ps, yy, bins):
                print(f"    {row[0]:10} {row[1]:4} {row[2]:6.3f} {row[3]:6.3f}")
        print()
    print("Calibration(" + ", ".join(f"{h}={t}" for h, t in fitted.items()) + ")")


if __name__ == "__main__":
    main()
