"""Backtest the race simulator on finished races, against naive baselines.

For each race: predict from pre-race data only (practice, sprint, qualifying, grid; season rates
exclude the race), then score against the classified result.

Usage: python -m src.sim.backtest [--year 2026] [--sims 1000]
"""

import argparse
import math
from dataclasses import dataclass

import numpy as np

from src.sim.inputs import WeekendInputs, build_inputs
from src.sim.race import SimParams, simulate
from src.warehouse.queries import connect


@dataclass(frozen=True)
class Score:
    spearman: float
    mae: float
    winner_hit: bool
    podium_overlap: int
    winner_logloss: float | None


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def actual_order(con, inputs: WeekendInputs) -> np.ndarray:
    """Finishing rank per input driver among classified finishers; NaN for DNF/DNS/DSQ.
    Retirements are unpredictable noise for a pace model, so scoring ignores them."""
    rows = {
        n: (pos, bool(dnf or dns or dsq))
        for n, pos, dnf, dns, dsq in con.execute(
            """SELECT driver_number, position, dnf, dns, dsq
               FROM raw_results WHERE session_key = ?""",
            [inputs.race_session_key],
        ).fetchall()
    }
    positions = np.array(
        [
            np.nan if (r := rows.get(d.number)) is None or r[1] or r[0] is None else r[0]
            for d in inputs.drivers
        ],
        dtype=float,
    )
    finished = ~np.isnan(positions)
    ranks = np.full(len(positions), np.nan)
    ranks[finished] = _ranks(positions[finished])
    return ranks


def score(predicted: np.ndarray, actual: np.ndarray, p_win: np.ndarray | None = None) -> Score:
    """Compare over classified finishers only (actual rank is NaN for non-finishers)."""
    finished = ~np.isnan(actual)
    pred_rank = _ranks(np.asarray(predicted, dtype=float)[finished])
    act = actual[finished]
    spearman = float(np.corrcoef(pred_rank, act)[0, 1])
    winner = int(np.argmin(act))
    logloss = None
    if p_win is not None:
        logloss = -math.log(max(float(np.asarray(p_win)[finished][winner]), 1e-3))
    return Score(
        spearman=spearman,
        mae=float(np.mean(np.abs(pred_rank - act))),
        winner_hit=int(np.argmin(pred_rank)) == winner,
        podium_overlap=len(set(np.argsort(pred_rank)[:3]) & set(np.argsort(act)[:3])),
        winner_logloss=logloss,
    )


def backtest(year: int = 2026, params: SimParams | None = None, sims: int = 1000, quiet=False):
    con = connect()
    meetings = con.execute(
        """SELECT DISTINCT location, min(date_start) OVER (PARTITION BY meeting_key) AS d
           FROM races WHERE year = ? AND session_name = 'Race' ORDER BY d""",
        [year],
    ).fetchall()
    results: dict[str, list[Score]] = {"sim": [], "grid": [], "quali": []}
    orders = {}
    for location, _ in meetings:
        inputs = build_inputs(con, year, location)
        orders[location] = (inputs, actual_order(con, inputs))
    for location, _ in meetings:
        inputs, actual = orders[location]
        # Probabilistic grid baseline: P(win | grid slot) from the season's OTHER races.
        wins = np.ones(len(inputs.drivers)) * 0.5  # Laplace-style smoothing
        for other, (o_inputs, o_actual) in orders.items():
            if other == location:
                continue
            winner = int(np.nanargmin(o_actual))
            slot = o_inputs.drivers[winner].grid - 1
            if slot < len(wins):
                wins[slot] += 1
        p_grid = np.array([wins[min(d.grid - 1, len(wins) - 1)] for d in inputs.drivers])
        p_grid = p_grid / p_grid.sum()
        pred = simulate(inputs, params, sims=sims)
        grid = np.array([d.grid for d in inputs.drivers], dtype=float)
        quali = np.array(
            [
                d.quali_delta_s if d.quali_delta_s is not None else 99 + d.grid
                for d in inputs.drivers
            ]
        )
        row = {
            "sim": score(pred.expected_position(), actual, pred.probability(1)),
            "grid": score(grid, actual, p_grid),
            "quali": score(quali, actual),
        }
        for k, v in row.items():
            results[k].append(v)
        if not quiet:
            s, g = row["sim"], row["grid"]
            print(
                f"{location:18} sim ρ {s.spearman:+.2f} mae {s.mae:4.1f} win {'✓' if s.winner_hit else '·'} "
                f"pod {s.podium_overlap}/3 | grid ρ {g.spearman:+.2f} mae {g.mae:4.1f} "
                f"win {'✓' if g.winner_hit else '·'} pod {g.podium_overlap}/3"
            )
    return results


def summarise(results) -> dict[str, dict[str, float]]:
    out = {}
    for name, scores in results.items():
        ll = [s.winner_logloss for s in scores if s.winner_logloss is not None]
        out[name] = {
            "spearman": float(np.mean([s.spearman for s in scores])),
            "mae": float(np.mean([s.mae for s in scores])),
            "winners": sum(s.winner_hit for s in scores),
            "podium": float(np.mean([s.podium_overlap for s in scores])),
            "winner_logloss": float(np.mean(ll)) if ll else float("nan"),
            "races": len(scores),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--sims", type=int, default=1000)
    args = ap.parse_args()
    summary = summarise(backtest(args.year, sims=args.sims))
    print()
    for name, m in summary.items():
        print(
            f"{name:6} ρ {m['spearman']:+.3f}  mae {m['mae']:.2f}  winners {m['winners']}/{m['races']}"
            f"  podium {m['podium']:.2f}/3  winner log-loss {m['winner_logloss']:.2f}"
        )


if __name__ == "__main__":
    main()
