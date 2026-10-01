"""Learned undercut model: P(the chaser ends up ahead) from what's known the lap it pits.

L2-regularised logistic regression on the attempts in undercut_data.py (2023 onwards), scored
leave-one-race-out against the base rate ("undercuts work ~1/3 of the time"). The simulator's
version had no skill (Brier 0.234 vs 0.217 base rate on 2026), so this has to beat the base rate
on races it hasn't seen before it's used for alerts.

Usage: python -m src.sim.undercut_model
"""

import math

import numpy as np

from src.sim.undercut_data import COMPOUND_RANK, Attempt, build

FEATURES = (
    "gap_s",
    "age_diff",  # chaser's tyre age minus the car ahead's (older tyres gain more from fresh)
    "chaser_age",
    "race_fraction",  # how far into the race
    "pace_delta_s",  # chaser's recent pace minus the car ahead's (negative = chaser faster)
    "pace_unknown",
    "softer_than_ahead",  # new compound softer than the car ahead's current one
    "pit_loss_s",
    "regs_2026",  # the 2026 regulations: undercuts worked far less (33% vs 45-52% in 2023-25)
)
L2 = 1.0


def features(a: Attempt) -> list[float]:
    new = COMPOUND_RANK.get(a.new_compound or "", 1)
    ahead = COMPOUND_RANK.get(a.ahead_compound, 1)
    return [
        a.gap_s,
        a.chaser_age - a.ahead_age,
        a.chaser_age,
        a.lap / max(a.total_laps, 1),
        a.pace_delta_s if a.pace_delta_s is not None else 0.0,
        float(a.pace_delta_s is None),
        float(new < ahead),
        a.pit_loss_s,
        float(a.year >= 2026),
    ]


class Logistic:
    def __init__(self, l2: float = L2) -> None:
        self.l2 = l2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Logistic":
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0)
        self.sd[self.sd == 0] = 1
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        w = np.zeros(Z.shape[1])
        reg = self.l2 * np.eye(Z.shape[1])
        reg[0, 0] = 0  # no penalty on the intercept
        for _ in range(50):  # Newton / IRLS
            p = 1 / (1 + np.exp(-Z @ w))
            grad = Z.T @ (p - y) + reg @ w
            hess = Z.T @ (Z * (p * (1 - p))[:, None]) + reg
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        return 1 / (1 + np.exp(-Z @ self.w))


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(pp > pn) + 0.5 * (pp == pn) for pp in pos for pn in neg]))


def evaluate(rows: list[Attempt]) -> dict:
    rows = [r for r in rows if r.worked is not None]
    races = sorted({(r.year, r.race) for r in rows})
    X = np.array([features(r) for r in rows], dtype=float)
    y = np.array([r.worked for r in rows], dtype=float)
    key = [(r.year, r.race) for r in rows]
    p_model, p_base = np.empty(len(rows)), np.empty(len(rows))
    for race in races:
        test = np.array([k == race for k in key])
        model = Logistic().fit(X[~test], y[~test])
        p_model[test] = model.predict(X[test])
        # The stricter baseline: the rate in the same season's other races.
        same = np.array([k[0] == race[0] for k in key]) & ~test
        p_base[test] = y[same].mean() if same.any() else y[~test].mean()

    def scores(p):
        p = np.clip(p, 1e-3, 1 - 1e-3)
        return {
            "brier": round(float(np.mean((p - y) ** 2)), 4),
            "log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
            "auc": round(_auc(p, y), 3),
        }

    table = []
    for lo, hi in ((0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.7), (0.7, 1.01)):
        m = (p_model >= lo) & (p_model < hi)
        if m.any():
            table.append(
                (
                    f"{lo:.2f}-{min(hi, 1):.2f}",
                    int(m.sum()),
                    round(float(p_model[m].mean()), 3),
                    round(float(y[m].mean()), 3),
                )
            )
    by_year = {}
    for yr in sorted({r.year for r in rows}):
        m = np.array([r.year == yr for r in rows])
        pm, pb, yy = np.clip(p_model[m], 1e-3, 1 - 1e-3), p_base[m], y[m]
        by_year[yr] = {
            "n": int(m.sum()),
            "brier_model": round(float(np.mean((pm - yy) ** 2)), 4),
            "brier_base": round(float(np.mean((pb - yy) ** 2)), 4),
        }
    full = Logistic().fit(X, y)
    return {
        "attempts": len(rows),
        "races": len(races),
        "base_rate": round(float(y.mean()), 3),
        "model": scores(p_model),
        "base": scores(p_base),
        "by_year": by_year,
        "reliability": table,
        "weights": dict(
            zip(("intercept", *FEATURES), [round(float(w), 3) for w in full.w], strict=True)
        ),
    }


def main() -> None:
    result = evaluate(build([2023, 2024, 2025, 2026]))
    print(
        f"{result['attempts']} resolved attempts in {result['races']} races; base rate {result['base_rate']:.0%}"
    )
    print("leave-one-race-out  model:", result["model"], " base rate:", result["base"])
    for yr, s in result["by_year"].items():
        print(
            f"  {yr}: n {s['n']:3}  Brier model {s['brier_model']:.4f}  base {s['brier_base']:.4f}"
        )
    print("reliability (bin, n, predicted, observed):")
    for row in result["reliability"]:
        print("  ", row)
    print("weights (standardised features):", result["weights"])
    if not math.isnan(result["model"]["auc"]):
        better = result["model"]["brier"] < result["base"]["brier"]
        print("\nbeats the base rate" if better else "\ndoes NOT beat the base rate")


if __name__ == "__main__":
    main()
