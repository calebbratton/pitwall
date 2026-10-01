"""Rival pit timing: P(a car pits on this lap), learned from every race lap 2023 onwards.

The north-star model: "McLaren are likely to bring Piastri in within 2 laps". A discrete-time
hazard: one row per car per lap (from lap 2), label = the car pits at the end of that lap. The
features are what a rival pit wall knows during that lap: tyre age and compound, stint, how far
into the race, a SC/VSC/red flag now, whether the cars just ahead/behind or the teammate pitted
last lap (covering an undercut, double stacks), how much of the field has stopped, the car's tyre
age relative to the field, and the gaps around it.

Scored leave-one-race-out (log-loss per car-lap) against a tyre-age-only baseline.

Usage: python -m src.sim.pit_hazard
"""

import functools
import statistics
from dataclasses import dataclass

import numpy as np

from src.sim.undercut_model import Logistic
from src.warehouse.queries import connect

COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
FEATURES = (
    "age",
    "age_sq",
    "first_stint",
    "race_fraction",
    "laps_left_lt5",
    "sc",
    "vsc",
    "red",
    "ahead_pitted_last_lap",
    "behind_pitted_last_lap",
    "teammate_pitted_last_lap",
    "teammate_pitting_now",  # unknown in advance... excluded from the live feature set below
    "share_stopped",
    "age_vs_field",
    "gap_ahead",
    "gap_behind",
    "position",
    "raining",
    *(f"compound_{c.lower()}" for c in COMPOUNDS),
    "regs_2026",
)
LIVE_FEATURES = tuple(f for f in FEATURES if f != "teammate_pitting_now")
BASELINE = ("age", "age_sq", "first_stint", "race_fraction")


@dataclass
class Rows:
    X: np.ndarray  # [n, len(FEATURES)]
    y: np.ndarray
    race: np.ndarray  # race index per row
    races: list[tuple[int, str]]
    car: np.ndarray  # car number per row (rows are in lap order within a race)


@functools.cache
def load(years: tuple[int, ...] = (2023, 2024, 2025, 2026)) -> Rows:
    con = connect()
    races = con.execute(
        f"""SELECT session_key, year, location FROM races WHERE session_name = 'Race'
            AND year IN ({",".join("?" * len(years))}) ORDER BY date_start""",
        list(years),
    ).fetchall()
    X, y, race_idx, names, car = [], [], [], [], []
    for ri, (sk, year, loc) in enumerate(races):
        names.append((year, loc))
        rows = con.execute(
            """SELECT l.driver_number, l.lap_number, l.position, coalesce(l.pit_in, false),
                      l.stint, l.compound, l.tyre_age, coalesce(l.neutralised, ''),
                      coalesce(l.gap_ahead, 99), coalesce(l.raining, false), d.team_name
               FROM laps l LEFT JOIN raw_drivers d
                 ON d.session_key = l.session_key AND d.driver_number = l.driver_number
               WHERE l.session_key = ? ORDER BY l.lap_number, l.position""",
            [sk],
        ).fetchall()
        if not rows:
            continue
        total = max(r[1] for r in rows)
        by_lap: dict[int, dict[int, tuple]] = {}
        for r in rows:
            by_lap.setdefault(r[1], {})[r[0]] = r
        stopped: set[int] = set()
        for n in sorted(by_lap):
            cur, prev = by_lap[n], by_lap.get(n - 1)
            if prev:
                for d, r in prev.items():
                    if r[3] and n - 1 > 1:
                        stopped.add(d)
            if n < 2 or not prev:
                continue
            order = sorted((r for r in prev.values() if r[2] is not None), key=lambda r: r[2])
            pos_index = {r[0]: i for i, r in enumerate(order)}
            ages = [r[6] for r in prev.values() if r[6] is not None]
            field_age = statistics.median(ages) if ages else 0
            teams: dict[str, list[int]] = {}
            for d, r in cur.items():
                if r[10]:
                    teams.setdefault(r[10], []).append(d)
            for d, r in cur.items():
                p = prev.get(d)
                if p is None or r[6] is None or r[2] is None:
                    continue
                i = pos_index.get(d)
                ahead = order[i - 1] if i else None
                behind = order[i + 1] if i is not None and i + 1 < len(order) else None
                mate = next((m for m in teams.get(r[10], []) if m != d), None)
                neut = r[7]
                feats = {
                    "age": r[6],
                    "age_sq": r[6] ** 2 / 100,
                    "first_stint": float(r[4] == 1),
                    "race_fraction": n / total,
                    "laps_left_lt5": float(total - n < 5),
                    "sc": float(neut == "SC"),
                    "vsc": float(neut == "VSC"),
                    "red": float(neut == "RED"),
                    "ahead_pitted_last_lap": float(bool(ahead and ahead[3])),
                    "behind_pitted_last_lap": float(bool(behind and behind[3])),
                    "teammate_pitted_last_lap": float(bool(mate and prev.get(mate, (0,) * 4)[3])),
                    "teammate_pitting_now": float(bool(mate and cur.get(mate, (0,) * 4)[3])),
                    "share_stopped": len(stopped) / max(len(cur), 1),
                    "age_vs_field": r[6] - field_age,
                    "gap_ahead": min(float(p[8]), 10.0),
                    "gap_behind": min(float(behind[8]), 10.0) if behind else 10.0,
                    "position": float(r[2]),
                    "raining": float(bool(r[9])),
                    **{f"compound_{c.lower()}": float(r[5] == c) for c in COMPOUNDS},
                    "regs_2026": float(year >= 2026),
                }

                X.append([feats[f] for f in FEATURES])
                y.append(float(bool(r[3])))
                race_idx.append(ri)
                car.append(d)
    con.close()
    return Rows(np.array(X), np.array(y), np.array(race_idx), names, np.array(car))


def columns(names: tuple[str, ...]) -> list[int]:
    return [FEATURES.index(f) for f in names]


def evaluate(rows: Rows, feature_set: tuple[str, ...]) -> np.ndarray:
    """Leave-one-race-out predicted P(pit this lap) for every row."""
    cols = columns(feature_set)
    X = rows.X[:, cols]
    p = np.empty(len(rows.y))
    for ri in np.unique(rows.race):
        test = rows.race == ri
        model = Logistic(l2=1.0).fit(X[~test], rows.y[~test])
        p[test] = model.predict(X[test])
    return p


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))[:, None]


def recalibrated(rows: Rows, p: np.ndarray) -> np.ndarray:
    """Platt scaling, leave-one-race-out: the raw model is overconfident above ~30%."""
    out = np.empty(len(p))
    z = _logit(p)
    for ri in np.unique(rows.race):
        test = rows.race == ri
        out[test] = Logistic(l2=0.0).fit(z[~test], rows.y[~test]).predict(z[test])
    return out


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@functools.cache
def model(feature_set: tuple[str, ...] = LIVE_FEATURES) -> Logistic:
    rows = load()
    return Logistic(l2=1.0).fit(rows.X[:, columns(feature_set)], rows.y)


def main() -> None:
    rows = load()
    print(f"{len(rows.y)} car-laps, {int(rows.y.sum())} stops, {len(rows.races)} races")
    base_rate = np.full(len(rows.y), rows.y.mean())
    results = {
        "constant": base_rate,
        "tyre age only": evaluate(rows, BASELINE),
        "live features": evaluate(rows, LIVE_FEATURES),
    }
    results["live, recalibrated"] = recalibrated(rows, results["live features"])
    years = np.array([rows.races[i][0] for i in rows.race])
    for name, p in results.items():
        per_year = "  ".join(
            f"{y}: {log_loss(p[years == y], rows.y[years == y]):.4f}" for y in sorted(set(years))
        )
        print(f"{name:14} log-loss {log_loss(p, rows.y):.4f}   {per_year}")
    p = results["live, recalibrated"]
    print("\nreliability (live, recalibrated): bin, car-laps, predicted, observed")
    for lo, hi in ((0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)):
        m = (p >= lo) & (p < hi)
        if m.any():
            print(
                f"  {lo:.2f}-{min(hi, 1):.2f} {int(m.sum()):6} {p[m].mean():.3f} {rows.y[m].mean():.3f}"
            )
    # Within the next 3 laps (what a strategist plans around): 1 - prod(1 - hazard).
    within, truth = [], []
    order = np.lexsort((np.arange(len(rows.y)),))
    del order
    by_car: dict[tuple, list[int]] = {}
    for i in range(len(rows.y)):
        by_car.setdefault((int(rows.race[i]), int(rows.car[i])), []).append(i)
    for idx in by_car.values():
        for k in range(len(idx)):
            nxt = idx[k : k + 3]
            within.append(1 - np.prod(1 - p[nxt]))
            truth.append(float(rows.y[nxt].max()))
    within, truth = np.array(within), np.array(truth)
    base3 = np.full(len(truth), truth.mean())
    print(
        f"\npits within 3 laps: log-loss model {log_loss(within, truth):.4f} vs constant "
        f"{log_loss(base3, truth):.4f}"
    )
    for lo, hi in ((0, 0.05), (0.05, 0.15), (0.15, 0.35), (0.35, 0.6), (0.6, 1.01)):
        m = (within >= lo) & (within < hi)
        if m.any():
            print(
                f"  {lo:.2f}-{min(hi, 1):.2f} {int(m.sum()):6} {within[m].mean():.3f} {truth[m].mean():.3f}"
            )
    stops = rows.y == 1
    print(
        f"\nof {int(stops.sum())} stops: P >= 0.3 the lap they happened for "
        f"{(p[stops] >= 0.3).mean():.0%}; P >= 0.1 for {(p[stops] >= 0.1).mean():.0%}"
    )
    w = model().w
    print(
        "weights:", dict(zip(("intercept", *LIVE_FEATURES), np.round(w, 2).tolist(), strict=True))
    )


if __name__ == "__main__":
    main()
