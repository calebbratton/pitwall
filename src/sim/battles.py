"""Battles: which close pairs are about to fight - the Director's "watch this onboard".

One row per green lap per pair of cars running within MAX_GAP_S (chaser right behind the car
ahead), from every race 2023 onwards. Targets:
  pass_5   the chaser is ahead of that car within the next 5 laps, neither having pitted (a real
           on-track pass, not the pit cycle)
  close_3  the gap drops under 1.0 s at some lap end within the next 3 laps (a fight to watch)
Features are known at the end of the lap: the gap and how fast it's closing, the pace difference,
tyre age and compound differences, how hard the circuit is to pass at (from earlier seasons), how
far into the race, the 2026 regulations (Overtake mode within 1 s).

Scored leave-one-race-out against a gap-only model. Usage: python -m src.sim.battles
"""

import functools
import itertools
import statistics
from dataclasses import dataclass

import numpy as np

from src.sim.pit_hazard import Blend, log_loss
from src.sim.undercut_model import Logistic
from src.warehouse.queries import connect

MAX_GAP_S = 3.0
PASS_LAPS = 5
CLOSE_LAPS = 3
COMPOUND_RANK = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
FEATURES = (
    "gap",
    "closing_3",  # gap 3 laps ago minus now (+ = chaser closing)
    "pace_delta",  # chaser's median of last 3 clean laps minus the car ahead's (- = faster)
    "age_delta",  # car ahead's tyre age minus the chaser's (+ = chaser on fresher tyres)
    "softer",  # chaser on a softer compound
    "pass_rel",  # circuit's on-track pass rate vs the median (earlier seasons)
    "position",
    "race_fraction",
    "regs_2026",
)
BASELINE = ("gap",)


@dataclass
class Rows:
    X: np.ndarray
    y_pass: np.ndarray
    y_close: np.ndarray
    race: np.ndarray
    races: list[tuple[int, str]]


def _median(values):
    vals = [v for v in values if v]
    return statistics.median(vals) if len(vals) >= 2 else None


def pair_features(
    gap: float,
    gap_3_ago: float | None,
    chaser_laps: list[float],
    ahead_laps: list[float],
    chaser_age: int,
    ahead_age: int,
    chaser_compound: str | None,
    ahead_compound: str | None,
    pass_rel: float,
    position: int,
    race_fraction: float,
    year: int,
) -> list[float]:
    """Shared by the dataset and the live Director, so both use the same features."""
    c, a = _median(chaser_laps[-3:]), _median(ahead_laps[-3:])
    return [
        gap,
        (gap_3_ago - gap) if gap_3_ago is not None else 0.0,
        (c - a) if c is not None and a is not None else 0.0,
        float(ahead_age - chaser_age),
        float(
            COMPOUND_RANK.get(chaser_compound or "", 1) < COMPOUND_RANK.get(ahead_compound or "", 1)
        ),
        pass_rel,
        float(position),
        race_fraction,
        float(year >= 2026),
    ]


@functools.cache
def load(years: tuple[int, ...] = (2023, 2024, 2025, 2026)) -> Rows:
    from src.sim.inputs import circuit_pass_rel

    con = connect()
    races = con.execute(
        f"""SELECT session_key, year, location, circuit_key FROM races
            WHERE session_name = 'Race' AND year IN ({",".join("?" * len(years))})
            ORDER BY date_start""",
        list(years),
    ).fetchall()
    X, yp, yc, race_idx, names = [], [], [], [], []
    for ri, (sk, year, loc, ck) in enumerate(races):
        names.append((year, loc))
        rel = circuit_pass_rel(ck, year)
        rows = con.execute(
            """SELECT driver_number, lap_number, position, gap_ahead, lap_time, tyre_age, compound,
                      coalesce(pit_in, false) OR coalesce(pit_out, false),
                      coalesce(neutralised, '') <> ''
               FROM laps WHERE session_key = ? AND position IS NOT NULL ORDER BY lap_number""",
            [sk],
        ).fetchall()
        if not rows:
            continue
        total = max(r[1] for r in rows)
        by_lap: dict[int, dict[int, tuple]] = {}
        times: dict[int, dict[int, float]] = {}
        for r in rows:
            by_lap.setdefault(r[1], {})[r[0]] = r
            if r[4] and not r[7] and not r[8]:
                times.setdefault(r[0], {})[r[1]] = r[4]
        for n in sorted(by_lap):
            cur = by_lap[n]
            if n < 4 or any(r[8] for r in cur.values()):
                continue  # start chaos or a neutralised lap
            order = sorted(cur.values(), key=lambda r: r[2])
            for ahead, chaser in itertools.pairwise(order):
                gap = chaser[3]
                if gap is None or not (0 < gap <= MAX_GAP_S) or chaser[7] or ahead[7]:
                    continue
                d_c, d_a = chaser[0], ahead[0]
                prev = by_lap.get(n - 3, {}).get(d_c)
                gap_3 = (
                    prev[3]
                    if prev and by_lap[n - 3].get(d_a) and prev[2] == by_lap[n - 3][d_a][2] + 1
                    else None
                )
                feats = pair_features(
                    gap,
                    gap_3,
                    [times.get(d_c, {}).get(k) for k in range(n - 2, n + 1)],
                    [times.get(d_a, {}).get(k) for k in range(n - 2, n + 1)],
                    chaser[5] or 0,
                    ahead[5] or 0,
                    chaser[6],
                    ahead[6],
                    rel,
                    ahead[2],
                    n / total,
                    year,
                )
                passed = close = False
                for k in range(1, PASS_LAPS + 1):
                    later = by_lap.get(n + k, {})
                    c, a = later.get(d_c), later.get(d_a)
                    if c is None or a is None or c[7] or a[7]:
                        break  # a stop ends the battle's window
                    if k <= CLOSE_LAPS and c[2] == a[2] + 1 and c[3] is not None and c[3] < 1.0:
                        close = True
                    if c[2] < a[2]:
                        passed = True
                        close = close or k <= CLOSE_LAPS
                        break
                X.append(feats)
                yp.append(float(passed))
                yc.append(float(close))
                race_idx.append(ri)
    con.close()
    return Rows(np.array(X), np.array(yp), np.array(yc), np.array(race_idx), names)


def evaluate(rows: Rows, target: np.ndarray, features: tuple[str, ...], blend: bool) -> np.ndarray:
    cols = [FEATURES.index(f) for f in features]
    X = rows.X[:, cols]
    p = np.empty(len(target))
    for ri in np.unique(rows.race):
        test = rows.race == ri
        model = (
            Blend(X[~test], target[~test])
            if blend
            else Logistic(l2=1.0).fit(X[~test], target[~test])
        )
        p[test] = model.predict(X[test])
    return p


@functools.cache
def models() -> tuple[Blend, Blend]:
    rows = load()
    return Blend(rows.X, rows.y_pass), Blend(rows.X, rows.y_close)


def main() -> None:
    rows = load()
    years = np.array([rows.races[i][0] for i in rows.race])
    print(
        f"{len(rows.y_pass)} pair-laps in {len(rows.races)} races; pass within {PASS_LAPS} laps "
        f"{rows.y_pass.mean():.1%}, within 1 s within {CLOSE_LAPS} laps {rows.y_close.mean():.1%}"
    )
    for name, y in (("pass_5", rows.y_pass), ("close_3", rows.y_close)):
        base = np.full(len(y), y.mean())
        gap_only = evaluate(rows, y, BASELINE, blend=False)
        full = evaluate(rows, y, FEATURES, blend=True)
        print(f"\n{name}:")
        for label, p in (("constant", base), ("gap only", gap_only), ("all (blend)", full)):
            per = "  ".join(
                f"{yr}: {log_loss(p[years == yr], y[years == yr]):.4f}" for yr in sorted(set(years))
            )
            print(f"  {label:12} log-loss {log_loss(p, y):.4f}   {per}")
        print("  reliability (all): bin, n, predicted, observed")
        for lo, hi in ((0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 1.01)):
            m = (full >= lo) & (full < hi)
            if m.any():
                print(
                    f"    {lo:.2f}-{min(hi, 1):.2f} {int(m.sum()):6} {full[m].mean():.3f} {y[m].mean():.3f}"
                )


if __name__ == "__main__":
    main()
