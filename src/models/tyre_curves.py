"""Tyre-age curves per Pirelli compound (C1..C5), pooled across a season's races.

    lap_time = race_lap effect + driver_in_race effect + curve[C-number][age band] + noise

Race-lap effects (every car on the same lap of the same race) absorb fuel burn, track evolution
and weather; driver-in-race effects absorb car/driver pace. What's left is how lap time changes
with tyre age for each compound — identified by cars on different tyre ages on the same lap.
Pooling races by C-number (not HARD/MEDIUM/SOFT, which are relative per weekend) is what makes
this comparable across circuits.

Bands the data can't identify (see `_solve`) are dropped, not guessed.

"Tyre life" = the first age band where the per-lap loss clearly accelerates (the drop-off),
if the data reaches it; otherwise the oldest well-supported band ("no drop-off seen up to N").

Usage: python -m src.models.tyre_curves [--year 2026]
"""

import argparse
import functools
import itertools
from dataclasses import dataclass

import numpy as np

from src.models.compounds import c_number
from src.warehouse.queries import connect

BAND = 4  # laps per age band
MAX_AGE = 44  # ages above are pooled into the last band
MIN_BAND_LAPS = 40  # below this a band isn't trusted
DROP_OFF_S_PER_LAP = 0.12  # marginal loss per lap of age that counts as "falling away"


@dataclass(frozen=True)
class CompoundCurve:
    compound: str  # "C3"
    bands: list[tuple[int, float, int]]  # (band start age, loss vs fresh in s, laps)
    drop_off_age: int | None  # first age where marginal loss >= DROP_OFF_S_PER_LAP
    supported_to: int  # oldest band start with >= MIN_BAND_LAPS laps

    @property
    def life_laps(self) -> int:
        """Usable life for 'can they make it?': the drop-off if seen, else how far the data
        reaches without one (a conservative lower bound)."""
        return self.drop_off_age if self.drop_off_age is not None else self.supported_to + BAND


def _band(age: int) -> int:
    return min(age, MAX_AGE) // BAND * BAND


def _solve(X: np.ndarray, y: np.ndarray, curve_columns: list[int]):
    """Least squares plus an identifiability check for the curve coefficients.

    The design is rank-deficient by construction (race-lap and driver-in-race dummies both sum
    to each race's total) and, worse, can be rank-deficient in the CURVE: in a first stint every
    car's tyre age equals lap − 1, so age and lap effects can't be told apart. A coefficient is
    identified only if it's orthogonal to the design's null space; unidentified bands are
    dropped rather than reported (lstsq would otherwise return an arbitrary number for them)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    # Null space from the eigen-decomposition of X'X (columns << rows, so this is cheap).
    gram = X.T @ X
    eigvals, eigvecs = np.linalg.eigh(gram)
    null = eigvecs[:, eigvals < eigvals.max() * 1e-10]
    identified = np.ones(X.shape[1], dtype=bool)
    if null.size:
        weight = np.linalg.norm(null, axis=1)
        for j in curve_columns:
            identified[j] = weight[j] < 1e-6
    return beta, identified


def fit_curves(con, year: int, exclude: tuple[int, ...] = ()) -> dict[str, CompoundCurve]:
    """`exclude`: race session keys to leave out (leave-one-race-out validation)."""
    rows = con.execute(
        """SELECT l.session_key, r.location, l.driver_number, l.lap_number, l.compound,
                  l.tyre_age, l.lap_time
           FROM clean_laps l JOIN races r USING (session_key)
           WHERE r.year = ? AND r.session_name = 'Race' AND l.tyre_age IS NOT NULL""",
        [year],
    ).fetchall()
    rows = [r for r in rows if r[0] not in exclude]
    data = []
    for sk, location, driver, lap, compound, age, time in rows:
        c = c_number(year, location, compound)
        if c is not None:
            data.append((sk, driver, lap, c, int(age), float(time)))
    if not data:
        return {}

    # Remove per-race outliers (traffic, mistakes): > 3% slower than that lap's field median.
    by_race_lap: dict[tuple, list[float]] = {}
    for sk, _, lap, _, _, t in data:
        by_race_lap.setdefault((sk, lap), []).append(t)
    medians = {k: float(np.median(v)) for k, v in by_race_lap.items()}
    data = [d for d in data if d[5] <= medians[(d[0], d[2])] * 1.03]

    race_laps = sorted({(d[0], d[2]) for d in data})
    drivers = sorted({(d[0], d[1]) for d in data})
    cells = sorted({(d[3], _band(d[4])) for d in data})
    # One reference band per compound (0-3 laps) is the baseline; each race's first lap and
    # each race's first driver absorb the constant.
    baseline = {c: min(b for cc, b in cells if cc == c) for c in {cell[0] for cell in cells}}
    curve_cells = [cell for cell in cells if cell[1] != baseline[cell[0]]]
    rl_idx = {k: i for i, k in enumerate(race_laps)}
    dr_idx = {k: len(race_laps) + i for i, k in enumerate(drivers)}
    cv_idx = {k: len(race_laps) + len(drivers) + i for i, k in enumerate(curve_cells)}

    X = np.zeros((len(data), len(race_laps) + len(drivers) + len(curve_cells)), dtype=np.float64)
    y = np.empty(len(data))
    for row, (sk, driver, lap, c, age, t) in enumerate(data):
        X[row, rl_idx[(sk, lap)]] = 1
        X[row, dr_idx[(sk, driver)]] = 1
        if (c, _band(age)) in cv_idx:
            X[row, cv_idx[(c, _band(age))]] = 1
        y[row] = t
    beta, identified = _solve(X, y, [cv_idx[cell] for cell in curve_cells])

    counts: dict[tuple[str, int], int] = {}
    for _, _, _, c, age, _ in data:
        counts[(c, _band(age))] = counts.get((c, _band(age)), 0) + 1
    curves = {}
    for c in sorted(baseline):
        bands = [(baseline[c], 0.0, counts.get((c, baseline[c]), 0))]
        bands += [
            (b, round(float(beta[cv_idx[(c, b)]]), 3), counts[(c, b)])
            for cc, b in curve_cells
            if cc == c and identified[cv_idx[(c, b)]]
        ]
        bands.sort()
        trusted = [b for b in bands if b[2] >= MIN_BAND_LAPS]
        drop = None
        for (a0, l0, _), (a1, l1, _) in itertools.pairwise(trusted):
            if (l1 - l0) / (a1 - a0) >= DROP_OFF_S_PER_LAP:
                drop = a1
                break
        curves[c] = CompoundCurve(c, bands, drop, trusted[-1][0] if trusted else 0)
    return curves


@functools.cache
def season_curves(year: int) -> dict[str, CompoundCurve]:
    """Cached season curves (about a second to fit); empty if the warehouse isn't available."""
    try:
        return fit_curves(connect(), year)
    except Exception:  # noqa: BLE001 — live tools must work without the warehouse
        return {}


def loss_by_age(curve: CompoundCurve, max_age: int = 70) -> list[float]:
    """Seconds lost vs a fresh set at each tyre age 0..max_age: linear between the trusted
    bands' midpoints, then the settled slope (late_slope) beyond the data."""
    pts = [(0.0, 0.0)] + [
        (a + (BAND - 1) / 2, loss) for a, loss, n in curve.bands if n >= MIN_BAND_LAPS and a > 0
    ]
    pts.sort()
    slope = late_slope(curve) or 0.03
    out = []
    for age in range(max_age + 1):
        if age >= pts[-1][0]:
            out.append(pts[-1][1] + slope * (age - pts[-1][0]))
            continue
        for (x0, y0), (x1, y1) in itertools.pairwise(pts):
            if x0 <= age <= x1:
                out.append(y0 + (y1 - y0) * (age - x0) / (x1 - x0))
                break
    return [round(v, 4) for v in out]


def late_slope(curve: CompoundCurve) -> float | None:
    """Degradation (s/lap) over the settled phase: from 8 laps to the oldest trusted band."""
    trusted = [b for b in curve.bands if b[2] >= MIN_BAND_LAPS and b[0] >= 8]
    if len(trusted) < 2:
        return None
    (a0, l0, _), (a1, l1, _) = trusted[0], trusted[-1]
    return max((l1 - l0) / (a1 - a0), 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()
    curves = fit_curves(connect(), args.year)
    for c, curve in curves.items():
        drop = (
            f"drop-off from ~{curve.drop_off_age} laps"
            if curve.drop_off_age is not None
            else f"no drop-off seen up to {curve.supported_to + BAND - 1} laps"
        )
        print(f"{c}: {drop} (life used: {curve.life_laps})")
        print("   " + "  ".join(f"{a:>2}+: {loss:+.2f}s ({n})" for a, loss, n in curve.bands))


if __name__ == "__main__":
    main()


DEG_SHRINK_RACES = 2  # pseudo-races at "average wear" in the per-circuit offset


@functools.cache
def _race_deg_rel() -> list[tuple[int, int | None, int, float]]:
    """(year, circuit_key, race session_key, mean of per-compound degradation minus that
    season's median for the compound) for every warehouse race: how hard the track is on tyres
    beyond the compound choice. Fuel-corrected stint fits, dry compounds."""
    from src.models.tyres import CleanLap, fit_tyre_model

    con = connect()
    try:
        races = con.execute(
            "SELECT session_key, year, circuit_key FROM races WHERE session_name = 'Race'"
        ).fetchall()
        fitted = []
        for sk, year, ck in races:
            rows = con.execute(
                """SELECT driver_number::VARCHAR, stint, compound, lap_number, tyre_age, lap_time
                   FROM clean_laps WHERE session_key = ? AND tyre_age IS NOT NULL
                     AND compound IN ('SOFT', 'MEDIUM', 'HARD')""",
                [sk],
            ).fetchall()
            if len(rows) < 200:
                continue
            fits = fit_tyre_model([CleanLap(*r) for r in rows], method="stint").compounds
            degs = {c: f.deg_s_per_lap for c, f in fits.items() if f.n_stints >= 4}
            if degs:
                fitted.append((year, ck, sk, degs))
    finally:
        con.close()
    by_season: dict[tuple[int, str], list[float]] = {}
    for year, _, _, degs in fitted:
        for c, d in degs.items():
            by_season.setdefault((year, c), []).append(d)
    median = {k: float(np.median(v)) for k, v in by_season.items()}
    return [
        (year, ck, sk, float(np.mean([d - median[(year, c)] for c, d in degs.items()])))
        for year, ck, sk, degs in fitted
    ]


def circuit_deg_offset(circuit_key: int | None, exclude: tuple[int, ...] = ()) -> float:
    """Extra seconds lost per lap of tyre age at this circuit vs the season norm, from its other
    races (2023 onwards), shrunk toward 0. 0 for circuits with no history."""
    if circuit_key is None:
        return 0.0
    mine = [rel for _, ck, sk, rel in _race_deg_rel() if ck == circuit_key and sk not in exclude]
    return sum(mine) / (len(mine) + DEG_SHRINK_RACES)
