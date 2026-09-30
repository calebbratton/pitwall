"""Tyre degradation model for one session, fitted from clean laps.

    lap_time = stint_intercept + deg[compound] * tyre_age - fuel_gain * lap_number + noise

Within a stint, tyre age and lap number both rise by one per lap, so the data alone can't tell
tyre wear from fuel burn-off. We correct laps with a fuel-gain prior (a stated assumption, with
its own uncertainty), then fit degradation per stint and pool stints per compound. Uncertainty
comes from bootstrapping over stints (stints, not laps, are the independent units).

2026 regulations changed the cars and tyres, so this is fitted from the current weekend only
(practice long runs, then the race so far) — never from older seasons.
"""

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# Assumption: lap time gained per lap of fuel burnt (~1.4-1.5 kg/lap at ~0.03 s/kg).
FUEL_GAIN_S_PER_LAP = 0.04
FUEL_GAIN_UNCERTAINTY = 0.015
MIN_STINT_LAPS = 5
OUTLIER_FACTOR = 1.03  # laps > 3% slower than the stint's median: traffic, mistakes
BOOTSTRAP_SAMPLES = 500


@dataclass(frozen=True)
class CleanLap:
    driver: str
    stint: int
    compound: str
    lap_number: int
    tyre_age: int
    lap_time: float


@dataclass(frozen=True)
class CompoundFit:
    compound: str
    deg_s_per_lap: float  # fuel-corrected: time lost per lap of tyre age
    deg_low: float  # 80% interval, including fuel-prior uncertainty
    deg_high: float
    offset_s: float | None  # fresh-tyre pace vs the reference compound (negative = faster)
    n_stints: int
    n_laps: int


@dataclass(frozen=True)
class TyreModel:
    fuel_gain_s_per_lap: float | None  # None for the panel method (absorbed by lap effects)
    reference_compound: str | None
    compounds: dict[str, CompoundFit] = field(default_factory=dict)
    method: str = "stint"  # "panel" (whole field, lap effects) or "stint" (fuel prior)
    note: str = ""
    # Panel method: how much faster/slower the track was on each lap for every car, relative to
    # the first clean lap (fuel burn + rubber + weather combined). Empty for the stint method.
    lap_effects: dict[int, float] = field(default_factory=dict)

    def lap_delta(self, compound: str, tyre_age: int) -> float | None:
        """Pace relative to a fresh reference-compound tyre (fuel excluded)."""
        fit = self.compounds.get(compound)
        if fit is None:
            return None
        return (fit.offset_s or 0.0) + fit.deg_s_per_lap * tyre_age


@dataclass(frozen=True)
class _StintFit:
    driver: str
    compound: str
    slope: float  # fuel-corrected degradation
    intercept: float  # fuel-corrected time at tyre age 0
    n: int


def _fit_stints(laps: list[CleanLap], fuel_gain: float) -> list[_StintFit]:
    by_stint: dict[tuple[str, int], list[CleanLap]] = defaultdict(list)
    for lap in laps:
        by_stint[(lap.driver, lap.stint)].append(lap)
    fits = []
    for (driver, _), stint in by_stint.items():
        median = statistics.median(l.lap_time for l in stint)
        stint = [l for l in stint if l.lap_time <= median * OUTLIER_FACTOR]
        if len(stint) < MIN_STINT_LAPS or len({l.tyre_age for l in stint}) < 3:
            continue
        ages = np.array([l.tyre_age for l in stint], dtype=float)
        # Add back the fuel burnt so every lap is compared at the stint's starting fuel load.
        corrected = np.array([l.lap_time + fuel_gain * l.lap_number for l in stint])
        slope, intercept = np.polyfit(ages, corrected, 1)
        fits.append(
            _StintFit(driver, stint[0].compound, float(slope), float(intercept), len(stint))
        )
    return fits


def _weighted_median(values: list[float], weights: list[float]) -> float:
    order = np.argsort(values)
    v, w = np.array(values)[order], np.array(weights, dtype=float)[order]
    cumulative = np.cumsum(w)
    return float(v[np.searchsorted(cumulative, cumulative[-1] / 2)])


def _compound_offsets(fits: list[_StintFit], reference: str) -> dict[str, float]:
    """Median over drivers who ran both compounds of (intercept_c - intercept_reference)."""
    by_driver: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in fits:
        by_driver[f.driver][f.compound].append(f.intercept)
    diffs: dict[str, list[float]] = defaultdict(list)
    for compounds in by_driver.values():
        if reference not in compounds:
            continue
        ref = statistics.median(compounds[reference])
        for compound, intercepts in compounds.items():
            if compound != reference:
                diffs[compound].append(statistics.median(intercepts) - ref)
    return {c: statistics.median(d) for c, d in diffs.items()}


def fit_stint_model(
    laps: list[CleanLap],
    fuel_gain: float = FUEL_GAIN_S_PER_LAP,
    fuel_uncertainty: float = FUEL_GAIN_UNCERTAINTY,
    seed: int = 0,
) -> TyreModel:
    fits = _fit_stints(laps, fuel_gain)
    by_compound: dict[str, list[_StintFit]] = defaultdict(list)
    for f in fits:
        by_compound[f.compound].append(f)
    if not by_compound:
        return TyreModel(fuel_gain, None)

    reference = max(by_compound, key=lambda c: sum(f.n for f in by_compound[c]))
    offsets = _compound_offsets(fits, reference)
    rng = np.random.default_rng(seed)
    compounds = {}
    for compound, stints in by_compound.items():
        slopes, weights = [f.slope for f in stints], [f.n for f in stints]
        deg = _weighted_median(slopes, weights)
        # Bootstrap over stints, and over the fuel prior: a higher fuel gain means the raw trend
        # hid more wear, shifting degradation up one-for-one.
        samples = []
        for _ in range(BOOTSTRAP_SAMPLES):
            idx = rng.integers(0, len(stints), len(stints))
            fuel_shift = rng.normal(0, fuel_uncertainty)
            samples.append(
                _weighted_median([slopes[i] for i in idx], [weights[i] for i in idx]) + fuel_shift
            )
        low, high = np.percentile(samples, [10, 90])
        compounds[compound] = CompoundFit(
            compound=compound,
            deg_s_per_lap=round(deg, 4),
            deg_low=round(float(low), 4),
            deg_high=round(float(high), 4),
            offset_s=0.0 if compound == reference else offsets.get(compound),
            n_stints=len(stints),
            n_laps=sum(weights),
        )
    return TyreModel(fuel_gain, reference, compounds, method="stint")


# --- panel method -------------------------------------------------------------------------
# lap_time = driver + lap + compound_offset + deg[compound] * tyre_age. The lap effects absorb
# fuel burn AND track evolution (shared by every car on that lap), so no fuel prior is needed.
# Degradation is identified by cars on different tyre ages on the same lap (staggered stops).

MIN_AGE_SPREAD = 2.0  # mean within-lap std of tyre age needed to trust the panel fit
PANEL_BOOTSTRAP = 200


def _filtered(laps: list[CleanLap]) -> list[CleanLap]:
    """Stint-level outlier removal and minimum stint length (same rules as the stint method)."""
    by_stint: dict[tuple[str, int], list[CleanLap]] = defaultdict(list)
    for lap in laps:
        by_stint[(lap.driver, lap.stint)].append(lap)
    kept = []
    for stint in by_stint.values():
        median = statistics.median(l.lap_time for l in stint)
        stint = [l for l in stint if l.lap_time <= median * OUTLIER_FACTOR]
        if len(stint) >= MIN_STINT_LAPS:
            kept.extend(stint)
    return kept


def age_spread(laps: list[CleanLap]) -> float:
    """Mean over laps of the std of tyre age across cars: how staggered the strategies are."""
    by_lap: dict[int, list[int]] = defaultdict(list)
    for lap in laps:
        by_lap[lap.lap_number].append(lap.tyre_age)
    spreads = [float(np.std(ages)) for ages in by_lap.values() if len(ages) >= 3]
    return float(np.mean(spreads)) if spreads else 0.0


def _panel_coefficients(
    laps: list[CleanLap], drivers: list[str], compounds: list[str], reference: str
) -> tuple[dict[str, float], dict[str, float], dict[int, float]]:
    all_laps = sorted({l.lap_number for l in laps})
    lap_numbers = all_laps[1:]  # first lap is the baseline
    d_idx = {d: i for i, d in enumerate(drivers)}
    l_idx = {n: len(drivers) + i for i, n in enumerate(lap_numbers)}
    offset_compounds = [c for c in compounds if c != reference]
    o_idx = {c: len(drivers) + len(lap_numbers) + i for i, c in enumerate(offset_compounds)}
    s_base = len(drivers) + len(lap_numbers) + len(offset_compounds)
    s_idx = {c: s_base + i for i, c in enumerate(compounds)}

    X = np.zeros((len(laps), s_base + len(compounds)))
    y = np.empty(len(laps))
    for row, lap in enumerate(laps):
        X[row, d_idx[lap.driver]] = 1
        if lap.lap_number in l_idx:
            X[row, l_idx[lap.lap_number]] = 1
        if lap.compound in o_idx:
            X[row, o_idx[lap.compound]] = 1
        X[row, s_idx[lap.compound]] = lap.tyre_age
        y[row] = lap.lap_time
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    deg = {c: float(beta[s_idx[c]]) for c in compounds}
    offsets = {reference: 0.0} | {c: float(beta[o_idx[c]]) for c in offset_compounds}
    lap_effects = {all_laps[0]: 0.0} | {n: float(beta[l_idx[n]]) for n in lap_numbers}
    return deg, offsets, lap_effects


def fit_panel_model(laps: list[CleanLap], seed: int = 0) -> TyreModel | None:
    """Whole-field fit, or None when strategies weren't staggered enough to identify it."""
    laps = _filtered(laps)
    if not laps or age_spread(laps) < MIN_AGE_SPREAD:
        return None
    n_laps: dict[str, int] = defaultdict(int)
    n_stints: dict[str, set] = defaultdict(set)
    for lap in laps:
        n_laps[lap.compound] += 1
        n_stints[lap.compound].add((lap.driver, lap.stint))
    compounds = sorted(n_laps)
    reference = max(compounds, key=lambda c: n_laps[c])
    drivers = sorted({l.driver for l in laps})
    deg, offsets, lap_effects = _panel_coefficients(laps, drivers, compounds, reference)

    # Cluster bootstrap over drivers: resampled drivers become distinct "copies".
    by_driver: dict[str, list[CleanLap]] = defaultdict(list)
    for lap in laps:
        by_driver[lap.driver].append(lap)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for _ in range(PANEL_BOOTSTRAP):
        picks = rng.integers(0, len(drivers), len(drivers))
        sample = [
            CleanLap(f"{drivers[p]}#{k}", l.stint, l.compound, l.lap_number, l.tyre_age, l.lap_time)
            for k, p in enumerate(picks)
            for l in by_driver[drivers[p]]
        ]
        present = sorted({l.compound for l in sample})
        if reference not in present:
            continue
        b_deg, b_off, _ = _panel_coefficients(
            sample, sorted({l.driver for l in sample}), present, reference
        )
        for c in present:
            samples[c].append((b_deg[c], b_off[c]))

    fits = {}
    for c in compounds:
        draws = np.array(samples[c]) if samples[c] else np.array([[deg[c], offsets[c]]])
        low, high = np.percentile(draws[:, 0], [10, 90])
        fits[c] = CompoundFit(
            compound=c,
            deg_s_per_lap=round(deg[c], 4),
            deg_low=round(float(low), 4),
            deg_high=round(float(high), 4),
            offset_s=round(offsets[c], 3),
            n_stints=len(n_stints[c]),
            n_laps=n_laps[c],
        )
    return TyreModel(None, reference, fits, method="panel", lap_effects=lap_effects)


def fit_tyre_model(
    laps: list[CleanLap],
    method: str = "auto",
    fuel_gain: float = FUEL_GAIN_S_PER_LAP,
    fuel_uncertainty: float = FUEL_GAIN_UNCERTAINTY,
    seed: int = 0,
) -> TyreModel:
    """`auto`: the panel model when strategies were staggered enough, else the stint model with
    the fuel prior (e.g. when the whole field stopped together under a safety car)."""
    if method in ("auto", "panel"):
        panel = fit_panel_model(laps, seed=seed)
        if panel is not None:
            return panel
        if method == "panel":
            raise ValueError("strategies not staggered enough for the panel model")
    model = fit_stint_model(laps, fuel_gain, fuel_uncertainty, seed)
    note = (
        f"stint method: tyre ages on the same lap too similar (spread "
        f"{age_spread(_filtered(laps)):.1f} "
        f"laps < {MIN_AGE_SPREAD}); fuel gain assumed, track evolution not separated"
        if method == "auto"
        else ""
    )
    return TyreModel(
        model.fuel_gain_s_per_lap, model.reference_compound, model.compounds, "stint", note
    )
