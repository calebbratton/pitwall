"""Which pre-race signals predict race results? A study over the ground-effect seasons.

2026 alone (15 races) can't separate real signals from noise, so this asks 2023–2025 first:
for each candidate predictor, does adding it to "qualifying pace + grid" improve predictions of
races it hasn't seen — in EACH season separately? A predictor only earns a place in the 2026
model if it does, and then also helps on 2026 itself.

Rows: one per classified driver per race. Targets (both luck-free, see luck.py):
  finish   luck-adjusted finishing rank (DNFs removed)
  pace     race pace: median clean lap (tyre-age corrected) vs the fastest driver, %
Features (known before the race; season-history ones use earlier races of the same season):
  quali_pct            best qualifying lap vs pole, %
  grid                 starting slot (penalties included)
  long_run_pct         practice/sprint long-run pace vs the field median, % of a lap
  team_form_pct        team's race-vs-qualifying pace bias in earlier races, %
  team_race_pace_pct   team's race pace gap in earlier races, % (shrunk toward 0)
  driver_sunday_gain   driver's mean places gained grid -> adjusted finish in earlier races (shrunk)
Protocol: forward chaining (each race predicted by a ridge regression fitted on every earlier
race, any season), features centred within each race, scored by rank correlation per race.

Usage: python -m src.analysis.predictors [--years 2023 2024 2025 2026]
"""

import argparse
import statistics
from dataclasses import dataclass

import numpy as np

from src.sim.backtest import actual_order
from src.sim.inputs import build_inputs, weekend_sessions
from src.warehouse.queries import connect

FEATURES = (
    "quali_pct",
    "grid",
    "long_run_pct",
    "team_form_pct",
    "team_race_pace_pct",
    "driver_sunday_gain",
)
BASE = ("quali_pct", "grid")
MIN_TRAIN_RACES = 8
RIDGE = 5.0
TEAM_SHRINK = 2  # pseudo-races of "average team" in the running means
DRIVER_SHRINK = 3


@dataclass
class RaceRows:
    year: int
    location: str
    drivers: list[int]
    X: dict[str, np.ndarray]  # feature -> per driver (NaN = unknown)
    finish: np.ndarray  # luck-adjusted rank among finishers (NaN = DNF)
    pace: np.ndarray  # race pace gap %, NaN if too few clean laps


def _race_pace(con, race_sk: int) -> dict[int, float]:
    rows = dict(
        con.execute(
            """SELECT driver_number, median(lap_time - 0.05 * tyre_age) FROM clean_laps
               WHERE session_key = ? GROUP BY 1 HAVING count(*) >= 10""",
            [race_sk],
        ).fetchall()
    )
    if not rows:
        return {}
    best = min(rows.values())
    return {d: 100 * (t - best) / best for d, t in rows.items()}


def collect(con, years: list[int]) -> list[RaceRows]:
    out: list[RaceRows] = []
    for year in years:
        meetings = con.execute(
            """SELECT location, min(date_start) FROM races WHERE year = ?
               GROUP BY meeting_key, location
               HAVING count(*) FILTER (session_name = 'Race') > 0
                  AND count(*) FILTER (session_name = 'Qualifying') > 0
               ORDER BY 2""",
            [year],
        ).fetchall()
        team_pace: dict[str, list[float]] = {}
        driver_gain: dict[int, list[float]] = {}
        for location, _ in meetings:
            try:
                inputs = build_inputs(con, year, location)
            except Exception as e:  # noqa: BLE001 — incomplete weekend data
                print(f"  skip {year} {location}: {e}")
                continue
            _, sessions = weekend_sessions(con, year, location)
            pole = con.execute(
                "SELECT min(lap_time) FROM laps WHERE session_key = ?", [sessions["Qualifying"]]
            ).fetchone()[0]
            if not pole or len(inputs.drivers) < 10:
                continue
            finish = actual_order(con, inputs)
            pace_now = _race_pace(con, inputs.race_session_key)
            ds = inputs.drivers

            def running(history, key, shrink):
                vals = history.get(key, [])
                return sum(vals) / (len(vals) + shrink)

            X = {
                "quali_pct": np.array(
                    [
                        np.nan if d.quali_delta_s is None else 100 * d.quali_delta_s / pole
                        for d in ds
                    ]
                ),
                "grid": np.array([float(d.grid) for d in ds]),
                "long_run_pct": np.array(
                    [
                        np.nan if d.long_run_delta_s is None else 100 * d.long_run_delta_s / pole
                        for d in ds
                    ]
                ),
                "team_form_pct": np.array([100 * d.form_s / pole for d in ds]),
                "team_race_pace_pct": np.array(
                    [running(team_pace, d.team, TEAM_SHRINK) for d in ds]
                ),
                "driver_sunday_gain": np.array(
                    [running(driver_gain, d.number, DRIVER_SHRINK) for d in ds]
                ),
            }
            pace = np.array([pace_now.get(d.number, np.nan) for d in ds])
            out.append(RaceRows(year, location, [d.number for d in ds], X, finish, pace))

            # Update season histories with this race's (luck-free) outcomes, for later races.
            by_team: dict[str, list[float]] = {}
            for d, p in zip(ds, pace, strict=True):
                if not np.isnan(p):
                    by_team.setdefault(d.team, []).append(p)
            for team, ps in by_team.items():
                team_pace.setdefault(team, []).append(statistics.mean(ps))
            for d, f in zip(ds, finish, strict=True):
                if not np.isnan(f):
                    driver_gain.setdefault(d.number, []).append(d.grid - f)
        print(f"  {year}: {sum(r.year == year for r in out)} races")
    return out


def _design(race: RaceRows, features: tuple[str, ...]) -> np.ndarray:
    """Features centred within the race; unknowns = the race mean (0 after centring)."""
    cols = []
    for f in features:
        x = race.X[f].astype(float)
        mean = np.nanmean(x) if np.isfinite(x).any() else 0.0
        x = np.where(np.isnan(x), mean, x) - mean
        cols.append(x)
    return np.column_stack(cols)


def _target(race: RaceRows, target: str) -> np.ndarray:
    y = race.finish if target == "finish" else race.pace
    return y - np.nanmean(y)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def forward_chain(races: list[RaceRows], features: tuple[str, ...], target: str) -> dict:
    """Rank correlation per predicted race, keyed by year; plus the last fit's weights."""
    scores: dict[int, list[float]] = {}
    beta = None
    for i, race in enumerate(races):
        if i < MIN_TRAIN_RACES:
            continue
        Xs, ys = [], []
        for past in races[:i]:
            y = _target(past, target)
            keep = ~np.isnan(y)
            Xs.append(_design(past, features)[keep])
            ys.append(y[keep])
        X, y = np.vstack(Xs), np.concatenate(ys)
        scale = X.std(axis=0)
        scale[scale == 0] = 1
        Z = X / scale
        beta = np.linalg.solve(Z.T @ Z + RIDGE * np.eye(Z.shape[1]), Z.T @ y) / scale
        yt = _target(race, target)
        keep = ~np.isnan(yt)
        if keep.sum() < 5:
            continue
        pred = _design(race, features)[keep] @ beta
        scores.setdefault(race.year, []).append(_spearman(pred, yt[keep]))
    return {
        "by_year": {y: float(np.mean(v)) for y, v in scores.items()},
        "races": {y: len(v) for y, v in scores.items()},
        "weights": dict(zip(features, np.round(beta, 4).tolist(), strict=True))
        if beta is not None
        else {},
    }


def study(races: list[RaceRows], target: str) -> dict:
    base = forward_chain(races, BASE, target)
    rows = {"base (quali + grid)": base}
    for f in FEATURES:
        if f not in BASE:
            rows[f"+ {f}"] = forward_chain(races, (*BASE, f), target)
    rows["all"] = forward_chain(races, FEATURES, target)
    rows["quali only"] = forward_chain(races, ("quali_pct",), target)
    rows["grid only"] = forward_chain(races, ("grid",), target)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    args = ap.parse_args()
    races = collect(connect(), args.years)
    years = sorted({r.year for r in races})
    for target in ("finish", "pace"):
        print(f"\ntarget: {target} (mean rank corr per race, forward-chained)")
        print(f"{'model':28}" + "".join(f"{y:>8}" for y in years))
        rows = study(races, target)
        base = rows["base (quali + grid)"]["by_year"]
        for name, r in rows.items():
            cells = ""
            for y in years:
                v = r["by_year"].get(y)
                if v is None:
                    cells += f"{'-':>8}"
                elif name.startswith("+"):
                    cells += f"{v - base[y]:>+8.3f}"
                else:
                    cells += f"{v:>8.3f}"
            print(f"{name:28}{cells}")
        print("  (+ rows: change vs base)  weights (all):", rows["all"]["weights"])


if __name__ == "__main__":
    main()
