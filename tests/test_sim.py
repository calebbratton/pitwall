from dataclasses import replace

import numpy as np

from src.sim.backtest import score
from src.sim.inputs import DriverInput, WeekendInputs
from src.sim.race import SimParams, race_pace, simulate


def _weekend(n=10, gaps=0.2, dnf=0.0, sc=0.0, laps=50):
    drivers = [
        DriverInput(
            i + 1, f"D{i + 1:02d}", "T", grid=i + 1, quali_delta_s=i * gaps, long_run_delta_s=None
        )
        for i in range(n)
    ]
    return WeekendInputs("Test", 2026, None, laps, drivers, {"MEDIUM": 0.05}, 22.0, 13.0, sc, dnf)


def test_faster_cars_from_the_front_usually_win_and_output_is_consistent():
    pred = simulate(_weekend(), sims=500, seed=1)
    assert pred.positions.shape == (500, 10)
    # Every simulated race is a permutation of 1..n.
    assert (np.sort(pred.positions, axis=1) == np.arange(1, 11)).all()
    p_win = pred.probability(1)
    assert p_win.sum() == 1.0
    assert p_win[0] > 0.5 and p_win[0] == p_win.max()
    exp = pred.expected_position()
    assert np.all(np.diff(exp) > -0.5)  # roughly ordered by pace/grid


def test_seeded_runs_are_reproducible():
    a = simulate(_weekend(), sims=200, seed=7).positions
    b = simulate(_weekend(), sims=200, seed=7).positions
    assert (a == b).all()


def test_harder_passing_keeps_a_fast_car_stuck_behind():
    w = _weekend(gaps=0.0)
    w.drivers[-1].quali_delta_s = -1.0  # fastest car starts last
    easy = simulate(w, replace(SimParams(), pass_threshold=0.2), sims=400, seed=2)
    hard = simulate(w, replace(SimParams(), pass_threshold=2.0), sims=400, seed=2)
    assert easy.expected_position()[-1] < hard.expected_position()[-1] - 2


def test_retirements_cap_points_probability():
    pred = simulate(_weekend(n=20, dnf=0.3), sims=1000, seed=3)
    # Retired cars are classified last, so the favourite's P(points) ~ P(finishing) ~ 0.7.
    assert 0.6 < pred.probability(10)[0] < 0.8


def test_race_pace_blends_and_centres():
    w = _weekend()
    w.drivers[3].quali_delta_s = None  # no quali lap: placed by grid slot
    pace = race_pace(w, SimParams())
    assert abs(np.median(pace)) < 1e-9
    assert pace[0] < pace[5]


def test_score_metrics():
    s = score(np.array([1.0, 2, 3, 4]), np.array([1.0, 2, 3, 4]), np.array([0.7, 0.2, 0.1, 0.0]))
    assert s.spearman == 1.0 and s.mae == 0 and s.winner_hit and s.podium_overlap == 3
    assert round(s.winner_logloss, 3) == 0.357
