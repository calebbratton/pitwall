from dataclasses import replace

import numpy as np

from src.sim.backtest import score
from src.sim.inputs import DriverInput, WeekendInputs
from src.sim.race import SimParams, race_pace, simulate

# Mechanics tests use fixed settings, independent of the tuned defaults (which change on retune).
BASE = SimParams(
    quali_weight=1.0, form_weight=0.0, pass_threshold=0.5, start_noise=0.6, lap_noise=0.35
)


def _weekend(n=10, gaps=0.2, dnf=0.0, sc=0.0, laps=50):
    drivers = [
        DriverInput(
            i + 1, f"D{i + 1:02d}", "T", grid=i + 1, quali_delta_s=i * gaps, long_run_delta_s=None
        )
        for i in range(n)
    ]
    return WeekendInputs("Test", 2026, None, laps, drivers, {"MEDIUM": 0.05}, 22.0, 13.0, sc, dnf)


def test_faster_cars_from_the_front_usually_win_and_output_is_consistent():
    pred = simulate(_weekend(), BASE, sims=500, seed=1)
    assert pred.positions.shape == (500, 10)
    # Every simulated race is a permutation of 1..n.
    assert (np.sort(pred.positions, axis=1) == np.arange(1, 11)).all()
    p_win = pred.probability(1)
    assert p_win.sum() == 1.0
    assert p_win[0] > 0.5 and p_win[0] == p_win.max()
    exp = pred.expected_position()
    assert np.all(np.diff(exp) > -0.5)  # roughly ordered by pace/grid


def test_seeded_runs_are_reproducible():
    a = simulate(_weekend(), BASE, sims=200, seed=7).positions
    b = simulate(_weekend(), BASE, sims=200, seed=7).positions
    assert (a == b).all()


def test_harder_passing_keeps_a_fast_car_stuck_behind():
    w = _weekend(gaps=0.0)
    w.drivers[-1].quali_delta_s = -1.0  # fastest car starts last
    easy = simulate(w, replace(BASE, pass_threshold=0.2), sims=400, seed=2)
    hard = simulate(w, replace(BASE, pass_threshold=2.0), sims=400, seed=2)
    assert easy.expected_position()[-1] < hard.expected_position()[-1] - 2


def test_retirements_cap_points_probability():
    pred = simulate(_weekend(n=20, dnf=0.3), replace(BASE, include_dnfs=True), sims=1000, seed=3)
    # Retired cars are classified last, so the favourite's P(points) ~ P(finishing) ~ 0.7.
    assert 0.6 < pred.probability(10)[0] < 0.8


def test_race_pace_blends_and_centres():
    w = _weekend()
    w.drivers[3].quali_delta_s = None  # no quali lap: placed by grid slot
    pace = race_pace(w, BASE)
    assert abs(np.median(pace)) < 1e-9
    assert pace[0] < pace[5]


def test_score_metrics():
    s = score(np.array([1.0, 2, 3, 4]), np.array([1.0, 2, 3, 4]), np.array([0.7, 0.2, 0.1, 0.0]))
    assert s.spearman == 1.0 and s.mae == 0 and s.winner_hit and s.podium_overlap == 3
    assert round(s.winner_logloss, 3) == 0.357


def test_dnfs_ignored_by_default():
    pred = simulate(_weekend(n=20, dnf=0.9), BASE, sims=300, seed=4)
    assert pred.probability(1)[0] > 0.5  # a 90% retirement rate changes nothing


def test_vsc_timing_can_cost_the_leader_the_race():
    """Madrid 2026: the leader passes the pit entry as the VSC comes out, the chasers get a
    cheap stop, he doesn't. Equal-pace cars, leader 6 s clear."""

    def two_cars(vsc):
        w = _weekend(n=2, gaps=0.0, laps=50)
        w.vsc_per_race, w.pit_loss_vsc, w.pit_loss_green = vsc, 14.0, 22.0
        return w

    params = replace(
        BASE, grid_spacing=6.0, start_noise=0.1, lap_noise=0.05, include_neutralisations=True
    )
    calm = simulate(two_cars(0.0), params, sims=2000, seed=5)
    vscs = simulate(two_cars(3.0), params, sims=2000, seed=5)
    assert calm.probability(1)[1] < 0.1  # without neutralisations the chaser rarely wins
    # A VSC only swings it when it lands in the pit window before either car stopped, and the
    # leader can't reach the pit entry in time — rare, but it multiplies the chaser's chances.
    assert vscs.probability(1)[1] > 2.5 * calm.probability(1)[1]


def test_score_ignores_non_finishers():
    actual = np.array([1.0, np.nan, 2, 3])  # car 2 retired
    s = score(np.array([1.0, 2, 3, 4]), actual, np.array([0.6, 0.3, 0.1, 0.0]))
    assert s.spearman == 1.0 and s.mae == 0 and s.winner_hit


def test_predictions_leave_out_neutralisations_by_default():
    w = _weekend(n=2, gaps=0.0)
    w.vsc_per_race, w.sc_per_race = 5.0, 5.0
    params = replace(BASE, grid_spacing=6.0, start_noise=0.1, lap_noise=0.05)
    assert simulate(w, params, sims=500, seed=6).probability(1)[1] < 0.05
