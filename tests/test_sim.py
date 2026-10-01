from dataclasses import replace

import numpy as np
import pytest

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


def test_calibration_flattens_and_keeps_totals():
    from src.sim.calibrate import Calibration, apply, scale

    p_win = np.array([0.9, 0.08, 0.02, 0.0, 0.0])
    flat = scale(p_win, "win", 2.0, sims=1000)
    assert flat.sum() == pytest.approx(1.0)
    assert flat[0] < 0.9 and flat[-1] > 0  # overconfidence flattened, zeros get a chance
    assert list(np.argsort(-flat, kind="stable")) == list(np.argsort(-p_win, kind="stable"))
    podium = scale(np.array([1.0, 0.95, 0.9, 0.1, 0.05, 0.0]), "podium", 2.0, sims=1000)
    assert podium.sum() == pytest.approx(3.0, abs=1e-3)
    assert np.allclose(scale(p_win, "win", 1.0, sims=10**9), p_win, atol=1e-6)
    table = [
        {"tla": t, "expected": 1.0, "p_win": w, "p_podium": 0.6, "p_points": 1.0}
        for t, w in zip("ABCDE", p_win, strict=True)
    ]
    assert apply(table, Calibration(), sims=10**9)[0]["p_win"] == pytest.approx(0.9, abs=1e-3)


def test_penalised_grid():
    from src.sim.inputs import penalised_grid

    quali = {n: n for n in range(1, 21)}  # car n qualified P n
    g = penalised_grid(quali, {1: 10})
    assert g[1] == 11 and g[2] == 1 and g[11] == 10 and g[12] == 12
    g = penalised_grid(quali, {19: 10, 20: 5})  # can't drop below the back
    assert sorted(g.values()) == list(range(1, 21)) and {g[19], g[20]} == {19, 20}
    g = penalised_grid(quali, {1: 10, 2: 9}, pit_lane={3})
    assert g[1] == 11 and g[2] == 12  # same target: better qualifier first, the other next
    assert g[3] == 20 and g[4] == 1
    assert sorted(g.values()) == list(range(1, 21))


def test_weather_mixture_uses_the_rain_share_and_opens_up_the_race():
    from src.sim.wet import WET_PARAMS, simulate_weather

    inputs = _weekend(n=10, gaps=0.3)
    dry = simulate_weather(inputs, 0.0, sims=400, dry=BASE)
    wet = simulate_weather(inputs, 1.0, sims=400, dry=BASE)
    mixed = simulate_weather(inputs, 0.25, sims=400, dry=BASE)
    assert dry.positions.shape == wet.positions.shape == mixed.positions.shape == (400, 10)
    # smaller pace gaps and more noise: the favourite wins less often in the wet
    assert WET_PARAMS.race_pace_scale < SimParams().race_pace_scale
    assert wet.probability(1)[0] < dry.probability(1)[0]
    assert wet.probability(1)[0] < mixed.probability(1)[0] < dry.probability(1)[0] + 0.05


def test_calibrated_chances_stay_ordered():
    from src.sim.calibrate import DEFAULT_CALIBRATION, apply

    table = [
        {"tla": "A", "expected": 1.0, "p_win": 1.0, "p_podium": 1.0, "p_points": 1.0},
        *[
            {"tla": f"C{i}", "expected": 2.0 + i, "p_win": 0.0, "p_podium": 0.1, "p_points": 0.5}
            for i in range(19)
        ],
    ]
    for row in apply(table, DEFAULT_CALIBRATION, sims=5000):
        assert row["p_win"] <= row["p_podium"] <= row["p_points"]


def test_qualifying_segments_and_teammate_gap():
    import duckdb

    from src.sim.inputs import _segment_bests, _teammate_gap

    con = duckdb.connect()
    con.execute(
        "CREATE TABLE laps (session_key INT, driver_number INT, lap_start TIMESTAMPTZ, lap_time DOUBLE)"
    )
    rows = [
        # Q1 (minutes 0-10): both cars
        (1, "2026-01-01 10:01:00+00", 91.0),
        (2, "2026-01-01 10:02:00+00", 90.8),
        # Q2 after an 8-minute break: only car 2 (car 1 knocked out); track 0.5 s faster
        (2, "2026-01-01 10:20:00+00", 90.3),
    ]
    con.executemany("INSERT INTO laps VALUES (7, ?, ?, ?)", rows)
    bests = _segment_bests(con, 7)
    assert bests == {1: {0: 91.0}, 2: {0: 90.8, 1: 90.3}}
    # compared in Q1, where both ran: 0.2 s, not the 0.7 s their overall bests suggest
    assert _teammate_gap(bests, 1, 2) == pytest.approx(0.2)
    assert _teammate_gap(bests, 1, 3) is None
