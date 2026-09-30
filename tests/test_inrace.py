import math

from src.sim.inrace import CarState, InRaceParams, RaceState, simulate_from, tyre_outlook


def _state(status, leader_owes=True):
    cars = [
        CarState("63", "RUS", 1, 0.0, "MEDIUM", 30, 0, leader_owes, 0.0),
        CarState("12", "ANT", 2, 40.0, "SOFT", 0, 1, False, 0.0),  # already stopped, 40 s back
    ]
    return RaceState(
        lap=31,
        laps_remaining=20,
        status=status,
        cars=cars,
        deg={"SOFT": 0.02, "MEDIUM": 0.02, "HARD": 0.02},
        life={"SOFT": 30, "MEDIUM": 30},
        pit_loss_green=22.0,
        pit_loss_sc=13.5,
    )


def test_leader_owing_a_stop_takes_it_under_the_safety_car():
    """Baku 2026 lap 31: the leaders still owed a stop when the SC came out; a car that had
    already stopped must not inherit the win just because the model stops the leader later."""
    # Mechanics test: no pace uncertainty (the tuned default spreads outcomes on purpose).
    out = simulate_from(_state("SAFETY_CAR"), sims=2000, seed=1, calib=InRaceParams(pace_sigma=0))
    rus = next(r for r in out["table"] if r["tla"] == "RUS")
    assert rus["p_win"] > 0.8


def test_tyre_outlook_flags_owed_stop_and_long_stints():
    state = _state("GREEN")
    assert tyre_outlook(state.cars[0], state)[0] == "high"  # owes the second compound
    long_run = CarState("1", "VER", 3, 5.0, "MEDIUM", 25, 1, False, 0.0)
    risk, note = tyre_outlook(long_run, state)
    assert risk == "high" and "past the longest typical" in note
    fresh = CarState("4", "NOR", 4, 6.0, "HARD", 2, 1, False, 0.0)
    assert tyre_outlook(fresh, state)[0] == "low"


def test_backtest_scoring_and_reliability_table():
    from src.sim.inrace_backtest import (
        Checkpoint,
        CheckpointScore,
        calibration_table,
        score_checkpoint,
        summarise,
    )

    cars = [
        CarState(str(i), f"D{i}", i, 2.0 * (i - 1), "MEDIUM", 5, 1, False, 0.3 * i)
        for i in range(1, 7)
    ]
    state = RaceState(
        lap=20,
        laps_remaining=30,
        status="GREEN",
        cars=cars,
        deg={"MEDIUM": 0.03},
        life={"MEDIUM": 40},
        pit_loss_green=22.0,
        pit_loss_sc=13.5,
    )
    cp = Checkpoint("Test GP", "GREEN", state, {str(i): i for i in range(1, 7)})
    tight = score_checkpoint(cp, InRaceParams(pace_sigma=0.0), sims=300)
    loose = score_checkpoint(cp, InRaceParams(pace_sigma=0.5), sims=300)
    assert tight.spearman > 0.9 and tight.winner_prob > 0.5
    assert loose.winner_prob < tight.winner_prob  # pace uncertainty spreads the probability

    scores = [
        CheckpointScore(0.9, 0.0, 1.0, [0.9, 0.1], [1, 0]),
        CheckpointScore(0.2, 0.0, 1.0, [0.8, 0.2], [0, 1]),
    ]
    assert round(summarise(scores)["log_loss"], 3) == round((-math.log(0.9) - math.log(0.2)) / 2, 3)
    table = {row[0]: row for row in calibration_table(scores)}
    _, n, predicted, observed = table["0.80-0.95"]
    assert (n, round(predicted, 3), observed) == (2, 0.85, 0.5)  # two ~85% calls, one came true
