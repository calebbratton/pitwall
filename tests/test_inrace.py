from src.sim.inrace import CarState, RaceState, simulate_from, tyre_outlook


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
    out = simulate_from(_state("SAFETY_CAR"), sims=2000, seed=1)
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
