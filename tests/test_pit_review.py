from src.livetiming.monitor import LapRecord
from src.sim.pit_review import _stops, measured_pit_loss


def _lap(lap, t, stint=1, compound="MEDIUM", pit_in=False, pit_out=False, neutralised=False):
    return LapRecord(lap, t, compound, lap, stint, pit_in, pit_out, neutralised)


def _car(stop_at: int, extra: float, neutralised=False):
    laps = []
    for n in range(2, 20):
        if n == stop_at:
            laps.append(_lap(n, 90 + extra * 0.4, pit_in=True, neutralised=neutralised))
        elif n == stop_at + 1:
            laps.append(_lap(n, 90 + extra * 0.6, stint=2, compound="HARD", pit_out=True))
        else:
            laps.append(_lap(n, 90.0, stint=1 if n < stop_at else 2))
    return laps


def test_stops_are_in_lap_and_new_compound():
    assert _stops(_car(10, 25)) == [(10, "HARD")]


def test_pit_loss_from_green_stops_only():
    laps = {
        "1": _car(8, 24),
        "2": _car(10, 26),
        "3": _car(12, 25),
        "4": _car(9, 12, neutralised=True),
    }
    assert measured_pit_loss(laps) == (25.0, 3)
    assert measured_pit_loss({"1": _car(8, 24)}) is None  # too few stops to trust
