import math
from datetime import timedelta

from src.livetiming.archive import Message
from src.livetiming.monitor import RaceMonitor
from src.livetiming.track import track_outline


def _msg(t: float, topic: str, data: dict) -> Message:
    return Message(timedelta(seconds=t), topic, data)


def _circle_race(radius: float = 1000.0) -> list[Message]:
    """Leader (#4) drives a circle once per 60s lap; #1 is in the pit lane at the origin."""
    messages = [
        _msg(0, "TimingData", {"Lines": {"4": {"Position": "1"}, "1": {"Position": "2"}}}),
    ]
    for lap in range(1, 6):
        messages.append(_msg((lap - 1) * 60, "LapCount", {"CurrentLap": lap}))
    for step in range(5 * 60):
        angle = 2 * math.pi * (step % 60) / 60
        entries = {
            "4": {
                "Status": "OnTrack",
                "X": round(radius * math.cos(angle)),
                "Y": round(radius * math.sin(angle)),
            },
            "1": {"Status": "OnTrack", "X": 0, "Y": 0},
        }
        messages.append(
            _msg(
                step + 0.5,
                "Position.z",
                {"Position": [{"Timestamp": f"t{step}", "Entries": entries}]},
            )
        )
    return sorted(messages, key=lambda m: m.offset)


def test_outline_traces_the_leaders_lap():
    outline = track_outline(_circle_race())
    assert outline is not None
    xs = [p[0] for p in outline["points"]]
    ys = [p[1] for p in outline["points"]]
    # A full lap of the leader's circle: every point on the radius, all quadrants covered.
    assert all(abs(math.hypot(x, y) - 1000) < 2 for x, y in outline["points"])
    assert min(xs) < -990 and max(xs) > 990 and min(ys) < -990 and max(ys) > 990
    assert outline["bounds"] == [min(xs), min(ys), max(xs), max(ys)]


def test_outline_needs_lap_boundaries():
    no_laps = [m for m in _circle_race() if m.topic != "LapCount"]
    assert track_outline(no_laps) is None


def test_monitor_positions_event_uses_newest_sample_and_tla():
    monitor = RaceMonitor()
    monitor.feed(_msg(0, "DriverList", {"4": {"Tla": "NOR"}}))
    assert monitor.positions_event() is None
    sample = lambda ts, x: {
        "Timestamp": ts,
        "Entries": {"4": {"Status": "OnTrack", "X": x, "Y": 5}},
    }
    assert monitor.feed(_msg(1, "Position.z", {"Position": [sample("a", 1), sample("b", 2)]})) == []
    event = monitor.positions_event()
    assert event == {
        "type": "positions",
        "utc": "b",
        "cars": [{"number": "4", "tla": "NOR", "x": 2, "y": 5, "status": "OnTrack"}],
    }
