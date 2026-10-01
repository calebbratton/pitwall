from datetime import timedelta

import numpy as np

from src.livetiming.archive import Message
from src.livetiming.director import Director
from src.livetiming.monitor import RaceMonitor
from src.sim.battles import FEATURES, pair_features


class Fixed:
    def __init__(self, p):
        self.p = p

    def predict(self, X):
        return np.full(len(X), self.p)


def msg(t, topic, data):
    return Message(timedelta(seconds=t), topic, data)


def test_pair_features_shape_and_signs():
    f = pair_features(
        0.8, 1.5, [90.0, 90.1, 90.0], [90.4, 90.5, 90.3], 5, 20, "SOFT", "HARD", 1.2, 4, 0.5, 2026
    )
    assert len(f) == len(FEATURES)
    assert f[FEATURES.index("closing_3")] > 0 and f[FEATURES.index("pace_delta")] < 0
    assert f[FEATURES.index("softer")] == 1.0 and f[FEATURES.index("age_delta")] == 15


def test_director_picks_close_pair_and_alerts_once(monkeypatch):
    monkeypatch.setattr("src.sim.battles.models", lambda: (Fixed(0.6), Fixed(0.9)))
    monkeypatch.setattr("src.livetiming.director._circuit_rel", lambda snap, fn: 1.0)
    monitor = RaceMonitor()
    lines = {
        "1": {"Position": "1", "GapToLeader": "LAP 10", "NumberOfLaps": 9},
        "2": {
            "Position": "2",
            "GapToLeader": "+0.8",
            "IntervalToPositionAhead": {"Value": "+0.8"},
            "NumberOfLaps": 9,
        },
        "3": {
            "Position": "3",
            "GapToLeader": "+20.0",
            "IntervalToPositionAhead": {"Value": "+19.2"},
            "NumberOfLaps": 9,
        },
    }
    for m in [
        msg(0, "SessionInfo", {"Name": "Race", "Type": "Race", "Meeting": {"Name": "Test GP"}}),
        msg(0, "TrackStatus", {"Status": "1"}),
        msg(1, "DriverList", {"1": {"Tla": "AAA"}, "2": {"Tla": "BBB"}, "3": {"Tla": "CCC"}}),
        msg(1, "LapCount", {"CurrentLap": 10, "TotalLaps": 50}),
        msg(1, "TimingData", {"Lines": lines}),
    ]:
        monitor.feed(m)
    director = Director()
    events = director.on_lap(monitor)
    pick = next(e for e in events if e["type"] == "director")
    assert pick["watch"]["chaser"] == "BBB" and pick["watch"]["ahead"] == "AAA"
    assert len(pick["battles"]) == 1  # CCC is 19 s back: not a battle
    assert [e["kind"] for e in events if e["type"] == "alert"] == ["battle"]
    assert not [e for e in director.on_lap(monitor) if e["type"] == "alert"]  # same battle: once


def test_alerted_battle_resolves_when_the_pass_happens(monkeypatch):
    monkeypatch.setattr("src.sim.battles.models", lambda: (Fixed(0.9), Fixed(0.9)))
    monkeypatch.setattr("src.livetiming.director._circuit_rel", lambda snap, fn: 1.0)
    monitor = RaceMonitor()
    lines = {
        "1": {"Position": "1", "GapToLeader": "LAP 10", "NumberOfLaps": 9},
        "2": {
            "Position": "2",
            "GapToLeader": "+0.8",
            "IntervalToPositionAhead": {"Value": "+0.8"},
            "NumberOfLaps": 9,
        },
    }
    for m in [
        msg(0, "SessionInfo", {"Name": "Race", "Type": "Race", "Meeting": {"Name": "Test GP"}}),
        msg(0, "TrackStatus", {"Status": "1"}),
        msg(1, "DriverList", {"1": {"Tla": "AAA"}, "2": {"Tla": "BBB"}}),
        msg(1, "LapCount", {"CurrentLap": 10, "TotalLaps": 50}),
        msg(1, "TimingData", {"Lines": lines}),
    ]:
        monitor.feed(m)
    director = Director()
    assert any(e.get("kind") == "battle" for e in director.on_lap(monitor))
    # BBB gets past on the next lap
    monitor.feed(msg(90, "TimingData", {"Lines": {"2": {"Position": "1"}, "1": {"Position": "2"}}}))
    monitor.feed(msg(91, "LapCount", {"CurrentLap": 11}))
    resolved = [e for e in director.on_lap(monitor) if e.get("status") == "resolved"]
    assert (
        resolved
        and resolved[0]["outcome"] == "worked"
        and "BBB passed AAA" in resolved[0]["detail"]
    )
