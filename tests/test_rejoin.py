import json
from pathlib import Path

import pytest

from src.livetiming.circuits import PitLoss
from src.livetiming.rejoin import rejoin, rejoin_table
from src.livetiming.snapshot import build_snapshot
from src.livetiming.state import PitLaneTime, TimingState

MIAMI_SC = Path(__file__).parent / "fixtures/livetiming/miami_2024_sc_lap28.json"
LOSS = PitLoss(green=20.0, safety_car=12.0, vsc=14.0, source="test")


@pytest.fixture(scope="module")
def miami():
    data = json.loads(MIAMI_SC.read_text())
    state = TimingState(
        topics=data["topics"],
        pit_lane_times=[PitLaneTime(**p) for p in data["pit_lane_times"]],
        clock=data["clock"],
    )
    return build_snapshot(state)


def test_rejoin_places_car_by_gap_plus_loss(miami):
    running = [d for d in miami.drivers if not d.retired and not d.laps_down and d.gap_to_leader_s]
    car = running[2]
    rows = {r.kind: r for r in rejoin(miami, car.tla, LOSS)}
    assert set(rows) == {"green", "sc", "vsc"}
    t = car.gap_to_leader_s + 20.0
    expected = 1 + sum(
        1
        for d in miami.drivers
        if d.tla != car.tla
        and not d.retired
        and not d.laps_down
        and (d.position == 1 or (d.gap_to_leader_s is not None and d.gap_to_leader_s <= t))
    )
    assert rows["green"].position == expected
    assert rows["sc"].position <= rows["vsc"].position <= rows["green"].position
    if rows["green"].ahead:
        assert rows["green"].gap_ahead_s >= 0


def test_leader_rejoin_and_table(miami):
    leader = miami.drivers[0]
    rows = rejoin(miami, leader.tla, LOSS)
    assert rows and all(r.position >= 1 for r in rows)
    table = rejoin_table(miami, LOSS)
    assert leader.tla in table and {r["kind"] for r in table[leader.tla]} == {"green", "sc", "vsc"}
    assert all(not d.retired for d in miami.drivers if d.tla in table)
