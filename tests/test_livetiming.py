import json
from pathlib import Path

import pytest

from src.livetiming.snapshot import build_snapshot
from src.livetiming.state import PitLaneTime, TimingState, merge
from src.livetiming.strategy import pit_calls

MIAMI_SC = Path(__file__).parent / "fixtures/livetiming/miami_2024_sc_lap28.json"


# --- merge semantics ------------------------------------------------------------------------


def test_merge_nested_dicts():
    state = {"Lines": {"4": {"Position": "1", "InPit": False}}}
    merge(state, {"Lines": {"4": {"InPit": True}, "81": {"Position": "2"}}})
    assert state == {"Lines": {"4": {"Position": "1", "InPit": True}, "81": {"Position": "2"}}}


def test_merge_list_by_index_updates_and_appends():
    state = {"Stints": [{"Compound": "MEDIUM", "TotalLaps": 10}]}
    merge(state, {"Stints": {"0": {"TotalLaps": 27}, "1": {"Compound": "HARD", "TotalLaps": 0}}})
    assert state["Stints"] == [
        {"Compound": "MEDIUM", "TotalLaps": 27},
        {"Compound": "HARD", "TotalLaps": 0},
    ]


def test_merge_deleted_keys():
    state = {"PitTimes": {"23": {"Duration": "22.8"}, "18": {"Duration": "23.0"}}}
    merge(state, {"PitTimes": {"_deleted": ["23"]}})
    assert state == {"PitTimes": {"18": {"Duration": "23.0"}}}


def test_merge_does_not_alias_update_objects():
    update = {"Messages": [{"Message": "GREEN"}]}
    state: dict = {}
    merge(state, update)
    update["Messages"][0]["Message"] = "mutated"
    assert state["Messages"][0]["Message"] == "GREEN"


def test_pit_lane_times_survive_deletion():
    state = TimingState()
    row = {"RacingNumber": "23", "Duration": "22.8", "Lap": "10"}
    state.apply("PitLaneTimeCollection", {"PitTimes": {"23": row}})
    state.apply("PitLaneTimeCollection", {"PitTimes": {"_deleted": ["23"]}})
    state.apply("PitLaneTimeCollection", {"PitTimes": {"23": row}})  # re-sent: no duplicate
    assert state.pit_lane_times == [PitLaneTime("23", 22.8, 10)]
    assert state.topics["PitLaneTimeCollection"]["PitTimes"] == {"23": row}


# --- snapshot + pit calls on Miami 2024, safety car lap 28 ----------------------------------


@pytest.fixture(scope="module")
def miami():
    data = json.loads(MIAMI_SC.read_text())
    state = TimingState(
        topics=data["topics"],
        pit_lane_times=[PitLaneTime(**p) for p in data["pit_lane_times"]],
        clock=data["clock"],
    )
    return build_snapshot(state)


def test_snapshot_matches_the_race(miami):
    assert (miami.meeting, miami.track_status) == ("Miami Grand Prix", "SAFETY_CAR")
    assert (miami.current_lap, miami.total_laps, miami.laps_remaining) == (28, 57, 29)
    assert [d.tla for d in miami.drivers[:3]] == ["NOR", "VER", "LEC"]
    nor, ver = miami.driver("NOR"), miami.driver("VER")
    assert (nor.gap_to_leader_s, nor.compound, nor.tyre_age_laps, nor.pit_stops) == (
        0.0,
        "MEDIUM",
        27,
        0,
    )
    assert nor.needs_second_compound
    assert ver.gap_to_leader_s == 11.392 and not ver.needs_second_compound
    assert miami.median_pit_lane_s == pytest.approx(22.6)


def test_safety_car_pit_calls(miami):
    report = pit_calls(miami)
    calls = {c.tla: c for c in report.calls}
    assert report.pit_loss_now_s == pytest.approx(8.8)
    # The race-winning call: must still use a second compound, nobody within the SC loss.
    assert calls["NOR"].call == "PIT" and calls["NOR"].places_at_risk == 0
    # Fresh tyres / too much to lose: stay out.
    for tla in ("VER", "PIA", "SAI", "HAM"):
        assert calls[tla].call == "STAY OUT", tla
    assert calls["LEC"].call == "STAY OUT" and calls["LEC"].places_at_risk == 2
    # Still on their first compound: the SC stop is the cheapest one they'll get.
    for tla in ("TSU", "ZHO", "RIC"):
        assert calls[tla].call == "PIT" and "second dry compound" in calls[tla].reason
    assert any("0.5" in a for a in report.assumptions)


def test_no_pit_calls_worth_it_near_the_end(miami):
    from dataclasses import replace

    late = replace(miami, current_lap=55)
    calls = {c.tla: c.call for c in pit_calls(late).calls}
    assert calls["BOT"] == "STAY OUT"  # old tyres, but only 2 laps left
    assert calls["NOR"] == "PIT"  # the mandatory compound still applies
