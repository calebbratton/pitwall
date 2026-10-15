from datetime import timedelta

import pytest

from src.livetiming.archive import Message
from src.livetiming.monitor import LapRecord, RaceMonitor
from src.sim.pit_review import (
    _stops,
    measured_pit_loss,
    neutralised_laps,
    review_race,
    what_if_race,
)


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


# --- a synthetic race through the live pipeline: VSC on lap 15, HAM stays out until lap 20 ---

CARS = {"44": ("HAM", 0.0), "16": ("LEC", 0.15), "63": ("RUS", 0.3), "12": ("ANT", 0.45)}
TOTAL = 30
VSC_LAP = 15
STOPS = {"44": 20, "16": VSC_LAP, "63": VSC_LAP, "12": VSC_LAP}  # in-laps


def _race() -> list[Message]:
    t = [0.0]

    def msg(topic: str, data: dict) -> Message:
        t[0] += 0.01
        return Message(timedelta(seconds=t[0]), topic, data)

    out = [
        msg("SessionInfo", {"Name": "Race", "Type": "Race", "Meeting": {"Name": "Test GP"}}),
        msg("TrackStatus", {"Status": "1"}),
        msg("DriverList", {n: {"Tla": tla} for n, (tla, _) in CARS.items()}),
        msg("LapCount", {"CurrentLap": 1, "TotalLaps": TOTAL}),
        msg(
            "TimingAppData",
            {
                "Lines": {
                    n: {"Stints": [{"Compound": "MEDIUM", "New": "true", "TotalLaps": 0}]}
                    for n in CARS
                }
            },
        ),
    ]
    elapsed = {n: i * 1.0 for i, n in enumerate(CARS)}
    stint_age = dict.fromkeys(CARS, 0)
    stint = dict.fromkeys(CARS, 0)
    for lap in range(1, TOTAL + 1):
        out.append(msg("LapCount", {"CurrentLap": lap}))
        if lap == VSC_LAP:
            out.append(msg("TrackStatus", {"Status": "6"}))
        for n, (_, pace) in CARS.items():
            if STOPS[n] == lap:
                out.append(msg("TimingData", {"Lines": {n: {"InPit": True}}}))
        times = {}
        for n, (_, pace) in CARS.items():
            time = 90 + pace + 0.08 * stint_age[n] + (20 if lap == VSC_LAP else 0)
            if STOPS[n] == lap:
                time += 6 if lap == VSC_LAP else 10
            if STOPS[n] == lap - 1:
                time += 6 if lap - 1 == VSC_LAP else 12
            times[n] = time
            elapsed[n] += time
        leader = min(elapsed.values())
        order = sorted(CARS, key=elapsed.__getitem__)
        for pos, n in enumerate(order, 1):
            stint_age[n] += 1
            line = {
                "NumberOfLaps": lap,
                "Position": str(pos),
                "GapToLeader": f"+{elapsed[n] - leader:.3f}",
                "LastLapTime": {"Value": f"1:{times[n] - 60:06.3f}"},
                "InPit": False,
                "PitOut": STOPS[n] == lap - 1,
                "NumberOfPitStops": 1 if lap > STOPS[n] else 0,
            }
            out.append(msg("TimingData", {"Lines": {n: line}}))
            out.append(
                msg(
                    "TimingAppData",
                    {"Lines": {n: {"Stints": {str(stint[n]): {"TotalLaps": stint_age[n]}}}}},
                )
            )
            if STOPS[n] == lap:  # out of the pits on fresh hards
                stint[n] += 1
                stint_age[n] = 0
                new = {"Compound": "HARD", "New": "true", "TotalLaps": 0}
                out.append(msg("TimingAppData", {"Lines": {n: {"Stints": {str(stint[n]): new}}}}))
        if lap == VSC_LAP:
            out.append(msg("TrackStatus", {"Status": "1"}))
    return out


def test_what_if_pitting_under_the_vsc_beats_a_later_green_stop():
    r = what_if_race(_race(), 2026, "HAM", VSC_LAP, sims=300)
    actual, hypo = r.branches
    assert actual.label == "actual: pit lap 20"
    assert hypo.label == "pit lap 15 under the VSC for HARD"
    assert r.status == "VSC" and (r.old_compound, r.new_compound) == ("MEDIUM", "HARD")
    assert hypo.expected < actual.expected
    assert hypo.p_better_than_actual > hypo.p_worse_than_actual
    assert r.verdict.startswith("Pitting on lap 15 would likely have been better")
    assert any("really finished P" in n for n in r.notes)


def test_what_if_refuses_a_lap_they_really_pitted_on_and_a_repeat_compound():
    with pytest.raises(ValueError, match="did pit at the end of lap 20"):
        what_if_race(_race(), 2026, "HAM", 20, sims=50)
    with pytest.raises(ValueError, match="second compound"):
        what_if_race(_race(), 2026, "HAM", VSC_LAP, compound="medium", sims=50)


def test_review_of_the_real_stop_still_works_on_the_shared_core():
    r = review_race(_race(), 2026, "HAM", 20, sims=200)
    assert r.branches[0].label == "pit lap 20 (actual)"
    assert r.status == "GREEN"


def test_neutralised_laps_are_the_cheap_stop_windows():
    monitor = RaceMonitor()
    for m in _race():
        monitor.feed(m)
    # The VSC ends after the field crossed the line: the next lap started under it too.
    assert neutralised_laps(monitor.laps["44"]) == [VSC_LAP, VSC_LAP + 1]
    r = what_if_race(_race(), 2026, "HAM", VSC_LAP + 1, sims=50)
    assert r.status == "GREEN" and any("so a normal stop" in n for n in r.notes)
