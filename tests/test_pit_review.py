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


def _race(
    stops: dict[str, int] = STOPS,
    cars: dict[str, tuple[str, float]] = CARS,
    extra: dict[tuple[str, int], float] | None = None,
) -> list[Message]:
    """`extra`: (car, lap) -> seconds added to that lap (e.g. a VSC that lasted longer where
    the car was on track)."""
    extra = extra or {}
    t = [0.0]

    def msg(topic: str, data: dict) -> Message:
        t[0] += 0.01
        return Message(timedelta(seconds=t[0]), topic, data)

    out = [
        msg("SessionInfo", {"Name": "Race", "Type": "Race", "Meeting": {"Name": "Test GP"}}),
        msg("TrackStatus", {"Status": "1"}),
        msg("DriverList", {n: {"Tla": tla} for n, (tla, _) in cars.items()}),
        msg("LapCount", {"CurrentLap": 1, "TotalLaps": TOTAL}),
        msg(
            "TimingAppData",
            {
                "Lines": {
                    n: {"Stints": [{"Compound": "MEDIUM", "New": "true", "TotalLaps": 0}]}
                    for n in cars
                }
            },
        ),
    ]
    elapsed = {n: i * 1.0 for i, n in enumerate(cars)}
    stint_age = dict.fromkeys(cars, 0)
    stint = dict.fromkeys(cars, 0)
    for lap in range(1, TOTAL + 1):
        out.append(msg("LapCount", {"CurrentLap": lap}))
        if lap == VSC_LAP:
            out.append(msg("TrackStatus", {"Status": "6"}))
        for n, (_, pace) in cars.items():
            if stops[n] == lap:
                out.append(msg("TimingData", {"Lines": {n: {"InPit": True}}}))
        times = {}
        for n, (_, pace) in cars.items():
            fuel = -0.04 * lap  # burn-off: the tyre fits correct for it
            time = 90 + pace + fuel + 0.08 * stint_age[n] + (20 if lap == VSC_LAP else 0)
            if stops[n] == lap:
                time += 6 if lap == VSC_LAP else 10
            if stops[n] == lap - 1:
                time += 6 if lap - 1 == VSC_LAP else 12
            time += extra.get((n, lap), 0.0)
            times[n] = time
            elapsed[n] += time
        leader = min(elapsed.values())
        order = sorted(cars, key=elapsed.__getitem__)
        for pos, n in enumerate(order, 1):
            stint_age[n] += 1
            line = {
                "NumberOfLaps": lap,
                "Position": str(pos),
                "GapToLeader": f"+{elapsed[n] - leader:.3f}",
                "LastLapTime": {"Value": f"1:{times[n] - 60:06.3f}"},
                "InPit": False,
                "PitOut": stops[n] == lap - 1,
                "NumberOfPitStops": 1 if lap > stops[n] else 0,
            }
            out.append(msg("TimingData", {"Lines": {n: line}}))
            out.append(
                msg(
                    "TimingAppData",
                    {"Lines": {n: {"Stints": {str(stint[n]): {"TotalLaps": stint_age[n]}}}}},
                )
            )
            if stops[n] == lap:  # out of the pits on fresh hards
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
    assert r.verdict.startswith("Pitting on lap 15 would have gained HAM about")
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


def test_time_ledger_when_the_driver_stayed_out_to_the_flag():
    # Singapore 2026 shape: HAM already stopped (lap 5) and ran to the flag; the others pitted
    # under the VSC. Fresh tyres were worth 0.08 s/lap per lap of age difference (10 laps).
    r = what_if_race(_race({**STOPS, "44": 5}), 2026, "HAM", VSC_LAP, sims=200)
    ledger = r.ledger
    assert r.branches[0].label == "actual: no more stops"
    assert ledger["compared_with"] == ["LEC", "RUS", "ANT"]
    assert ledger["fresh_tyre_gain_s"] == pytest.approx(0.8 * (TOTAL - VSC_LAP - 1), abs=1.5)
    assert ledger["net_s"] == pytest.approx(
        ledger["fresh_tyre_gain_s"] - ledger["stop_cost_s"], abs=0.11
    )
    assert "(measured from LEC, RUS, ANT)" in r.verdict


def test_time_ledger_measures_each_cars_own_wear():
    # The race-wide rate for a compound needn't fit one car's stint (Singapore 2026: 0.046 s/lap
    # on softs, while HAM's softs held flat to ~35 laps). Each car's own stint sets its wear, so
    # a wrong race-wide rate leaves the gain right: 0.08 s/lap per lap of age difference.
    from src.sim.pit_review import _load, time_ledger

    race = _load(_race({**STOPS, "44": 5}), 2026, "HAM")
    ledger = time_ledger(race, VSC_LAP, "HARD", {"MEDIUM": 0.3, "HARD": 0.3})
    assert ledger["fresh_tyre_gain_s"] == pytest.approx(0.8 * (TOTAL - VSC_LAP - 1), abs=1.5)


def test_time_ledger_counts_the_real_later_stop_as_saved():
    ledger = what_if_race(_race(), 2026, "HAM", VSC_LAP, sims=50).ledger
    # Older tyres before HAM's lap-20 stop, fresher after it: they cancel out here.
    assert ledger["fresh_tyre_gain_s"] == pytest.approx(0.0, abs=0.5)
    assert ledger["saved_real_stop_lap_20_s"] > ledger["stop_cost_s"]
    assert ledger["would_have_finished_ahead_of"]


def test_time_ledger_stop_cost_is_measured_against_the_drivers_own_laps():
    # Singapore 2026: the VSC was ending on lap 40, so cars elsewhere on track ran it at full
    # speed while HAM and the cars stopping near him were still slowed. Against the field's
    # median those two laps inflated the stop cost (19.5 s vs the 12.7-14.2 s LEC and ANT
    # really lost to HAM); and a green stop on the lap before isn't a VSC stop.
    cars = {**CARS, "1": ("VER", -0.3), "4": ("NOR", -0.2), "81": ("PIA", -0.1)}
    stops = {**STOPS, "44": 5, "1": 25, "4": VSC_LAP - 1, "81": 25}
    near = ("44", "16", "63", "12")
    extra = {(n, VSC_LAP + 1): 15.0 for n in near}
    r = what_if_race(_race(stops, cars, extra), 2026, "HAM", VSC_LAP, sims=100)
    assert r.ledger["compared_with"] == ["LEC", "RUS", "ANT"]
    # The synthetic stop costs 6 s on the in-lap and 6 s on the out-lap.
    assert r.ledger["stop_cost_s"] == pytest.approx(12.0, abs=0.7)


def test_archive_race_falls_back_to_the_race_date_when_the_name_misses(monkeypatch):
    # OpenF1 calls Singapore "Marina Bay", which isn't in the archive's folder names; the race
    # date finds it without the warehouse (which may not have ingested the race yet).
    from src.sim import pit_review

    def no_folder(year, place, session="Race"):
        raise LookupError("0 archived Race sessions match")

    def warehouse(*_):
        raise AssertionError("the race date should be enough")

    monkeypatch.setattr(pit_review, "find_session", no_folder)
    monkeypatch.setattr("src.warehouse.queries.connect", warehouse)
    monkeypatch.setattr(
        pit_review,
        "list_sessions",
        lambda year: [
            {
                "path": "2026/2026-10-04_Azerbaijan/2026-10-04_Race/",
                "meeting": "",
                "date": "2026-10-04",
            },
            {
                "path": "2026/2026-10-11_Singapore/2026-10-11_Race/",
                "meeting": "",
                "date": "2026-10-11",
            },
        ],
    )
    race = pit_review.archive_race(2026, "Marina Bay", day="2026-10-11")
    assert race.path == "2026/2026-10-11_Singapore/2026-10-11_Race/"
    # Night races start the next UTC day in some time zones (Las Vegas): one day either side.
    race = pit_review.archive_race(2026, "Marina Bay", day="2026-10-12")
    assert race.path == "2026/2026-10-11_Singapore/2026-10-11_Race/"


def test_review_verdict_never_calls_the_stop_best_when_another_option_scores_better():
    from src.sim.pit_review import Branch, review_verdict

    def b(label, expected, better=0.0, worse=0.0):
        return Branch(label, expected, 0.0, 1.0, better, worse)

    actual = b("pit lap 39 (actual)", 2.01)
    # Singapore 2026 LEC: no stop scored 1.95 - close, but better.
    v = review_verdict(
        39,
        [
            actual,
            b("stay out 1 more lap", 2.03, 0, 0.03),
            b("no stop (run to the flag)", 1.95, 0.05),
        ],
    )
    assert "right call" not in v
    assert "no stop (run to the flag)" in v and "P1.95" in v and "P2.01" in v
    v = review_verdict(39, [actual, b("stay out 1 more lap", 2.03, 0, 0.03)])
    assert v.startswith("Pitting on lap 39 was the right call")
    v = review_verdict(39, [actual, b("no stop (run to the flag)", 1.4, 0.6)])
    assert v.startswith("No stop (run to the flag) would likely have been better")


def test_baseline_check_shows_the_real_finish_and_the_head_to_heads_the_sim_gets_wrong():
    import numpy as np

    from src.sim.pit_review import baseline_check

    # Singapore 2026: what HAM really did simulates to ~P4.9, ahead of PIA in every run; PIA
    # really finished 3.5 s ahead of him (HAM P6).
    cars = ["1", "81", "44"]  # NOR, PIA, HAM
    tla = {"1": "NOR", "81": "PIA", "44": "HAM"}
    sim = np.array([[4, 6, 5]] * 9 + [[5, 6, 4]])
    note = baseline_check("44", cars, sim, {"1": 4, "81": 5, "44": 6}, tla)
    assert note.startswith("what HAM really did simulates to P4.9; HAM really finished P6")
    assert "ahead of PIA in 100% of runs" in note and "NOR" not in note
    assert "compare the options with each other" in note
    # Within a place: just the two numbers.
    assert baseline_check("44", cars, sim, {"1": 4, "81": 6, "44": 5}, tla) == (
        "what HAM really did simulates to P4.9; HAM really finished P5"
    )
    assert baseline_check("44", cars, sim, {"1": 4, "81": 5}, tla) is None  # didn't finish
