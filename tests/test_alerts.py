from src.livetiming.alerts import Alert, AlertEngine, hit_rate, sc_call_alert
from src.livetiming.strategy import PitCall, PitCallReport
from src.sim.inrace import CarState, RaceState


def _report(status="SAFETY_CAR"):
    calls = (
        PitCall("NOR", 1, "PIT", "Tyres 25 laps old; free stop", 0, "MEDIUM", 25),
        PitCall("RUS", 2, "STAY OUT", "Track position", 2, "HARD", 10),
        PitCall("HUL", 14, "PIT", "", 0, "SOFT", 20),
    )
    return PitCallReport(status, 30, 20, 22.0, 13.5, calls, ())


def test_sc_call_alert_summarises_the_front_runners_once():
    alert = sc_call_alert(_report())
    assert alert.headline == "SC: pit: NOR; stay out: RUS"
    assert "HUL" not in alert.drivers  # outside the top 10
    engine = AlertEngine()
    assert len(engine.on_pit_calls(_report())) == 1
    assert engine.on_pit_calls(_report()) == []  # same SC, said once
    assert sc_call_alert(_report("VSC")).headline.startswith("VSC:")


def _car(number, tla, pos, gap, age, compound="MEDIUM", pace=0.0, owes=False):
    return CarState(number, tla, pos, gap, compound, age, 0 if owes else 1, owes, pace)


def test_undercut_pairs_and_simulation():
    state = RaceState(
        lap=25,
        laps_remaining=30,
        status="GREEN",
        cars=[
            _car("1", "AAA", 1, 0.0, 25, owes=True),
            _car("2", "BBB", 2, 1.2, 25, owes=True),
            _car("3", "CCC", 3, 30.0, 5, compound="HARD"),
        ],
        deg={"MEDIUM": 0.12, "HARD": 0.04},
        life={"MEDIUM": 28, "HARD": 40},
        pit_loss_green=21.0,
        pit_loss_sc=13.0,
    )
    pairs = AlertEngine._pairs(state)
    assert [(a.tla, b.tla) for a, b in pairs] == [("BBB", "AAA")]
    p_now, p_base = AlertEngine._undercut(state, "2", "1")
    # worn mediums: pitting first and getting the fresh-tyre lap gains the place
    assert p_now > p_base


def test_hit_rate_counts_only_taken_alerts():
    alerts = [
        Alert("u1", "undercut", 10, "", "", [], outcome="worked"),
        Alert("u2", "undercut", 12, "", "", [], outcome="failed"),
        Alert("u3", "undercut", 14, "", "", [], outcome="not taken"),
        Alert("s1", "sc_call", 20, "", "", []),
    ]
    assert hit_rate(alerts) == {"alerts": 3, "taken": 2, "worked": 1, "hit_rate": 0.5}
