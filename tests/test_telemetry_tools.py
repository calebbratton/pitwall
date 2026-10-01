import json
from pathlib import Path

import pytest

from src.tools.models import Lap
from src.tools.openf1 import MockOpenF1Client
from src.tools.telemetry import build_telemetry_tools, key_race_events, pace_summary

MONACO_2024 = Path(__file__).parent / "fixtures/openf1/2024_monaco"


@pytest.fixture
def tools():
    client = MockOpenF1Client(MONACO_2024)
    return {t.name: t for t in build_telemetry_tools(client, client.get_session(2024, "Monaco"))}


def _lap(n: int, t: float | None, pit_out: bool = False) -> Lap:
    return Lap(driver_number=4, lap_number=n, lap_duration=t, is_pit_out_lap=pit_out)


def test_pace_summary_excludes_lap_one_pit_out_and_slow_laps():
    laps = [_lap(1, 95.0), _lap(2, 80.0), _lap(3, 80.2, pit_out=True), _lap(4, 80.4), _lap(5, 120)]
    laps += [_lap(6, 80.6), _lap(7, None)]
    summary = pace_summary(laps)
    assert summary["laps_used"] == 3
    assert summary["laps_excluded"] == [1, 3, 5, 7]
    assert summary["best_s"] == 80.0
    assert summary["trend_s_per_lap"] == pytest.approx(0.15, abs=1e-3)


@pytest.mark.parametrize("driver", ["NOR", "nor", "4", "Norris", "Lando NORRIS"])
def test_driver_resolution(tools, driver):
    stints = json.loads(tools["get_tyre_stints"].invoke({"driver": driver}))
    assert {s["driver"] for s in stints} == {"NOR"}


def test_leaked_reasoning_in_driver_arg_returns_error_with_options(tools):
    out = tools["get_tyre_stints"].invoke({"driver": "Liddle? Actually Lando Norris"})
    assert out.startswith("ERROR: unknown driver")
    assert "NOR (#4)" in out


def test_list_drivers_by_team(tools):
    rows = json.loads(tools["list_drivers"].invoke({"team": "mclaren"}))
    assert sorted(r[1] for r in rows) == ["NOR", "PIA"]
    assert tools["list_drivers"].invoke({"team": "Brabham"}).startswith("ERROR")


def test_lap_times_window_and_oversized_request_falls_back_to_summary(tools):
    rows = json.loads(
        tools["get_lap_times"].invoke({"driver": "NOR", "lap_start": 29, "lap_end": 31})
    )
    assert rows[1] == [30, 78.403, False]
    summary = json.loads(
        tools["get_lap_times"].invoke({"driver": "NOR", "lap_start": 1, "lap_end": 78})
    )
    assert "pace summary" in summary["note"]
    assert "mean_s" in summary


def test_key_race_events_has_red_flag_and_finish():
    events = json.loads(key_race_events(MockOpenF1Client(MONACO_2024), 9523))
    assert [1, "RED FLAG"] in events
    assert [78, "CHEQUERED FLAG"] in events


def test_race_summary_monaco_2024():
    from src.tools.telemetry import race_summary

    client = MockOpenF1Client(MONACO_2024)
    summary = json.loads(race_summary(client, client.get_session(2024, "Monaco")))
    rows = {r[0]: r for r in summary["rows"]}
    lec = rows["LEC"]
    assert lec[1:6] == ["Ferrari", 1, 1, 0, 25.0]  # team, grid, finish, gained, points
    assert lec[7] == "M1-H78"  # the lap-1 red flag tyre change
    assert summary["rows"][0][0] == "LEC"  # ordered by finish
    assert len(json.dumps(summary)) < 3000  # small enough for the free-tier token budget


def test_neutralised_laps_excluded_from_pace():
    from datetime import UTC, datetime, timedelta

    from src.tools.models import RaceControlMessage
    from src.tools.telemetry import neutralised_windows, representative_laps

    start = datetime(2026, 1, 1, 12, tzinfo=UTC)
    laps = [
        Lap(
            driver_number=4,
            lap_number=n,
            lap_duration=90.0,
            date_start=(start + timedelta(seconds=90 * n)).isoformat(),
        )
        for n in range(1, 21)
    ]
    rc = [
        RaceControlMessage(
            date=(start + timedelta(seconds=90 * 10 + 5)).isoformat(),
            category="SafetyCar",
            message="SAFETY CAR DEPLOYED",
        ),
        RaceControlMessage(
            date=(start + timedelta(seconds=90 * 12 + 5)).isoformat(),
            category="SafetyCar",
            message="SAFETY CAR IN THIS LAP",
        ),
    ]
    windows = neutralised_windows(rc, laps)
    assert [w[2] for w in windows] == ["SC"]
    kept = [lap.lap_number for lap in representative_laps(laps, windows)]
    # SC out during lap 10, in at the end of lap 12: laps 10-12 and the restart lap 13 are
    # excluded (lap 1 always is).
    assert kept == [2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20]


def test_2026_vsc_wording_and_vsc_upgraded_to_sc():
    from datetime import UTC, datetime, timedelta

    from src.tools.models import RaceControlMessage
    from src.tools.telemetry import neutralised_windows

    t0 = datetime(2026, 9, 13, 13, tzinfo=UTC)

    def msg(minutes, text):
        return RaceControlMessage(
            date=(t0 + timedelta(minutes=minutes)).isoformat(), category="SafetyCar", message=text
        )

    kinds = [w[2] for w in neutralised_windows([msg(0, "VSC DEPLOYED"), msg(2, "VSC ENDING")], [])]
    assert kinds == ["VSC"]
    upgraded = neutralised_windows(
        [
            msg(10, "VSC DEPLOYED"),
            msg(11, "SAFETY CAR DEPLOYED"),
            msg(15, "SAFETY CAR IN THIS LAP"),
        ],
        [],
    )
    vsc, sc = upgraded
    assert vsc[2] == "VSC" and vsc[1] == t0 + timedelta(
        minutes=11
    )  # closed by the SC, not open forever
    assert sc[2] == "SC"


def test_review_pit_stop_lists_stops_without_a_lap(tools):
    out = json.loads(tools["review_pit_stop"].invoke({"driver": "LEC"}))
    assert out["driver"] == "LEC" and isinstance(out["pitted_on_laps"], list)
    assert tools["review_pit_stop"].invoke({"driver": "Brabham"}).startswith("ERROR")
