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
