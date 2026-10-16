"""Race timing sources: the OpenF1 rebuild of the live feed and the recording -> archive ->
OpenF1 order, on a small in-memory race. No network."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.livetiming.archive import Message
from src.livetiming.monitor import RaceMonitor
from src.livetiming.openf1_feed import openf1_messages
from src.livetiming.sources import covers_race, race_messages
from src.tools.openf1 import OpenF1Client

START = datetime.fromisoformat("2026-10-11T12:00:00+00:00")
LAPS = 6


def _iso(t: datetime) -> str:
    return t.isoformat()


class FakeOpenF1(OpenF1Client):
    """Cars 1 and 2 run 6 laps; car 2 pits at the end of lap 3 under a VSC (MEDIUM -> HARD)
    and finishes 0.1 s behind car 1 officially, though its last lap's start + duration would
    put it ahead. Car 3 retires before completing a lap."""

    def __init__(self):
        laps = []
        for car, pace in ((1, 90.0), (2, 90.2)):
            t = START + timedelta(seconds=0.5 * (car - 1))
            for n in range(1, LAPS + 1):
                duration = pace + (25.0 if car == 2 and n in (3, 4) else 0.0)
                if car == 2 and n == LAPS:
                    duration -= 1.0  # the estimate puts car 2 ahead at the flag
                laps.append(
                    {
                        "driver_number": car,
                        "lap_number": n,
                        "date_start": _iso(t),
                        "lap_duration": duration,
                        "is_pit_out_lap": car == 2 and n == 4,
                    }
                )
                t += timedelta(seconds=duration)
        lap3 = START + timedelta(seconds=2 * 90.0)
        self.rows = {
            "sessions": [
                {
                    "session_key": 9,
                    "meeting_key": 1,
                    "session_name": "Race",
                    "session_type": "Race",
                    "country_name": "Singapore",
                    "location": "Marina Bay",
                    "circuit_short_name": "Singapore",
                    "circuit_key": None,
                    "year": 2026,
                    "date_start": _iso(START),
                    "date_end": _iso(START + timedelta(hours=2)),
                }
            ],
            "drivers": [
                {"driver_number": n, "name_acronym": a, "full_name": a, "team_name": "T"}
                for n, a in ((1, "AAA"), (2, "BBB"), (3, "CCC"))
            ],
            "laps": laps,
            "stints": [
                {
                    "driver_number": 1,
                    "stint_number": 1,
                    "compound": "MEDIUM",
                    "lap_start": 1,
                    "lap_end": 6,
                    "tyre_age_at_start": 0,
                },
                {
                    "driver_number": 2,
                    "stint_number": 1,
                    "compound": "MEDIUM",
                    "lap_start": 1,
                    "lap_end": 3,
                    "tyre_age_at_start": 0,
                },
                {
                    "driver_number": 2,
                    "stint_number": 2,
                    "compound": "HARD",
                    "lap_start": 4,
                    "lap_end": 6,
                    "tyre_age_at_start": 0,
                },
            ],
            "pit": [
                {
                    "driver_number": 2,
                    "lap_number": 3,
                    "date": _iso(lap3 + timedelta(seconds=100)),
                    "lane_duration": 22.0,
                }
            ],
            "race_control": [
                {
                    "date": _iso(lap3 + timedelta(seconds=30)),
                    "lap_number": 3,
                    "category": "SafetyCar",
                    "flag": None,
                    "message": "VSC DEPLOYED",
                },
                {
                    "date": _iso(lap3 + timedelta(seconds=200)),
                    "lap_number": 4,
                    "category": "SafetyCar",
                    "flag": None,
                    "message": "VSC ENDING",
                },
            ],
            "session_result": [
                {"driver_number": 1, "position": 1, "number_of_laps": LAPS, "gap_to_leader": 0},
                {"driver_number": 2, "position": 2, "number_of_laps": LAPS, "gap_to_leader": 0.1},
                {"driver_number": 3, "position": None, "number_of_laps": 0, "dnf": True},
            ],
            "position": [
                {"date": _iso(START), "driver_number": n, "position": n} for n in (1, 2, 3)
            ],
        }

    def _fetch(self, endpoint, params):
        rows = self.rows.get(endpoint, [])
        return [
            r
            for r in rows
            if all(r.get(k) == v for k, v in params.items() if k in ("driver_number",))
        ]


def _monitor(messages):
    monitor = RaceMonitor()
    for m in messages:
        monitor.feed(m)
    return monitor


def test_openf1_rebuild_gives_the_monitor_laps_stops_tyres_and_flags():
    client = FakeOpenF1()
    messages = openf1_messages(client, client.get_session(2026, "Singapore"))
    monitor = _monitor(messages)
    laps = monitor.laps["2"]
    assert [r.lap for r in laps] == list(range(1, LAPS + 1))
    by_lap = {r.lap: r for r in laps}
    assert by_lap[3].pit_in and by_lap[4].pit_out and not by_lap[5].pit_out
    assert by_lap[3].compound == "MEDIUM" and by_lap[5].compound == "HARD"
    assert by_lap[3].neutralised and not by_lap[2].neutralised
    assert by_lap[3].time_s == pytest.approx(115.2)
    snap = monitor.snapshot()
    # The official gap decides the finish, not the last lap's start + duration.
    assert [d.tla for d in snap.drivers] == ["AAA", "BBB", "CCC"]
    assert snap.driver("BBB").gap_to_leader_s == pytest.approx(0.1, abs=0.01)
    assert snap.driver("CCC").retired
    assert snap.total_laps == LAPS
    assert snap.median_pit_lane_s == 22.0


def _laps_message(car: str, n: int) -> Message:
    return Message(timedelta(seconds=n), "TimingData", {"Lines": {car: {"NumberOfLaps": n}}})


def test_a_recording_must_cover_the_race():
    header = [Message(timedelta(0), "LapCount", {"CurrentLap": 1, "TotalLaps": 50})]
    complete = header + [_laps_message("1", n) for n in range(1, 51)]
    late_start = header + [_laps_message("1", n) for n in range(30, 51)]
    final_state_only = header + [_laps_message("1", 50)]
    overlapping = header + [_laps_message("1", n) for n in [*range(1, 51), *range(1, 30)]]
    assert covers_race(complete)
    assert not covers_race(late_start)
    assert not covers_race(final_state_only)
    assert not covers_race(overlapping)


def test_sources_fall_through_recording_and_archive_to_openf1(monkeypatch, tmp_path):
    from src.sim import pit_review

    def no_archive(*_):
        raise LookupError("0 archived Race sessions match")

    monkeypatch.setattr(pit_review, "archive_race", no_archive)
    # A recording that only caught the final state after a reconnect: not usable.
    folder = Path(tmp_path / "2026-10-11T140000Z")
    folder.mkdir()
    (folder / "SessionInfo.jsonStream").write_text(
        '00:00:01.000{"Name":"Race","Type":"Race","StartDate":"2026-10-11T20:00:00"}\n'
    )
    (folder / "LapCount.jsonStream").write_text('00:00:01.000{"CurrentLap":62,"TotalLaps":62}\n')
    (folder / "TimingData.jsonStream").write_text(
        '00:00:01.000{"Lines":{"1":{"NumberOfLaps":62}}}\n'
    )
    client = FakeOpenF1()
    race = race_messages(2026, "Singapore", "2026-10-11", client=client, recordings=tmp_path)
    assert race.source == "OpenF1"
    assert any(m.topic == "TimingData" for m in race.messages)
    with pytest.raises(LookupError, match="no complete recording"):
        race_messages(
            2026,
            "Singapore",
            "2026-10-11",
            sources=("recording",),
            client=client,
            recordings=tmp_path,
        )
