import asyncio
from datetime import timedelta

from src.livetiming.archive import Message
from src.livetiming.monitor import RaceMonitor, replay


def _msg(seconds: float, topic: str, data: dict) -> Message:
    return Message(timedelta(seconds=seconds), topic, data)


def _race_start() -> list[Message]:
    return [
        _msg(0, "SessionInfo", {"Name": "Race", "Meeting": {"Name": "Test GP"}}),
        _msg(0, "TrackStatus", {"Status": "1", "Message": "AllClear"}),
        _msg(1, "DriverList", {"4": {"Tla": "NOR"}, "1": {"Tla": "VER"}}),
        _msg(1, "LapCount", {"CurrentLap": 1, "TotalLaps": 50}),
        _msg(
            1,
            "TimingData",
            {
                "Lines": {
                    "4": {"Position": "1", "GapToLeader": "LAP 1", "NumberOfLaps": 0},
                    "1": {"Position": "2", "GapToLeader": "+20.0", "NumberOfLaps": 0},
                }
            },
        ),
        _msg(
            1,
            "TimingAppData",
            {
                "Lines": {
                    "4": {"Stints": [{"Compound": "MEDIUM", "New": "true", "TotalLaps": 0}]},
                    "1": {"Stints": [{"Compound": "HARD", "New": "true", "TotalLaps": 0}]},
                }
            },
        ),
        _msg(
            2,
            "RaceControlMessages",
            {"Messages": [{"Lap": 1, "Category": "Flag", "Message": "GREEN LIGHT"}]},
        ),
    ]


def _types(events):
    """Event types, without the per-lap history events (tested separately)."""
    return [e["type"] for e in events if e["type"] != "lap"]


def test_monitor_emits_session_status_race_control_and_pit_calls_on_safety_car():
    monitor = RaceMonitor(radio_base_url="https://static/session/")
    events = [e for m in _race_start() for e in monitor.feed(m)]
    assert _types(events) == ["track_status", "session", "race_control"]
    assert events[1] == {
        "type": "session",
        "meeting": "Test GP",
        "session": "Race",
        "session_type": "",  # the test feed's SessionInfo has no Type; the name says Race
        "total_laps": 50,
    }

    monitor.feed(_msg(3, "LapCount", {"CurrentLap": 20}))
    sc = monitor.feed(_msg(4, "TrackStatus", {"Status": "4", "Message": "SCDeployed"}))
    assert _types(sc) == ["track_status", "pit_calls"]
    calls = {c["tla"]: c["call"] for c in sc[1]["report"]["calls"]}
    assert calls == {"NOR": "PIT", "VER": "PIT"}  # both still need a second compound

    # Staying under the safety car doesn't re-trigger calls; a VSC straight after doesn't either.
    assert monitor.feed(_msg(5, "TrackStatus", {"Status": "4"})) == []
    assert _types(monitor.feed(_msg(6, "TrackStatus", {"Status": "6"}))) == ["track_status"]


def test_monitor_race_control_and_radio_are_emitted_once_each():
    monitor = RaceMonitor(radio_base_url="https://static/session/")
    for m in _race_start():
        monitor.feed(m)
    rc = monitor.feed(
        _msg(
            3,
            "RaceControlMessages",
            {"Messages": {"1": {"Lap": 2, "Message": "SAFETY CAR DEPLOYED"}}},
        )
    )
    assert [e["message"] for e in rc] == ["SAFETY CAR DEPLOYED"]

    clip = {"Utc": "t", "RacingNumber": "4", "Path": "TeamRadio/NOR_1.mp3"}
    radio = monitor.feed(_msg(4, "TeamRadio", {"Captures": [clip]}))
    assert radio == [
        {
            "type": "radio",
            "driver": "4",
            "tla": "NOR",
            "utc": "t",
            "url": "https://static/session/TeamRadio/NOR_1.mp3",
            "text": None,
        }
    ]
    assert monitor.feed(_msg(5, "TeamRadio", {"Captures": {"0": {"Utc": "t"}}})) == []


class FakeSession:
    path = "2099/test/race/"

    def __init__(self, messages):
        self._messages = messages

    def messages(self, topics):
        return iter(self._messages)


def test_replay_fast_forwards_then_streams_with_snapshots():
    messages = [
        *_race_start(),
        _msg(90, "LapCount", {"CurrentLap": 2}),
        _msg(180, "LapCount", {"CurrentLap": 3}),
        _msg(181, "TrackStatus", {"Status": "4"}),
    ]

    async def collect():
        return [e async for e in replay(FakeSession(messages), speed=1e6, from_lap=2)]

    events = asyncio.run(collect())
    kinds = _types(events)
    # Fast-forward drops the pre-lap-2 events except `session`, then a snapshot marks the jump.
    assert kinds[:2] == ["session", "snapshot"]
    assert "race_control" not in kinds
    assert kinds.count("pit_calls") == 1
    assert kinds[-1] == "end"
    assert events[-2]["type"] == "snapshot" and events[-2]["lap"] == 3
    assert events[-2]["drivers"][0]["tla"] == "NOR"


class FakeTranscriber:
    def __init__(self, cache: dict[str, str]):
        self.cache = cache
        self.calls: list[tuple[str, list[str]]] = []

    def cached(self, url):
        return self.cache.get(url)

    async def transcribe(self, url, names=()):
        self.calls.append((url, list(names)))
        await asyncio.sleep(0)
        return "Box box, safety car."


def test_replay_transcribes_new_clips_in_background_and_uses_cache():
    base = "https://livetiming.formula1.com/static/2099/test/race/"
    messages = [
        *_race_start(),
        _msg(3, "DriverList", {"4": {"FirstName": "Lando", "LastName": "Norris"}}),
        _msg(4, "TeamRadio", {"Captures": [{"RacingNumber": "4", "Path": "TeamRadio/new.mp3"}]}),
        _msg(
            5, "TeamRadio", {"Captures": {"1": {"RacingNumber": "1", "Path": "TeamRadio/old.mp3"}}}
        ),
        _msg(6, "LapCount", {"CurrentLap": 2}),
    ]
    transcriber = FakeTranscriber({base + "TeamRadio/old.mp3": "Copy."})

    async def collect():
        session = FakeSession(messages)
        return [e async for e in replay(session, speed=1e6, transcriber=transcriber)]

    events = asyncio.run(collect())
    radios = {e["url"].rsplit("/", 1)[-1]: e for e in events if e["type"] == "radio"}
    assert radios["new.mp3"]["text"] is None  # sent immediately, transcript follows
    assert radios["old.mp3"]["text"] == "Copy."  # cached: filled in, no API call
    transcript = next(e for e in events if e["type"] == "radio_transcript")
    assert transcript == {
        "type": "radio_transcript",
        "url": base + "TeamRadio/new.mp3",
        "tla": "NOR",
        "text": "Box box, safety car.",
    }
    assert transcriber.calls == [(base + "TeamRadio/new.mp3", ["Lando Norris"])]
    kinds = [e["type"] for e in events]
    assert kinds.index("radio_transcript") > kinds.index("radio") and kinds[-1] == "end"


def _grid_messages(session_name: str = "Race") -> list[Message]:
    lines = {
        str(n): {"RacingNumber": str(n), "GridPos": str(i + 1)} for i, n in enumerate(range(1, 13))
    }
    return [
        _msg(0, "SessionInfo", {"Name": session_name, "Meeting": {"Name": "Test GP"}}),
        _msg(1, "LapCount", {"CurrentLap": 1, "TotalLaps": 50}),
        _msg(8, "TimingAppData", {"Lines": lines}),
        _msg(9, "TimingAppData", {"Lines": {"1": {"Stints": [{"Compound": "SOFT"}]}}}),
        _msg(100, "LapCount", {"CurrentLap": 2}),
    ]


def test_on_grid_fires_once_with_the_official_grid():
    calls = []

    def on_grid(monitor):
        calls.append(monitor.starting_grid())
        return {"type": "prediction", "lap": 0}

    async def collect(messages):
        return [e async for e in replay(FakeSession(messages), speed=1e6, on_grid=on_grid)]

    events = asyncio.run(collect(_grid_messages()))
    assert len(calls) == 1 and calls[0][1] == 1 and calls[0][12] == 12
    assert [e for e in events if e["type"] == "prediction"] == [{"type": "prediction", "lap": 0}]

    calls.clear()
    asyncio.run(collect(_grid_messages("Qualifying")))
    assert calls == []


def test_weather_event_and_session_start():
    from datetime import UTC, datetime

    from src.livetiming.monitor import forecast_event, session_start_utc, weather_event

    e = weather_event({"AirTemp": "26.2", "TrackTemp": "47.3", "Rainfall": "1", "WindSpeed": "0.4"})
    assert e["track_c"] == 47.3 and e["raining"] is True and e["humidity"] is None
    assert weather_event({"AirTemp": "20"}) is None
    info = {
        "StartDate": "2026-09-26T15:00:00",
        "GmtOffset": "04:00:00",
        "Meeting": {"Location": "Baku"},
    }
    assert session_start_utc(info) == datetime(2026, 9, 26, 11, 0, tzinfo=UTC)
    assert session_start_utc(
        {"StartDate": "2026-06-07T14:00:00", "GmtOffset": "-04:00:00"}
    ) == datetime(2026, 6, 7, 18, 0, tzinfo=UTC)
    assert forecast_event(info) is None  # forecasts are blocked in tests: no event, no crash


def test_lap_events_carry_the_lap_history():
    monitor = RaceMonitor()
    events = []
    for m in [
        *_race_start(),
        _msg(
            90,
            "TimingData",
            {
                "Lines": {
                    "4": {
                        "NumberOfLaps": 1,
                        "LastLapTime": {
                            "Value": "1:32.500",
                            "PersonalFastest": True,
                            "OverallFastest": True,
                        },
                    }
                }
            },
        ),
    ]:
        events += monitor.feed(m)
    laps = [e for e in events if e["type"] == "lap"]
    assert laps == [
        {
            "type": "lap",
            "number": "4",
            "tla": "NOR",
            "lap": 1,
            "time": 92.5,
            "compound": "MEDIUM",
            "tyre_age": 0,
            "pit_in": False,
            "pit_out": False,
            "neutralised": False,
            "personal_best": True,
            "overall_best": True,
        }
    ]


def test_practice_and_qualifying_use_best_lap_timing_and_skip_race_calls():
    lines = {
        "1": {"Position": "1", "BestLapTime": {"Value": "1:45.387"}, "TimeDiffToFastest": ""},
        "4": {
            "Position": "2",
            "BestLapTime": {"Value": "1:45.787"},
            "TimeDiffToFastest": "+0.400",
            "TimeDiffToPositionAhead": "+0.400",
        },
    }
    monitor = RaceMonitor()
    events = []
    for m in [
        _msg(
            0,
            "SessionInfo",
            {"Name": "Practice 1", "Type": "Practice", "Meeting": {"Name": "Test GP"}},
        ),
        _msg(1, "DriverList", {"4": {"Tla": "NOR"}, "1": {"Tla": "VER"}}),
        _msg(1, "TimingData", {"Lines": lines}),
        _msg(2, "TrackStatus", {"Status": "6"}),  # a VSC test in practice
    ]:
        events += monitor.feed(m)
    assert next(e for e in events if e["type"] == "session")["session_type"] == "Practice"
    assert not [e for e in events if e["type"] == "pit_calls"]
    snap = monitor.snapshot()
    assert not snap.is_race
    nor = snap.driver("NOR")
    assert (nor.gap_to_leader_s, nor.interval_s, nor.best_lap_s) == (0.4, 0.4, 105.787)
    assert monitor.snapshot_event()["rejoin"] == {}

    quali = RaceMonitor()
    q_line = {
        "Position": "2",
        "Stats": [
            {"TimeDiffToFastest": "+0.3"},
            {"TimeDiffToFastest": "+0.2"},
            {"TimeDiffToFastest": "+0.9"},
        ],
        "BestLapTimes": [{"Value": "1:44.0"}, {"Value": "1:43.7"}, {"Value": "1:44.1"}],
        "KnockedOut": False,
    }
    for m in [
        _msg(
            0,
            "SessionInfo",
            {"Name": "Qualifying", "Type": "Qualifying", "Meeting": {"Name": "Test GP"}},
        ),
        _msg(1, "DriverList", {"4": {"Tla": "NOR"}}),
        _msg(1, "TimingData", {"SessionPart": 2, "Lines": {"4": q_line}}),
    ]:
        quali.feed(m)
    nor = quali.snapshot().driver("NOR")
    assert (nor.gap_to_leader_s, nor.best_lap_s) == (0.2, 103.7)  # the Q2 numbers
