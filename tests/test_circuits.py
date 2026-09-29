import asyncio
import json
from pathlib import Path

from src.livetiming.circuits import Circuit, PitLoss, parse_circuit
from src.livetiming.monitor import replay
from src.livetiming.snapshot import build_snapshot
from src.livetiming.state import PitLaneTime, TimingState
from src.livetiming.strategy import pit_calls
from tests.test_monitor import FakeSession, _msg, _race_start

MIAMI_SC = Path(__file__).parent / "fixtures/livetiming/miami_2024_sc_lap28.json"
RAW = {
    "circuitName": "Miami",
    "x": [0, 100, 100, 0],
    "y": [0, 0, 50, 50],
    "rotation": 2,
    "corners": [{"number": 1, "trackPosition": {"x": 100, "y": 0}}],
    "marshalSectors": [{"number": 1, "trackPosition": {"x": 50, "y": 0}}],
    "pitLoss": {"normal": "20.27", "sc": "12.84", "vsc": "14.66"},
}


def test_parse_circuit_and_track_event():
    circuit = parse_circuit(RAW, key=151, year=2024)
    assert circuit.pit_loss == PitLoss(20.27, 12.84, 14.66, "measured (MultiViewer)")
    event = circuit.track_event()
    assert event["source"] == "multiviewer" and event["rotation"] == 2.0
    assert event["points"] == [[0, 0], [100, 0], [100, 50], [0, 50]]
    assert event["bounds"] == [0, 0, 100, 50]
    assert event["corners"] == [{"number": 1, "x": 100, "y": 0}]


def test_missing_pit_loss_is_tolerated():
    raw = {k: v for k, v in RAW.items() if k != "pitLoss"}
    assert parse_circuit(raw, 151, 2024).pit_loss is None


def test_pit_calls_use_measured_loss_when_given():
    data = json.loads(MIAMI_SC.read_text())
    snap = build_snapshot(
        TimingState(
            topics=data["topics"],
            pit_lane_times=[PitLaneTime(**p) for p in data["pit_lane_times"]],
        )
    )
    measured = pit_calls(snap, PitLoss(20.27, 12.84, 14.66, "measured (MultiViewer)"))
    assert measured.pit_loss_now_s == 12.8 and measured.green_pit_loss_s == 20.3
    nor = next(c for c in measured.calls if c.tla == "NOR")
    assert nor.call == "PIT" and nor.places_at_risk == 1  # VER, 11.4s behind, is now at risk
    assert "measured (MultiViewer)" in measured.assumptions[0]
    assert "estimated" in pit_calls(snap).assumptions[0]


def test_replay_prefers_circuit_data_for_track_and_pit_loss():
    circuit = parse_circuit(RAW, key=151, year=2024)
    messages = [
        *_race_start(),
        _msg(90, "LapCount", {"CurrentLap": 20}),
        _msg(91, "TrackStatus", {"Status": "4"}),
    ]

    async def collect():
        loader = lambda msgs: circuit
        return [e async for e in replay(FakeSession(messages), speed=1e6, circuit_loader=loader)]

    events = asyncio.run(collect())
    assert events[0]["type"] == "track" and events[0]["source"] == "multiviewer"
    report = next(e for e in events if e["type"] == "pit_calls")["report"]
    assert report["pit_loss_now_s"] == 12.8


def test_circuit_is_frozen_dataclass():
    assert Circuit.__dataclass_params__.frozen
