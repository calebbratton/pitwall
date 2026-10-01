import json

from src.sim import prediction_log
from src.sim.scorecard import score

TABLE = [
    {"tla": "RUS", "expected": 1.4, "p_win": 0.5, "p_podium": 0.8, "p_points": 0.95},
    {"tla": "VER", "expected": 2.6, "p_win": 0.2, "p_podium": 0.6, "p_points": 0.9},
    {"tla": "LEC", "expected": 3.1, "p_win": 0.15, "p_podium": 0.5, "p_points": 0.9},
    {"tla": "PIA", "expected": 4.3, "p_win": 0.05, "p_podium": 0.3, "p_points": 0.8},
]


def test_predictions_are_written_once():
    path = prediction_log.save(2026, "Kuala Lumpur", "pre-Q3", {"table": TABLE, "notes": ["n"]})
    assert path.name == "kuala-lumpur-pre-Q3.json"
    record = json.loads(path.read_text())
    assert record["stage"] == "pre-Q3" and record["table"][0]["tla"] == "RUS"
    assert "sim" in record["model"] and "made_at" in record
    # a second save for the same weekend and stage never overwrites the record
    assert prediction_log.save(2026, "Kuala Lumpur", "pre-Q3", {"table": []}) is None
    assert len(prediction_log.load_all(2026)) == 1


def test_score_against_an_order():
    s = score(TABLE, ["VER", "RUS", "LEC", "PIA"])
    assert s["winner"] == "VER" and s["favourite"] == "RUS" and not s["winner_hit"]
    assert s["podium_overlap"] == 3
    assert 0 < s["rank_corr"] < 1
