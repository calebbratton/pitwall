from src.sim.pit_review import Branch, PitReview
from src.sim.verdicts import verdict_for


def _review(expected, notes=()):
    branches = [Branch(label, e, 0.1, 0.3, 0.0, 0.0) for label, e in expected]
    return PitReview("NOR", 15, "GREEN", "MEDIUM", "HARD", 14, {}, branches, "", list(notes))


def test_verdict_labels():
    assert (
        verdict_for(_review([("pit lap 15 (actual)", 1.4), ("stay out 1 more lap", 1.5)]))["label"]
        == "right call"
    )
    unlucky = _review(
        [("pit lap 15 (actual)", 1.4), ("stay out 1 more lap", 1.45)],
        [
            "ANT stopped under a SC/VSC ...: the cost came from the neutralisation's timing, not from this decision"
        ],
    )
    assert verdict_for(unlucky)["label"] == "unlucky"
    better = verdict_for(_review([("pit lap 15 (actual)", 3.0), ("stay out 3 more laps", 2.2)]))
    assert better["label"] == "better option" and "Stay out 3 more laps" in better["summary"]
