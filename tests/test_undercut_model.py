import numpy as np

from src.sim.undercut_data import Attempt
from src.sim.undercut_model import FEATURES, Logistic, evaluate, features


def _attempt(race, gap, worked, year=2025):
    return Attempt(
        year,
        race,
        20,
        50,
        "AAA",
        "BBB",
        gap,
        20,
        18,
        "MEDIUM",
        "MEDIUM",
        "HARD",
        -0.1,
        22.0,
        1,
        worked,
    )


def test_logistic_recovers_a_known_effect():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 1))
    y = (rng.random(400) < 1 / (1 + np.exp(-2 * x[:, 0]))).astype(float)
    p = Logistic(l2=0.1).fit(x, y).predict(np.array([[-2.0], [0.0], [2.0]]))
    assert p[0] < 0.1 and 0.4 < p[1] < 0.6 and p[2] > 0.9


def test_features_and_leave_one_race_out():
    assert len(features(_attempt("R1", 1.0, True))) == len(FEATURES)
    # small gaps work, big gaps fail, in every race: the model should beat the base rate
    rows = [_attempt(f"R{i}", g, g < 2) for i in range(8) for g in (0.5, 1.0, 1.5, 2.5, 3.0, 3.5)]
    result = evaluate(rows)
    assert result["model"]["brier"] < result["base"]["brier"]
    assert result["model"]["auc"] > 0.9
