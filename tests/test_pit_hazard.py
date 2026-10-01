import numpy as np

from src.sim.pit_hazard import FEATURES, Rows, columns, evaluate, log_loss, recalibrated


def test_hazard_learns_tyre_age_leave_one_race_out():
    rng = np.random.default_rng(1)
    n = 6000
    X = np.zeros((n, len(FEATURES)))
    age = rng.integers(1, 40, n).astype(float)
    X[:, FEATURES.index("age")] = age
    X[:, FEATURES.index("age_sq")] = age**2 / 100
    y = (rng.random(n) < 1 / (1 + np.exp(-(age - 30) / 3))).astype(float)
    rows = Rows(
        X, y, np.repeat(np.arange(6), n // 6), [(2025, f"R{i}") for i in range(6)], np.zeros(n)
    )
    p = evaluate(rows, ("age", "age_sq"))
    assert log_loss(p, y) < log_loss(np.full(n, y.mean()), y)
    q = recalibrated(rows, p)
    assert q.shape == p.shape and ((q > 0) & (q < 1)).all()
    assert columns(("age",)) == [FEATURES.index("age")]
