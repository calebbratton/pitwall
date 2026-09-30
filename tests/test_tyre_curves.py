import duckdb
import numpy as np

from src.models.tyre_curves import BAND, fit_curves, late_slope
from src.sim.inrace import CarState, RaceState, _life, tyre_outlook

C4_LOSS_PER_LAP = 0.03


def _warehouse(seed=0):
    """Two synthetic 'Baku' races (C4 = MEDIUM): staggered stops, fuel/evolution per lap,
    driver pace, and a known linear C4 wear of 0.03 s/lap; C5 (SOFT) wears 0.08 s/lap."""
    rng = np.random.default_rng(seed)
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE races (session_key INT, location VARCHAR, year INT, session_name VARCHAR)"
    )
    con.execute(
        "CREATE TABLE clean_laps (session_key INT, driver_number INT, lap_number INT, "
        "compound VARCHAR, tyre_age INT, lap_time DOUBLE)"
    )
    rows = []
    for sk in (1, 2):
        con.execute("INSERT INTO races VALUES (?, 'Baku', 2026, 'Race')", [sk])
        for d in range(16):
            base = 100 + rng.normal(0, 0.4)
            stop = 12 + d  # staggered
            used = d % 3  # some start on used sets, as in real races (breaks age == lap - 1)
            first = "SOFT" if d % 2 else "MEDIUM"
            second = "MEDIUM" if first == "SOFT" else "SOFT"
            for lap in range(2, 51):
                compound, age = (first, lap - 1 + used) if lap <= stop else (second, lap - stop - 1)
                wear = C4_LOSS_PER_LAP * age if compound == "MEDIUM" else 0.08 * age
                t = (
                    base
                    - 0.07 * lap
                    + wear
                    + (0.3 if compound == "MEDIUM" else 0)
                    + rng.normal(0, 0.1)
                )
                rows.append((sk, d, lap, compound, age, t))
    con.executemany("INSERT INTO clean_laps VALUES (?, ?, ?, ?, ?, ?)", rows)
    return con


def test_curves_recover_known_wear_by_c_number():
    curves = fit_curves(_warehouse(), 2026)
    assert set(curves) == {"C4", "C5"}  # labels mapped to Baku's C-numbers
    c4 = curves["C4"]
    for age, loss, n in c4.bands:
        if n >= 40:
            assert abs(loss - C4_LOSS_PER_LAP * age) < 0.12, (age, loss)
    assert abs(late_slope(c4) - C4_LOSS_PER_LAP) < 0.01
    assert c4.drop_off_age is None and c4.life_laps == c4.supported_to + BAND


def test_tyre_note_cites_measured_life():
    curves = fit_curves(_warehouse(), 2026)
    life = _life("C4", curves, 2026)
    car = CarState("1", "VER", 1, 0.0, "MEDIUM", 10, 1, False, 0.0, "C4", **life)
    state = RaceState(20, 15, "GREEN", [car], {}, {}, 22.0, 13.5)
    risk, note = tyre_outlook(car, state)
    assert risk == "low" and "C4 ran" in note and "no drop-off" in note
    over = CarState("1", "VER", 1, 0.0, "MEDIUM", life["life_laps"], 1, False, 0.0, "C4", **life)
    risk, note = tyre_outlook(over, state)
    assert risk in ("medium", "high") and "beyond anything seen" in note


def test_bands_the_data_cannot_identify_are_dropped():
    """Everyone on fresh SOFT from lap 1 and stopping together: tyre age == lap for the whole
    field in stint 1, so the SOFT curve can't be separated from lap effects there."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE races (session_key INT, location VARCHAR, year INT, session_name VARCHAR)"
    )
    con.execute("INSERT INTO races VALUES (1, 'Baku', 2026, 'Race')")
    con.execute(
        "CREATE TABLE clean_laps (session_key INT, driver_number INT, lap_number INT, "
        "compound VARCHAR, tyre_age INT, lap_time DOUBLE)"
    )
    rows = [
        (1, d, lap, "SOFT", lap - 1, 100.0 + 0.05 * lap + d * 0.1)
        for d in range(10)
        for lap in range(2, 20)
    ]
    con.executemany("INSERT INTO clean_laps VALUES (?, ?, ?, ?, ?, ?)", rows)
    curve = fit_curves(con, 2026)["C5"]
    assert [b[0] for b in curve.bands] == [0]  # only the baseline: nothing identifiable
