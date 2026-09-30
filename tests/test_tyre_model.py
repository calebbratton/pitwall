import numpy as np
import pytest

from src.models.tyres import CleanLap, age_spread, fit_panel_model, fit_tyre_model

TRUE_DEG = {"MEDIUM": 0.05, "SOFT": 0.12}
TRUE_OFFSET = {"MEDIUM": 0.0, "SOFT": -0.6}  # fresh soft is 0.6 s faster than fresh medium
FUEL = 0.04


def _race(seed=1, noise=0.15, drivers=16):
    """Two-stop-free synthetic race: each driver runs MEDIUM for laps 1-25, SOFT for 26-50."""
    rng = np.random.default_rng(seed)
    laps = []
    for d in range(drivers):
        base = 90 + rng.normal(0, 0.4)  # car/driver pace
        for stint, (compound, start, end) in enumerate([("MEDIUM", 1, 25), ("SOFT", 26, 50)], 1):
            for lap in range(start, end + 1):
                age = lap - start
                t = base + TRUE_OFFSET[compound] + TRUE_DEG[compound] * age - FUEL * lap
                laps.append(CleanLap(f"D{d}", stint, compound, lap, age, t + rng.normal(0, noise)))
    return laps


def test_recovers_degradation_and_compound_offset():
    model = fit_tyre_model(_race(), method="stint", fuel_gain=FUEL)
    medium, soft = model.compounds["MEDIUM"], model.compounds["SOFT"]
    assert model.reference_compound in ("MEDIUM", "SOFT")
    assert medium.deg_s_per_lap == pytest.approx(0.05, abs=0.01)
    assert soft.deg_s_per_lap == pytest.approx(0.12, abs=0.01)
    assert medium.deg_low <= 0.05 <= medium.deg_high
    offset = soft.offset_s - medium.offset_s
    assert offset == pytest.approx(-0.6, abs=0.15)
    assert (soft.n_stints, medium.n_laps) == (16, 16 * 25)
    # Soft is faster fresh, but wears faster: slower than a fresh medium beyond ~5 laps old.
    assert model.lap_delta("SOFT", 0) < model.lap_delta("MEDIUM", 0)
    assert model.lap_delta("SOFT", 10) > model.lap_delta("MEDIUM", 0)


def test_wrong_fuel_prior_shifts_degradation_and_is_reflected_in_uncertainty():
    no_fuel = fit_tyre_model(_race(), method="stint", fuel_gain=0.0)
    # Without fuel correction the raw trend understates wear by exactly the fuel gain.
    assert no_fuel.compounds["MEDIUM"].deg_s_per_lap == pytest.approx(0.05 - FUEL, abs=0.01)
    width = lambda f: f.deg_high - f.deg_low
    assert width(
        fit_tyre_model(_race(), method="stint", fuel_uncertainty=0.03).compounds["SOFT"]
    ) > width(fit_tyre_model(_race(), method="stint", fuel_uncertainty=0.0).compounds["SOFT"])


def test_short_stints_and_traffic_laps_are_ignored():
    laps = _race()
    # A 3-lap stint (too short) and a lap stuck in traffic (+3 s).
    laps += [CleanLap("X", 1, "HARD", n, n - 1, 95.0) for n in range(1, 4)]
    laps = [
        CleanLap(l.driver, l.stint, l.compound, l.lap_number, l.tyre_age, l.lap_time + 3.0)
        if (l.driver, l.lap_number) == ("D0", 10)
        else l
        for l in laps
    ]
    model = fit_tyre_model(laps, method="stint")
    assert "HARD" not in model.compounds
    assert model.compounds["MEDIUM"].deg_s_per_lap == pytest.approx(0.05, abs=0.01)


def test_empty_input():
    model = fit_tyre_model([])
    assert model.compounds == {} and model.reference_compound is None


EVOLUTION = 0.03  # track gets 0.03 s/lap faster as rubber goes down


def _staggered_race(seed=2, noise=0.15, drivers=16, laps_total=50):
    """Drivers pit on different laps (15..30), with track evolution on top of fuel burn."""
    rng = np.random.default_rng(seed)
    laps = []
    for d in range(drivers):
        base = 90 + rng.normal(0, 0.4)
        stop = 15 + d
        order = (
            [("MEDIUM", 1, stop), ("HARD", stop + 1, laps_total)]
            if d % 2
            else [("HARD", 1, stop), ("MEDIUM", stop + 1, laps_total)]
        )
        for stint, (compound, start, end) in enumerate(order, 1):
            for lap in range(start, end + 1):
                age = lap - start
                t = base + OFFSETS[compound] + DEG[compound] * age - (FUEL + EVOLUTION) * lap
                laps.append(CleanLap(f"D{d}", stint, compound, lap, age, t + rng.normal(0, noise)))
    return laps


DEG = {"MEDIUM": 0.06, "HARD": 0.03}
OFFSETS = {"MEDIUM": 0.0, "HARD": 0.4}  # fresh hard 0.4 s slower than fresh medium


def test_panel_model_separates_track_evolution_the_stint_model_cannot():
    laps = _staggered_race()
    assert age_spread(laps) > 2
    panel = fit_tyre_model(laps)
    assert panel.method == "panel" and panel.fuel_gain_s_per_lap is None
    for c, true_deg in DEG.items():
        assert panel.compounds[c].deg_s_per_lap == pytest.approx(true_deg, abs=0.01)
        assert panel.compounds[c].deg_low <= true_deg <= panel.compounds[c].deg_high
    hard_vs_medium = panel.compounds["HARD"].offset_s - panel.compounds["MEDIUM"].offset_s
    assert hard_vs_medium == pytest.approx(0.4, abs=0.1)
    # The stint model with a fuel-only prior mistakes track evolution for less tyre wear.
    stint = fit_tyre_model(laps, method="stint", fuel_gain=FUEL)
    assert stint.compounds["MEDIUM"].deg_s_per_lap == pytest.approx(
        DEG["MEDIUM"] - EVOLUTION, abs=0.01
    )


def test_panel_falls_back_when_everyone_stops_together():
    laps = _race()  # everyone pits on lap 25: no age variation within a lap
    assert age_spread(laps) < 1
    assert fit_panel_model(laps) is None
    model = fit_tyre_model(laps)
    assert model.method == "stint" and "too similar" in model.note
