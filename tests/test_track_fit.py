from src.analysis.track_fit import RaceFit, _outline_length_km, correlations


def test_outline_length_units_are_decimetres():
    square_1km = [[0, 0], [2500, 0], [2500, 2500], [0, 2500]]  # 4 x 250 m
    assert round(_outline_length_km(square_1km), 3) == 1.0


def test_correlations_rank_features_and_skip_sparse_ones():
    fits = [
        RaceFit(
            f"R{i}",
            avg_speed_kmh=200 + 10 * i,
            top_speed_kmh=300.0,
            corners_per_km=None,
            length_km=5.0 + (i % 2),
            compound_softness=3.0 + (i % 3),
            quali_gap_pct=1.0 - 0.1 * i,
            race_pace_gap_pct=None,
            adjusted_finish=None,
        )
        for i in range(8)
    ]
    rows = correlations(fits, "quali_gap_pct")
    assert rows[0]["feature"] == "avg_speed_kmh" and rows[0]["r"] == -1.0  # faster = closer to pole
    assert "corners_per_km" not in {r["feature"] for r in rows}  # no data
    assert correlations(fits, "race_pace_gap_pct") == []
