def test_known_penalties_maps_tlas_to_cars(tmp_path):
    from src.sim.prerace import known_penalties

    f = tmp_path / "p.json"
    f.write_text(
        '{"_note": "x", "Kuala Lumpur": {"penalties": {"HAD": 5, "XXX": 3}, "pit_lane": ["ver"]}}'
    )
    numbers = {"HAD": 6, "VER": 3}
    assert known_penalties(2026, "kuala lumpur", numbers, str(f)) == ({6: 5}, {3})
    assert known_penalties(2026, "Sakhir", numbers, str(f)) == ({}, set())
    assert known_penalties(2026, "Sakhir", numbers, str(tmp_path / "none.json")) == ({}, set())
