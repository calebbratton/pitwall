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


def test_refresh_warehouse_skips_while_openf1_is_closed(monkeypatch):
    """OpenF1 refuses every request during a live session; the predictions carry on."""
    import src.warehouse.build as build_mod
    import src.warehouse.ingest as ingest_mod
    from src.sim.prerace import refresh_warehouse
    from src.tools.openf1 import OpenF1Error

    def closed(*a, **k):
        raise OpenF1Error("OpenF1 refused sessions (HTTP 401)")

    built = []
    monkeypatch.setattr(ingest_mod, "ingest", closed)
    monkeypatch.setattr(build_mod, "build", lambda: built.append(1))
    assert refresh_warehouse(2026) == 0 and built == []

    monkeypatch.setattr(ingest_mod, "ingest", lambda *a, **k: [11730])
    assert refresh_warehouse(2026) == 1 and built == [1]
