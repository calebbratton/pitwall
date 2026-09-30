from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from src.tools.openf1 import MockOpenF1Client
from src.warehouse.build import build
from src.warehouse.ingest import ingest

MONACO_2024 = Path(__file__).parent / "fixtures/openf1/2024_monaco"


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("warehouse")
    client = MockOpenF1Client(MONACO_2024)
    now = datetime(2024, 5, 28, tzinfo=UTC)  # after Monaco, before Canada
    ingested = ingest([2024], ["Race"], client=client, raw_dir=tmp / "raw", now=now)
    # Finished rounds up to Monaco (the fixture calendar has data only for Monaco); later
    # rounds are skipped as not yet run.
    assert 9523 in ingested and len(ingested) == 8
    assert ingest([2024], ["Race"], client=client, raw_dir=tmp / "raw", now=now) == []  # idempotent
    counts = build(raw_dir=tmp / "raw", db_path=tmp / "w.duckdb")
    assert counts["races"] == 8 and counts["laps"] == 234
    con = duckdb.connect(str(tmp / "w.duckdb"), read_only=True)
    yield con
    con.close()


def _row(con, sql):
    cur = con.execute(sql)
    names = [d[0] for d in cur.description]
    return dict(zip(names, cur.fetchone(), strict=True))


def test_laps_are_enriched_with_tyres_and_red_flag(db):
    lap30 = _row(db, "SELECT * FROM laps WHERE driver_number = 4 AND lap_number = 30")
    assert lap30["lap_time"] == 78.403
    assert (lap30["compound"], lap30["stint"], lap30["tyre_age"]) == ("HARD", 2, 29)
    assert lap30["neutralised"] is None and not lap30["pit_in"]
    assert lap30["position"] is not None
    lap1 = _row(db, "SELECT * FROM laps WHERE driver_number = 4 AND lap_number = 1")
    assert lap1["neutralised"] == "RED"


def test_clean_laps_view_excludes_non_representative_laps(db):
    counts = _row(
        db,
        "SELECT count(*) AS n, min(lap_number) AS first, count(DISTINCT compound) AS compounds "
        "FROM clean_laps WHERE driver_number = 4",
    )
    assert counts["first"] > 1 and counts["compounds"] == 1  # lap 1 (red flag) gone; all HARD
    assert counts["n"] < 78


def test_one_row_per_driver_lap_even_with_overlapping_stints(db):
    dupes = db.execute(
        "SELECT count(*) FROM (SELECT session_key, driver_number, lap_number, count(*) AS c "
        "FROM laps GROUP BY ALL HAVING c > 1)"
    ).fetchone()[0]
    assert dupes == 0
    lap1 = _row(db, "SELECT compound, stint FROM laps WHERE driver_number = 16 AND lap_number = 1")
    assert lap1 == {"compound": "MEDIUM", "stint": 1}  # the red-flag tyre change came after
