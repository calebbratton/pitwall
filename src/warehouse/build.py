"""Build query-ready DuckDB tables from the raw layer.

Rebuilt from scratch every time (seconds): the raw JSON Lines are the source of truth.
The key table is `laps`: one row per driver-lap with everything strategy needs attached.

Usage: python -m src.warehouse.build
"""

import argparse
from pathlib import Path

import duckdb

from src.warehouse.ingest import RAW_DIR

DB_PATH = Path("data/warehouse/pitwall.duckdb")

# Columns the build SQL relies on, used when a raw table has no data yet (e.g. a fixture session
# without weather) so the joins still work.
EMPTY_SCHEMAS = {
    "sessions": "session_key BIGINT, meeting_key BIGINT, year BIGINT, session_name VARCHAR, "
    "location VARCHAR, country_name VARCHAR, circuit_short_name VARCHAR, circuit_key BIGINT, "
    "date_start VARCHAR, date_end VARCHAR",
    "laps": "session_key BIGINT, driver_number BIGINT, lap_number BIGINT, date_start VARCHAR, "
    "lap_duration DOUBLE, duration_sector_1 DOUBLE, duration_sector_2 DOUBLE, "
    "duration_sector_3 DOUBLE, st_speed BIGINT, is_pit_out_lap BOOLEAN",
    "stints": "session_key BIGINT, driver_number BIGINT, stint_number BIGINT, compound VARCHAR, "
    "lap_start BIGINT, lap_end BIGINT, tyre_age_at_start BIGINT",
    "pits": "session_key BIGINT, driver_number BIGINT, lap_number BIGINT, date VARCHAR, "
    "lane_duration DOUBLE",
    "weather": "session_key BIGINT, date VARCHAR, track_temperature DOUBLE, "
    "air_temperature DOUBLE, rainfall DOUBLE, humidity DOUBLE, wind_speed DOUBLE",
    "positions": "session_key BIGINT, driver_number BIGINT, date VARCHAR, position BIGINT",
    "intervals": "session_key BIGINT, driver_number BIGINT, date VARCHAR, interval VARCHAR, "
    "gap_to_leader VARCHAR",
    "neutralised": 'session_key BIGINT, start VARCHAR, "end" VARCHAR, kind VARCHAR',
}

RAW_TABLES = (
    "sessions",
    "drivers",
    "results",
    "laps",
    "stints",
    "pits",
    "race_control",
    "weather",
    "positions",
    "intervals",
    "neutralised",
)

# Enriched laps. ASOF joins attach the latest weather / position / gap reading at the moment a
# lap ends (or starts, for weather). Gaps can be "+1 LAP" strings for lapped cars -> NULL.
LAPS_SQL = """
CREATE OR REPLACE TABLE laps AS
WITH base AS (
    SELECT
        l.session_key, l.driver_number, l.lap_number,
        l.date_start::TIMESTAMPTZ AS lap_start,
        l.date_start::TIMESTAMPTZ + to_microseconds((l.lap_duration * 1e6)::BIGINT) AS lap_end,
        l.lap_duration AS lap_time,
        l.duration_sector_1 AS s1, l.duration_sector_2 AS s2, l.duration_sector_3 AS s3,
        l.st_speed AS speed_trap,
        COALESCE(l.is_pit_out_lap, FALSE) AS pit_out
    FROM raw_laps l
)
SELECT
    b.*,
    s.stint_number AS stint,
    s.compound,
    s.tyre_age_at_start + (b.lap_number - s.lap_start) AS tyre_age,
    EXISTS (SELECT 1 FROM raw_pits p
            WHERE p.session_key = b.session_key AND p.driver_number = b.driver_number
              AND p.lap_number = b.lap_number) AS pit_in,
    (SELECT n.kind FROM raw_neutralised n
      WHERE n.session_key = b.session_key
        -- Untimed laps (e.g. a red-flagged lap) have no end: use their start.
        AND b.lap_start < n."end"::TIMESTAMPTZ
        AND COALESCE(b.lap_end, b.lap_start) >= n.start::TIMESTAMPTZ
      LIMIT 1) AS neutralised,
    w.track_temperature AS track_temp, w.air_temperature AS air_temp, w.rainfall > 0 AS raining,
    pos.position,
    TRY_CAST(i.interval AS DOUBLE) AS gap_ahead,
    TRY_CAST(i.gap_to_leader AS DOUBLE) AS gap_to_leader
FROM base b
LEFT JOIN (
    -- Stints can overlap on a lap (a tyre change during a red flag makes lap 1 belong to two
    -- stints); keep the stint the car started the lap on.
    SELECT l.session_key, l.driver_number, l.lap_number, s.stint_number, s.compound,
           s.tyre_age_at_start, s.lap_start
    FROM raw_laps l
    JOIN raw_stints s
      ON s.session_key = l.session_key AND s.driver_number = l.driver_number
     AND l.lap_number BETWEEN s.lap_start AND s.lap_end
    QUALIFY row_number() OVER (
        PARTITION BY l.session_key, l.driver_number, l.lap_number ORDER BY s.stint_number
    ) = 1
) s
    ON s.session_key = b.session_key AND s.driver_number = b.driver_number
   AND s.lap_number = b.lap_number
ASOF LEFT JOIN (SELECT session_key, date::TIMESTAMPTZ AS t, * EXCLUDE (session_key, date)
                FROM raw_weather) w
    ON w.session_key = b.session_key AND b.lap_start >= w.t
ASOF LEFT JOIN (SELECT session_key, driver_number, date::TIMESTAMPTZ AS t, position
                FROM raw_positions) pos
    ON pos.session_key = b.session_key AND pos.driver_number = b.driver_number
   AND b.lap_end >= pos.t
ASOF LEFT JOIN (SELECT session_key, driver_number, date::TIMESTAMPTZ AS t, interval, gap_to_leader
                FROM raw_intervals) i
    ON i.session_key = b.session_key AND i.driver_number = b.driver_number
   AND b.lap_end >= i.t
"""

# A lap is "clean" for pace/degradation modelling when it's a timed green-flag racing lap.
VIEWS_SQL = """
CREATE OR REPLACE VIEW clean_laps AS
SELECT * FROM laps
WHERE lap_time IS NOT NULL AND lap_number > 1 AND NOT pit_out AND NOT pit_in
  AND neutralised IS NULL AND compound IS NOT NULL;

CREATE OR REPLACE VIEW races AS
SELECT session_key, meeting_key, year, session_name, location, country_name, circuit_short_name,
       circuit_key, date_start::TIMESTAMPTZ AS date_start
FROM raw_sessions;
"""


def build(raw_dir: Path = RAW_DIR, db_path: Path = DB_PATH) -> dict[str, int]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        for table in RAW_TABLES:
            files = sorted((raw_dir / table).glob("*.jsonl"))
            files = [f for f in files if f.stat().st_size > 0]
            if files:
                paths = ", ".join(f"'{f}'" for f in files)
                con.execute(
                    f"CREATE OR REPLACE TABLE raw_{table} AS "
                    f"SELECT * FROM read_json_auto([{paths}], union_by_name = true)"
                )
            else:
                schema = EMPTY_SCHEMAS.get(table, "session_key BIGINT")
                con.execute(f"CREATE OR REPLACE TABLE raw_{table} ({schema})")
        con.execute(LAPS_SQL)
        con.execute(VIEWS_SQL)
        return {
            name: con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            for name in ("races", "laps", "clean_laps", "raw_intervals")
        }
    finally:
        con.close()


def main() -> None:
    argparse.ArgumentParser().parse_args()
    for table, count in build().items():
        print(f"{table:14} {count:>9,} rows")


if __name__ == "__main__":
    main()
