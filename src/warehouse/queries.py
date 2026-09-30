"""Read-side helpers over the warehouse (DuckDB, read-only)."""

from pathlib import Path

import duckdb

from src.models.tyres import CleanLap
from src.warehouse.build import DB_PATH


def connect(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path), read_only=True)


def find_session(con, year: int, place: str, session_name: str = "Race") -> int:
    """session_key by location / country / circuit name (case-insensitive substring)."""
    rows = con.execute(
        """
        SELECT session_key FROM races
        WHERE year = ? AND session_name = ?
          AND (location ILIKE ? OR country_name ILIKE ? OR circuit_short_name ILIKE ?)
        ORDER BY date_start
        """,
        [year, session_name, f"%{place}%", f"%{place}%", f"%{place}%"],
    ).fetchall()
    if len(rows) != 1:
        raise LookupError(f"{len(rows)} warehouse sessions match {place!r} {year} {session_name}")
    return rows[0][0]


def clean_laps(con, session_key: int) -> list[CleanLap]:
    rows = con.execute(
        """
        SELECT coalesce(d.name_acronym, l.driver_number::VARCHAR), l.stint, l.compound,
               l.lap_number, l.tyre_age, l.lap_time
        FROM clean_laps l
        LEFT JOIN raw_drivers d
          ON d.session_key = l.session_key AND d.driver_number = l.driver_number
        WHERE l.session_key = ? AND l.tyre_age IS NOT NULL
        ORDER BY 1, l.lap_number
        """,
        [session_key],
    ).fetchall()
    return [CleanLap(*row) for row in rows]
