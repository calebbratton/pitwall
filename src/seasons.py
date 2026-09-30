"""Which seasons Pit Wall covers for users: the current season and the one before it.

A rolling window, so it moves forward on its own each January. Test fixtures may use older races
(e.g. Monaco/Miami 2024); code paths that serve users should check `supported_seasons()`.
"""

from datetime import UTC, date, datetime


def supported_seasons(today: date | None = None) -> list[int]:
    year = (today or datetime.now(UTC).date()).year
    return [year, year - 1]


def out_of_scope_message(year: int, today: date | None = None) -> str:
    current, previous = supported_seasons(today)
    return f"Pit Wall covers the {current} and {previous} seasons; {year} isn't available."
