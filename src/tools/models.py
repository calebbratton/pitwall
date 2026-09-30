"""Typed OpenF1 records. Only fields the analysis needs; unknown fields are dropped."""

from pydantic import BaseModel, ConfigDict


class _Record(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class Session(_Record):
    session_key: int
    meeting_key: int
    session_name: str
    session_type: str
    country_name: str
    location: str
    circuit_short_name: str
    year: int
    date_start: str
    date_end: str | None = None

    @property
    def label(self) -> str:
        return f"{self.location} ({self.country_name}, {self.date_start[:10]})"


class Driver(_Record):
    driver_number: int
    name_acronym: str
    full_name: str
    team_name: str | None = None


class Stint(_Record):
    driver_number: int
    stint_number: int
    compound: str | None = None
    lap_start: int | None = None
    lap_end: int | None = None
    tyre_age_at_start: int | None = None


class Lap(_Record):
    driver_number: int
    lap_number: int
    lap_duration: float | None = None
    duration_sector_1: float | None = None
    duration_sector_2: float | None = None
    duration_sector_3: float | None = None
    is_pit_out_lap: bool = False
    st_speed: int | None = None


class RaceControlMessage(_Record):
    date: str
    lap_number: int | None = None
    category: str
    flag: str | None = None
    scope: str | None = None
    driver_number: int | None = None
    message: str


class Position(_Record):
    date: str
    driver_number: int
    position: int


class Interval(_Record):
    date: str
    driver_number: int
    gap_to_leader: float | str | None = None  # "+1 LAP" strings for lapped cars
    interval: float | str | None = None


class PitStop(_Record):
    date: str
    driver_number: int
    lap_number: int
    lane_duration: float | None = None  # pit entry to pit exit, seconds


class SessionResult(_Record):
    driver_number: int
    position: int | None = None
    points: float | None = None
    number_of_laps: int | None = None
    dnf: bool = False
    dns: bool = False
    dsq: bool = False
