"""A typed, source-independent view of a race at one moment.

Strategy code reads only `RaceSnapshot`, never raw feed JSON, so the same pit-call logic runs on
live data, archive replays, or a snapshot built from another source.
"""

import re
import statistics
from dataclasses import dataclass, field

from src.livetiming.state import TimingState

TRACK_STATUS = {
    "1": "GREEN",
    "2": "YELLOW",
    "4": "SAFETY_CAR",
    "5": "RED_FLAG",
    "6": "VSC",
    "7": "VSC_ENDING",
}
DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}


@dataclass(frozen=True)
class DriverState:
    number: str
    tla: str
    team: str
    position: int | None
    gap_to_leader_s: float | None  # None when lapped or unknown
    laps_down: int
    interval_s: float | None  # to the car ahead
    in_pit: bool
    retired: bool
    laps_completed: int
    pit_stops: int
    compound: str | None
    tyre_new_when_fitted: bool | None
    tyre_age_laps: int | None
    compounds_used: tuple[str, ...]

    @property
    def needs_second_compound(self) -> bool:
        """Hasn't yet used two different dry compounds (the mandatory race requirement,
        waived if intermediates/wets were used)."""
        wet = any(c not in DRY_COMPOUNDS for c in self.compounds_used)
        return not wet and len(set(self.compounds_used) & DRY_COMPOUNDS) < 2


@dataclass(frozen=True)
class RaceSnapshot:
    meeting: str
    session: str
    clock: str
    track_status: str
    current_lap: int | None
    total_laps: int | None
    drivers: tuple[DriverState, ...]  # running order, retired cars last
    pit_lane_times_s: tuple[float, ...] = field(default=())

    @property
    def laps_remaining(self) -> int | None:
        if self.current_lap is None or self.total_laps is None:
            return None
        return self.total_laps - self.current_lap

    @property
    def median_pit_lane_s(self) -> float | None:
        return statistics.median(self.pit_lane_times_s) if self.pit_lane_times_s else None

    def driver(self, key: str) -> DriverState:
        for d in self.drivers:
            if key.upper() in (d.tla, d.number):
                return d
        raise KeyError(key)


_SECONDS = re.compile(r"^\+?(\d+(?:\.\d+)?)$")
_LAPPED = re.compile(r"^\+?(\d+)\s*L(?:APS?)?$", re.IGNORECASE)


def _gap(value: str | None) -> tuple[float | None, int]:
    """Parse "+11.392" -> (11.392, 0); "1L" / "+2 LAPS" -> (None, n); leader's "LAP 28" or ""."""
    text = (value or "").strip()
    if m := _SECONDS.match(text):
        return float(m.group(1)), 0
    if m := _LAPPED.match(text):
        return None, int(m.group(1))
    return None, 0


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_snapshot(state: TimingState) -> RaceSnapshot:
    t = state.topics
    info = t.get("SessionInfo", {})
    driver_list = t.get("DriverList", {})
    timing = t.get("TimingData", {}).get("Lines", {})
    app = t.get("TimingAppData", {}).get("Lines", {})
    laps = t.get("LapCount", {})

    drivers = []
    for number, line in timing.items():
        meta = driver_list.get(number, {})
        stints = [s for s in (app.get(number, {}).get("Stints") or []) if s.get("Compound")]
        current = stints[-1] if stints else {}
        position = _int(line.get("Position"))
        gap, laps_down = _gap(line.get("GapToLeader"))
        if position == 1:
            gap, laps_down = 0.0, 0
        interval, _ = _gap((line.get("IntervalToPositionAhead") or {}).get("Value"))
        drivers.append(
            DriverState(
                number=number,
                tla=meta.get("Tla", number),
                team=meta.get("TeamName", ""),
                position=position,
                gap_to_leader_s=gap,
                laps_down=laps_down,
                interval_s=interval,
                in_pit=bool(line.get("InPit")),
                retired=bool(line.get("Retired") or line.get("Stopped")),
                laps_completed=_int(line.get("NumberOfLaps")) or 0,
                pit_stops=_int(line.get("NumberOfPitStops")) or 0,
                compound=current.get("Compound"),
                tyre_new_when_fitted=(
                    None if "New" not in current else str(current["New"]).lower() == "true"
                ),
                tyre_age_laps=_int(current.get("TotalLaps")),
                compounds_used=tuple(dict.fromkeys(s["Compound"] for s in stints)),
            )
        )
    drivers.sort(key=lambda d: (d.retired, d.position or 99))

    return RaceSnapshot(
        meeting=info.get("Meeting", {}).get("Name", ""),
        session=info.get("Name", ""),
        clock=state.clock,
        track_status=TRACK_STATUS.get(str(t.get("TrackStatus", {}).get("Status")), "UNKNOWN"),
        current_lap=_int(laps.get("CurrentLap")),
        total_laps=_int(laps.get("TotalLaps")),
        drivers=tuple(drivers),
        pit_lane_times_s=tuple(p.duration_s for p in state.pit_lane_times),
    )
