"""A finished race's live-timing messages rebuilt from OpenF1, for when no recording or archive
copy exists (F1's static archive has answered 401 since 2026-10-16).

The pit review engine and the in-race simulator read the live feed's topics through
`RaceMonitor`; this emits the subset they use, at the moments they happened:
  - SessionInfo / DriverList / LapCount header (circuit key and date, so MultiViewer pit loss
    still loads)
  - TimingData per lap completion: laps, position, gap to the leader, interval, last lap time,
    pit-in on the in-lap and pit-out on the out-lap, stops so far, retirements
  - TimingAppData: each stint's compound / new / tyre age, updated every lap
  - TrackStatus from race control (SC / VSC / VSC ending / red flag / green)
  - PitLaneTimeCollection from OpenF1's pit-lane times
Times come from OpenF1's lap start stamps: a lap ends when the next one starts (or start +
duration for a car's last lap), so positions and gaps follow the order cars crossed the line.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from src.livetiming.archive import Message
from src.tools.models import Lap, Session
from src.tools.openf1 import OpenF1Client

# Same instant: track status, lap count, pit times, timing, then tyres - the feed updates a
# stint's tyre age just after the lap's timing line, so a lap's record carries the age it started.
_ORDER = {"TrackStatus": 0, "LapCount": 1, "PitLaneTimeCollection": 2, "TimingAppData": 5}


@dataclass(frozen=True)
class _Event:
    at: datetime
    topic: str
    data: dict
    rank: int = 4


def _lap_time(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}:{rest:06.3f}"


def _stamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _crossings(laps: list[Lap]) -> dict[int, dict[int, datetime]]:
    """car -> {lap: when they crossed the line to complete it}."""
    starts: dict[int, dict[int, Lap]] = {}
    for lap in laps:
        if lap.date_start:
            starts.setdefault(lap.driver_number, {})[lap.lap_number] = lap
    out: dict[int, dict[int, datetime]] = {}
    for car, by_lap in starts.items():
        for n, lap in by_lap.items():
            if n + 1 in by_lap:
                out.setdefault(car, {})[n] = _stamp(by_lap[n + 1].date_start)
            elif lap.lap_duration:
                out.setdefault(car, {})[n] = _stamp(lap.date_start) + timedelta(
                    seconds=lap.lap_duration
                )
    return out


def _official_finish(crossings: dict[int, dict[int, datetime]], results: dict, total: int) -> None:
    """The last lap's crossings from the official gaps to the winner: a final-lap estimate from
    lap start + duration can be off by more than a close finish (Baku 2026: 0.196 s)."""
    winner = next((n for n, r in results.items() if r.position == 1), None)
    if winner is None or total not in crossings.get(winner, {}):
        return
    t_win = crossings[winner][total]
    for car, r in results.items():
        gap = r.gap_to_leader
        if car != winner and isinstance(gap, int | float) and total in crossings.get(car, {}):
            crossings[car][total] = t_win + timedelta(seconds=float(gap))


def _track_status(client: OpenF1Client, session: Session, leader_crossings: list[datetime]):
    """(time, feed status code) from race control: 4 SC, 6 VSC, 7 VSC ending, 5 red, 1 green.
    A safety car "in this lap" goes green when the leader next crosses the line; a red flag
    when the next lap starts after it."""
    out: list[tuple[datetime, str]] = []
    pending_sc_end = False
    for m in sorted(client.get_race_control(session.session_key), key=lambda m: m.date):
        text = m.message.upper().replace("VIRTUAL SAFETY CAR", "VSC")
        when = _stamp(m.date)
        if pending_sc_end:
            # The SC came in at the leader's next crossing after the announcement.
            ends = [t for t in leader_crossings if t > pending_sc_end]
            if ends and ends[0] <= when:
                out.append((ends[0], "1"))
                pending_sc_end = False
        if "VSC DEPLOYED" in text:
            out.append((when, "6"))
        elif "VSC ENDING" in text:
            out.append((when, "7"))
            out.append((when + timedelta(seconds=15), "1"))
        elif "SAFETY CAR DEPLOYED" in text:
            out.append((when, "4"))
            pending_sc_end = False
        elif "SAFETY CAR IN THIS LAP" in text:
            pending_sc_end = when
        elif m.flag == "RED":
            out.append((when, "5"))
            restart = next((t for t in leader_crossings if t > when), None)
            if restart:
                out.append((restart, "1"))
    if pending_sc_end:
        ends = [t for t in leader_crossings if t > pending_sc_end]
        if ends:
            out.append((ends[0], "1"))
    return out


def openf1_messages(client: OpenF1Client, session: Session) -> list[Message]:
    """The race's live-timing messages, rebuilt from OpenF1 (see module docstring)."""
    sk = session.session_key
    drivers = client._fetch("drivers", {"session_key": sk})
    laps = client.get_all_laps(sk)
    stints = sorted(client.get_stints(sk), key=lambda s: (s.driver_number, s.stint_number))
    pits = client.get_pit_stops(sk)
    results = {r.driver_number: r for r in client.get_results(sk)}
    crossings = _crossings(laps)
    if not crossings:
        raise LookupError(f"OpenF1 has no lap timing for {session.label}")
    total = max(
        [r.number_of_laps for r in results.values() if r.number_of_laps]
        or [max(t) for t in crossings.values()]
    )
    _official_finish(crossings, results, total)
    first_start = min(_stamp(lap.date_start) for lap in laps if lap.date_start)
    origin = first_start - timedelta(minutes=1)

    by_lap: dict[int, list[tuple[datetime, int]]] = {}
    for car, times in crossings.items():
        for n, t in times.items():
            by_lap.setdefault(n, []).append((t, car))
    for rows in by_lap.values():
        rows.sort()
    leader_crossings = sorted(rows[0][0] for rows in by_lap.values())

    events: list[_Event] = []

    def emit(at: datetime, topic: str, data: dict) -> None:
        events.append(_Event(at, topic, data, _ORDER.get(topic, 4)))

    # Grid: the first OpenF1 position reading per car.
    grid: dict[int, int] = {}
    for p in sorted(client.get_positions(sk), key=lambda p: p.date):
        grid.setdefault(p.driver_number, p.position)
    cars = sorted({int(d["driver_number"]) for d in drivers} | set(crossings))
    first_stint = {}
    for s in stints:
        first_stint.setdefault(s.driver_number, s)
    emit(
        origin,
        "TimingAppData",
        {
            "Lines": {
                str(car): {
                    "Stints": [
                        {
                            "Compound": s.compound,
                            "New": "true" if (s.tyre_age_at_start or 0) == 0 else "false",
                            "TotalLaps": s.tyre_age_at_start or 0,
                        }
                    ]
                }
                for car, s in first_stint.items()
                if s.compound
            }
        },
    )
    emit(
        origin,
        "TimingData",
        {
            "Lines": {
                str(car): {"Position": str(grid.get(car, i)), "InPit": False}
                for i, car in enumerate(cars, 1)
            }
        },
    )

    for n, rows in by_lap.items():
        leader_t = rows[0][0]
        if n < total:
            emit(leader_t, "LapCount", {"CurrentLap": n + 1})
        for pos, (t, car) in enumerate(rows, 1):
            # Laps down: the leader's crossings between their lap n and this car's.
            laps_down = sum(1 for lt in leader_crossings if leader_t < lt < t)
            if pos == 1:
                gap = f"LAP {min(n + 1, total)}"
            elif laps_down:
                gap = f"{laps_down}L"
            else:
                gap = f"+{(t - leader_t).total_seconds():.3f}"
            interval = "" if pos == 1 else f"+{(t - rows[pos - 2][0]).total_seconds():.3f}"
            line: dict = {
                "NumberOfLaps": n,
                "Position": str(pos),
                "GapToLeader": gap,
                "IntervalToPositionAhead": {"Value": interval},
            }
            emit(t, "TimingData", {"Lines": {str(car): line}})

    lap_by_key = {(lap.driver_number, lap.lap_number): lap for lap in laps}
    for car, times in crossings.items():
        car_stints = [s for s in stints if s.driver_number == car and s.lap_start is not None]
        car_pits = sorted(p.lap_number for p in pits if p.driver_number == car)
        last = max(times)
        for n, t in times.items():
            stint_idx = max(
                (i for i, s in enumerate(car_stints) if s.lap_start <= n),
                default=None,
            )
            if stint_idx is not None:
                s = car_stints[stint_idx]
                age = (s.tyre_age_at_start or 0) + n - s.lap_start + 1
                change = {"TotalLaps": age}
                if stint_idx > 0 and s.lap_start == n and s.compound:
                    # New set fitted: announced as the car leaves the pits (start of the lap).
                    start = times.get(n - 1, t - timedelta(seconds=1))
                    emit(
                        start + timedelta(milliseconds=1),
                        "TimingAppData",
                        {
                            "Lines": {
                                str(car): {
                                    "Stints": {
                                        str(stint_idx): {
                                            "Compound": s.compound,
                                            "New": "true"
                                            if (s.tyre_age_at_start or 0) == 0
                                            else "false",
                                            "TotalLaps": s.tyre_age_at_start or 0,
                                        }
                                    }
                                }
                            }
                        },
                    )
                emit(
                    t,
                    "TimingAppData",
                    {"Lines": {str(car): {"Stints": {str(stint_idx): change}}}},
                )
            lap = lap_by_key.get((car, n))
            in_lap, out_lap = n in car_pits, (n - 1) in car_pits
            line = {
                "InPit": in_lap,
                "PitOut": out_lap,
                "NumberOfPitStops": sum(1 for p in car_pits if p <= n),
            }
            if lap and lap.lap_duration:
                line["LastLapTime"] = {"Value": _lap_time(lap.lap_duration)}
            # Same instant as the lap's NumberOfLaps line: merged before it by the sort below.
            emit(t - timedelta(microseconds=1), "TimingData", {"Lines": {str(car): line}})
            if in_lap:
                emit(t - timedelta(seconds=5), "TimingData", {"Lines": {str(car): {"InPit": True}}})
            if out_lap:
                emit(
                    t + timedelta(microseconds=1),
                    "TimingData",
                    {"Lines": {str(car): {"PitOut": False}}},
                )
        result = results.get(car)
        if result and result.dnf:
            # Retired cars have no gap to the leader (as in the feed).
            emit(
                times[last] + timedelta(seconds=1),
                "TimingData",
                {"Lines": {str(car): {"Retired": True, "GapToLeader": ""}}},
            )
    for car, result in results.items():
        if (result.dnf or result.dns) and car not in crossings:
            # Out before completing a lap: retired from the start.
            emit(
                first_start,
                "TimingData",
                {"Lines": {str(car): {"Retired": True, "GapToLeader": ""}}},
            )

    for p in pits:
        if p.lane_duration:
            emit(
                _stamp(p.date),
                "PitLaneTimeCollection",
                {
                    "PitTimes": {
                        str(p.driver_number): {
                            "RacingNumber": str(p.driver_number),
                            "Duration": str(p.lane_duration),
                            "Lap": str(p.lap_number),
                        }
                    }
                },
            )
    for when, code in _track_status(client, session, leader_crossings):
        emit(when, "TrackStatus", {"Status": code})

    events.sort(key=lambda e: (e.at, e.rank))
    header = [
        (
            "SessionInfo",
            {
                "Meeting": {
                    "Name": session.country_name,
                    "Location": session.location,
                    "Circuit": {
                        "Key": session.circuit_key,
                        "ShortName": session.circuit_short_name,
                    },
                },
                "Type": "Race",
                "Name": session.session_name,
                "StartDate": session.date_start[:19],
                "Source": "OpenF1",
            },
        ),
        (
            "DriverList",
            {
                str(d["driver_number"]): {
                    "Tla": d.get("name_acronym", str(d["driver_number"])),
                    "TeamName": d.get("team_name") or "",
                    "TeamColour": d.get("team_colour") or "",
                    "FullName": d.get("full_name") or "",
                }
                for d in drivers
            },
        ),
        ("LapCount", {"CurrentLap": 1, "TotalLaps": total}),
        ("TrackStatus", {"Status": "1"}),
    ]
    messages = [Message(timedelta(0), topic, data) for topic, data in header]
    messages += [Message(e.at - origin, e.topic, e.data) for e in events]
    return messages
