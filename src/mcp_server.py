"""Pit Wall as an MCP server: race data and strategy models as tools, the client's LLM reasons.

General building blocks rather than question-shaped tools: results, laps, stints, pit stops,
positions, gaps, race control, weather, regulations and stewards' rulings, plus the models an LLM
can't reproduce by reading tables (pit-stop reviews and what-ifs, who wins from here, pre-race
prediction, luck-adjusted results, tyre life). No LLM calls inside the tools.

Every result is compact JSON with a `source`; anything that isn't known is reported as unknown
(`null` or an `ERROR: ...` string with the valid options), never filled in.

Run:
  python -m src.mcp_server                       # stdio (Claude Code / Claude Desktop)
  uvicorn src.api:app --port 8000                # streamable HTTP at http://localhost:8000/mcp
"""

import json
import re
import unicodedata
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from src.seasons import out_of_scope_message, supported_seasons
from src.tools.models import Driver, Lap, Session
from src.tools.openf1 import HttpOpenF1Client, OpenF1Client, OpenF1Error
from src.tools.telemetry import (
    _overlaps,
    build_telemetry_tools,
    key_race_events,
    neutralised_windows,
    race_incidents,
    race_summary,
    representative_laps,
)

OPENF1 = "OpenF1 (api.openf1.org)"
ARCHIVE = "F1 live-timing archive (livetiming.formula1.com)"
WAREHOUSE = "Pit Wall warehouse (DuckDB, built from OpenF1)"
MAX_ROWS = 400  # per result; narrow the lap range beyond that

# Broadcast nicknames that aren't part of a driver's name. First names, surnames, acronyms and
# numbers are matched from the feed itself.
NICKNAMES = {"checo": "PER", "lewis hamilton": "HAM"}

INSTRUCTIONS = """\
Pit Wall: Formula 1 race data and strategy models for the current and previous season.
- Take every number from a tool result; if a tool says unknown or returns ERROR, say so rather
  than estimating. Check a question's premise against the data (who pitted when, who finished).
- Name races by year + place ("2026", "Madrid"); an ambiguous place returns the options.
  Use find_driver for first names and nicknames ("Kimi", "Checo").
- Strategy questions: review_pit_stop simulates the alternatives with traffic; pace and stints
  alone don't. Cite regulation articles from search_regulations and rulings from
  stewards_decisions by their ids.
- Luck principle: safety car / VSC / red flag timing is luck. luck_adjusted_result removes it;
  predictions leave it out."""


def _json(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"), default=str)


def _fold(s: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(s).strip().casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _error(msg: str) -> str:
    return f"ERROR: {msg}"


class PitwallTools:
    """The tool implementations, plain methods so they test without an MCP transport."""

    def __init__(
        self,
        client: OpenF1Client,
        seasons: list[int] | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        regulations: Callable[[], Any] | None = None,
        decisions: Callable[[], Any] | None = None,
    ):
        self.client = client
        self._seasons = seasons
        self._now = now
        self._make_regulations = regulations
        self._make_decisions = decisions
        self._regulations = None
        self._decisions = None
        self._telemetry: dict[int, dict[str, Any]] = {}

    # --- resolution -------------------------------------------------------------------------

    def seasons(self) -> list[int]:
        return self._seasons or supported_seasons(self._now().date())

    def _race(self, year: int, race: str, allow_future: bool = False) -> Session | str:
        if year not in self.seasons():
            return _error(out_of_scope_message(year, self._now().date()))
        try:
            if _fold(race) in ("latest", "last", "last race", "most recent"):
                session = self.client.get_latest_session(self._now())
            else:
                session = self.client.get_session(year, race)
        except OpenF1Error as e:
            return _error(str(e))
        if not allow_future and datetime.fromisoformat(session.date_start) > self._now():
            return _error(f"{session.label} hasn't happened yet, so there is no race data.")
        return session

    def _tool(self, session: Session, name: str) -> Any:
        if session.session_key not in self._telemetry:
            tools = build_telemetry_tools(self.client, session)
            self._telemetry[session.session_key] = {t.name: t for t in tools}
        return self._telemetry[session.session_key][name]

    def _drivers(self, session: Session) -> list[Driver]:
        return self.client.get_drivers(session.session_key)

    def _match(self, drivers: list[Driver], name: str) -> list[Driver]:
        key = _fold(name).lstrip("#")
        alias = NICKNAMES.get(key)
        exact, partial = [], []
        for d in drivers:
            given = d.first_name or " ".join(d.full_name.split()[:-1])
            names = {
                str(d.driver_number),
                d.name_acronym,
                d.full_name,
                given,
                *given.split(),
                d.last_name or d.full_name.split()[-1],
            }
            folded = {_fold(n) for n in names if n}
            if key in folded or (alias and d.name_acronym == alias):
                exact.append(d)
            elif len(key) >= 3 and any(n.startswith(key) for n in folded):
                partial.append(d)
        return exact or partial

    def _driver(self, session: Session, name: str) -> Driver | str:
        drivers = self._drivers(session)
        if not drivers:
            return _error(f"no entry list published for {session.label} yet; name another race")
        found = self._match(drivers, name)
        if len(found) == 1:
            return found[0]
        options = ", ".join(f"{d.name_acronym} ({d.full_name}, #{d.driver_number})" for d in found)
        if found:
            return _error(f"{name!r} matches several drivers: {options}")
        valid = ", ".join(f"{d.name_acronym} ({d.full_name})" for d in drivers)
        return _error(f"unknown driver {name!r} in {session.label}. Drivers: {valid}")

    def _laps(self, session: Session) -> list[Lap]:
        return self.client.get_all_laps(session.session_key)

    def _windows(self, session: Session):
        return neutralised_windows(
            self.client.get_race_control(session.session_key), self._laps(session)
        )

    # --- data tools -------------------------------------------------------------------------

    def list_races(self, year: int) -> str:
        """Every Grand Prix of a season with its date and whether it has run yet."""
        if year not in self.seasons():
            return _error(out_of_scope_message(year, self._now().date()))
        rows = self.client._fetch("sessions", {"year": year, "session_name": "Race"})
        sessions = sorted((Session.model_validate(r) for r in rows), key=lambda s: s.date_start)
        now = self._now()
        return _json(
            {
                "source": OPENF1,
                "races": [
                    {
                        "location": s.location,
                        "country": s.country_name,
                        "circuit": s.circuit_short_name,
                        "date": s.date_start[:10],
                        "finished": datetime.fromisoformat(s.date_start) <= now,
                    }
                    for s in sessions
                ],
            }
        )

    def find_driver(self, year: int, name: str, race: str | None = None) -> str:
        """Find a driver by number, acronym, first name, surname, full name or nickname
        ("Kimi", "Checo", "Lando", "44"). Uses the given race's entry list, else the season's
        latest race. Returns every match when the name is ambiguous."""
        session = self._race(year, race or "latest")
        if isinstance(session, str):
            return session
        found = self._match(self._drivers(session), name)
        if not found:
            return self._driver(session, name)
        return _json(
            {
                "source": OPENF1,
                "entry_list_of": session.label,
                "matches": [
                    {
                        "acronym": d.name_acronym,
                        "name": d.full_name,
                        "number": d.driver_number,
                        "team": d.team_name,
                    }
                    for d in found
                ],
            }
        )

    def race_results(self, year: int, race: str) -> str:
        """Final classification: position, driver, team, grid slot, status (finished / DNF /
        DNS / DSQ), laps completed, gap to the winner (seconds, or "+N LAP") and points."""
        session = self._race(year, race)
        if isinstance(session, str):
            return session
        sk = session.session_key
        drivers = {d.driver_number: d for d in self._drivers(session)}
        grid: dict[int, int] = {}
        for p in sorted(self.client.get_positions(sk), key=lambda p: p.date):
            grid.setdefault(p.driver_number, p.position)
        rows = []
        for r in self.client.get_results(sk):
            d = drivers.get(r.driver_number)
            status = "DSQ" if r.dsq else "DNS" if r.dns else "DNF" if r.dnf else "finished"
            rows.append(
                {
                    "position": r.position,
                    "driver": d.name_acronym if d else r.driver_number,
                    "team": d.team_name if d else None,
                    "grid": grid.get(r.driver_number),
                    "status": status,
                    "laps": r.number_of_laps,
                    "gap_to_winner": r.gap_to_leader,
                    "points": r.points,
                }
            )
        if not rows:
            return _error(f"no classification published for {session.label} yet")
        return _json(
            {
                "source": OPENF1,
                "race": session.label,
                "notes": "grid = first position reading (includes penalties); gap null = "
                "not published",
                "results": rows,
            }
        )

    def race_summary(self, year: int, race: str) -> str:
        """The whole field in one table (strategy, stints, pace per stint, pit-lane time) plus
        the race's key events (safety cars, VSCs, red flags). A good first call for any
        question about a race."""
        session = self._race(year, race)
        if isinstance(session, str):
            return session
        return _json(
            {
                "source": OPENF1,
                "race": session.label,
                "key_events": json.loads(key_race_events(self.client, session.session_key)),
                "field": json.loads(race_summary(self.client, session)),
            }
        )

    def lap_times(self, year: int, race: str, driver: str, lap_start: int, lap_end: int) -> str:
        """One driver's lap times over a lap range (at most 20 laps; longer ranges return a pace
        summary). Rows are [lap, seconds, pit-out lap]."""
        session = self._race(year, race)
        if isinstance(session, str):
            return session
        d = self._driver(session, driver)
        if isinstance(d, str):
            return d
        out = self._tool(session, "get_lap_times").invoke(
            {"driver": str(d.driver_number), "lap_start": lap_start, "lap_end": lap_end}
        )
        return _json({"source": OPENF1, "driver": d.name_acronym, "laps": json.loads(out)})

    def pace_summary(self, year: int, race: str, driver: str, lap_start: int, lap_end: int) -> str:
        """Mean / median / best lap and lap-time trend for one driver over a lap range,
        excluding lap 1, pit laps, safety car / VSC / red flag laps and outliers."""
        session = self._race(year, race)
        if isinstance(session, str):
            return session
        d = self._driver(session, driver)
        if isinstance(d, str):
            return d
        out = self._tool(session, "get_pace_summary").invoke(
            {"driver": str(d.driver_number), "lap_start": lap_start, "lap_end": lap_end}
        )
        return _json({"source": OPENF1} | json.loads(out))

    def tyre_stints(self, year: int, race: str, driver: str | None = None) -> str:
        """Tyre stints (compound, laps, tyre age at the start) for one driver or all, with each
        stint's own measured wear: fuel-corrected seconds lost per lap of tyre age, from its
        clean laps (null when the stint has too few clean laps to fit)."""
        from src.models.tyres import CleanLap, stint_slopes

        session = self._race(year, race)
        if isinstance(session, str):
            return session
        number = None
        if driver:
            d = self._driver(session, driver)
            if isinstance(d, str):
                return d
            number = d.driver_number
        sk = session.session_key
        stints = self.client.get_stints(sk)
        acronyms = {d.driver_number: d.name_acronym for d in self._drivers(session)}
        clean = representative_laps(self._laps(session), self._windows(session))
        rows: list[CleanLap] = []
        for lap in clean:
            st = next(
                (
                    s
                    for s in stints
                    if s.driver_number == lap.driver_number
                    and s.lap_start is not None
                    and s.lap_end is not None
                    and s.lap_start <= lap.lap_number <= s.lap_end
                ),
                None,
            )
            if st and st.compound and lap.lap_duration:
                age = (st.tyre_age_at_start or 0) + lap.lap_number - st.lap_start
                rows.append(
                    CleanLap(
                        str(lap.driver_number),
                        st.stint_number,
                        st.compound,
                        lap.lap_number,
                        age,
                        lap.lap_duration,
                    )
                )
        wear = stint_slopes(rows)
        return _json(
            {
                "source": OPENF1,
                "race": session.label,
                "wear_note": "fuel-corrected (0.04 s/lap assumed fuel gain), from clean laps",
                "stints": [
                    {
                        "driver": acronyms.get(s.driver_number, s.driver_number),
                        "stint": s.stint_number,
                        "compound": s.compound,
                        "laps": [s.lap_start, s.lap_end],
                        "tyre_age_at_start": s.tyre_age_at_start,
                        "wear_s_per_lap": (
                            round(w, 3)
                            if (w := wear.get((str(s.driver_number), s.stint_number))) is not None
                            else None
                        ),
                    }
                    for s in stints
                    if number is None or s.driver_number == number
                ],
            }
        )

    def pit_stops(self, year: int, race: str, driver: str | None = None) -> str:
        """Every pit stop: driver, in-lap, pit-lane time (entry to exit), stationary time (null
        where not published) and whether a safety car / VSC / red flag was out that lap."""
        session = self._race(year, race)
        if isinstance(session, str):
            return session
        number = None
        if driver:
            d = self._driver(session, driver)
            if isinstance(d, str):
                return d
            number = d.driver_number
        acronyms = {d.driver_number: d.name_acronym for d in self._drivers(session)}
        laps = {(lap.driver_number, lap.lap_number): lap for lap in self._laps(session)}
        windows = self._windows(session)
        rows = []
        for p in sorted(self.client.get_pit_stops(session.session_key), key=lambda p: p.date):
            if number is not None and p.driver_number != number:
                continue
            lap = laps.get((p.driver_number, p.lap_number))
            status = (
                [
                    k
                    for k in ("SC", "VSC", "RED")
                    if _overlaps(lap, [w for w in windows if w[2] == k])
                ]
                if lap
                else None
            )
            rows.append(
                {
                    "driver": acronyms.get(p.driver_number, p.driver_number),
                    "lap": p.lap_number,
                    "pit_lane_s": p.lane_duration,
                    "stationary_s": p.stop_duration,
                    "neutralised": status,
                }
            )
        return _json(
            {
                "source": OPENF1,
                "race": session.label,
                "notes": "pit_lane_s over ~60 s is usually a red-flag or garage stop, not a "
                "racing stop; neutralised null = lap times missing",
                "stops": rows,
            }
        )

    def _crossings(self, session: Session) -> dict[int, dict[int, datetime]]:
        """driver -> {lap: time they crossed the line to complete it}."""
        out: dict[int, dict[int, datetime]] = {}
        for lap in self._laps(session):
            if lap.date_start and lap.lap_duration:
                start = datetime.fromisoformat(lap.date_start)
                out.setdefault(lap.driver_number, {})[lap.lap_number] = start + _seconds(
                    lap.lap_duration
                )
        return out

    def positions_by_lap(
        self,
        year: int,
        race: str,
        lap_start: int,
        lap_end: int,
        drivers: list[str] | None = None,
    ) -> str:
        """Running order at the end of each lap (from timing: the order cars crossed the line
        on that lap; lapped cars count behind), for all drivers or a list."""
        session = self._race(year, race)
        if isinstance(session, str):
            return session
        wanted = None
        if drivers:
            wanted = set()
            for name in drivers:
                d = self._driver(session, name)
                if isinstance(d, str):
                    return d
                wanted.add(d.driver_number)
        acronyms = {d.driver_number: d.name_acronym for d in self._drivers(session)}
        crossings = self._crossings(session)
        rows = {}
        for n in range(lap_start, lap_end + 1):
            done = sorted(
                ((t[n], car) for car, t in crossings.items() if n in t), key=lambda x: x[0]
            )
            order = [car for _, car in done]
            # Cars that never completed lap n are behind everyone who did, in the order they
            # completed their last lap (latest last lap first).
            behind = sorted(
                (car for car, t in crossings.items() if n not in t and t),
                key=lambda car: -max(t for t in crossings[car]),
            )
            order += behind
            rows[n] = {
                acronyms.get(car, car): pos
                for pos, car in enumerate(order, 1)
                if wanted is None or car in wanted
            }
        if sum(len(r) for r in rows.values()) > MAX_ROWS:
            return _error(f"more than {MAX_ROWS} entries; narrow the lap range or drivers")
        return _json({"source": OPENF1, "race": session.label, "positions": rows})

    def gap_between(
        self, year: int, race: str, driver: str, other: str, lap_start: int, lap_end: int
    ) -> str:
        """Time gap between two drivers at the end of each lap (seconds; positive = `driver`
        behind `other`), from when each crossed the line on the same lap."""
        session = self._race(year, race)
        if isinstance(session, str):
            return session
        a, b = self._driver(session, driver), self._driver(session, other)
        for d in (a, b):
            if isinstance(d, str):
                return d
        crossings = self._crossings(session)
        ta, tb = crossings.get(a.driver_number, {}), crossings.get(b.driver_number, {})
        gaps = {
            n: (round((ta[n] - tb[n]).total_seconds(), 3) if n in ta and n in tb else None)
            for n in range(lap_start, lap_end + 1)
        }
        if len(gaps) > MAX_ROWS:
            return _error(f"more than {MAX_ROWS} laps; narrow the range")
        return _json(
            {
                "source": OPENF1,
                "race": session.label,
                "driver": a.name_acronym,
                "other": b.name_acronym,
                "notes": "null = one of them has no timed lap there (pit, retirement, missing data)",
                "gap_s": gaps,
            }
        )

    def race_control(
        self,
        year: int,
        race: str,
        lap_start: int | None = None,
        lap_end: int | None = None,
        category: str | None = None,
    ) -> str:
        """Race control messages: flags, safety car, VSC, red flags, investigations, penalties.
        Optional lap range and category ("Flag", "SafetyCar", "Other")."""
        session = self._race(year, race)
        if isinstance(session, str):
            return session
        out = self._tool(session, "get_race_control").invoke(
            {"lap_start": lap_start, "lap_end": lap_end, "category": category}
        )
        return _json({"source": OPENF1, "race": session.label, "messages": json.loads(out)})

    def race_incidents(self, year: int, race: str) -> str:
        """Stewarding record of a race: per-driver track-limits deletions (lap, turn),
        black-and-white flags, and every incident noted / investigated / penalised."""
        session = self._race(year, race)
        if isinstance(session, str):
            return session
        return _json(
            {
                "source": OPENF1,
                "race": session.label,
                "notes": "the track-limits threshold for a penalty is set in the Race "
                "Director's Event Notes, which this server doesn't have",
                "incidents": json.loads(race_incidents(self.client, session.session_key)),
            }
        )

    def weather(self, year: int, race: str) -> str:
        """Weather during the race: air and track temperature range, humidity, wind, and when
        rain was reported (with the leader's lap at the time)."""
        session = self._race(year, race)
        if isinstance(session, str):
            return session
        rows = self.client._fetch("weather", {"session_key": session.session_key})
        if not rows:
            return _error(f"no weather data published for {session.label}")
        rows.sort(key=lambda r: r["date"])
        start = datetime.fromisoformat(session.date_start)
        end = datetime.fromisoformat(session.date_end) if session.date_end else None
        rows = [
            r
            for r in rows
            if datetime.fromisoformat(r["date"]) >= start
            and (end is None or datetime.fromisoformat(r["date"]) <= end)
        ] or rows

        def span(key: str) -> list[float] | None:
            # 0.0 is a sensor dropout in the temperature feeds, not a reading.
            values = [r[key] for r in rows if r.get(key) is not None]
            if key.endswith("temperature"):
                values = [v for v in values if v != 0]
            return [min(values), max(values)] if values else None

        leader_laps = sorted(
            (datetime.fromisoformat(lap.date_start), lap.lap_number)
            for lap in self._laps(session)
            if lap.date_start
        )

        def lap_at(stamp: str) -> int | None:
            t = datetime.fromisoformat(stamp)
            before = [n for s, n in leader_laps if s <= t]
            return max(before) if before else None

        rain = [r for r in rows if r.get("rainfall")]
        return _json(
            {
                "source": OPENF1,
                "race": session.label,
                "air_temp_c": span("air_temperature"),
                "track_temp_c": span("track_temperature"),
                "humidity_pct": span("humidity"),
                "wind_speed_ms": span("wind_speed"),
                "notes": "rain lap null with rain reported = before the first lap started",
                "rain_reported": bool(rain),
                "rain_from_lap": lap_at(rain[0]["date"]) if rain else None,
                "rain_until_lap": lap_at(rain[-1]["date"]) if rain else None,
            }
        )

    def search_regulations(
        self, query: str, year: int | None = None, race: str | None = None
    ) -> str:
        """Search the FIA Sporting Regulations (issue in force at the race, or the season's
        latest issue). Returns article ids and text to quote."""
        from src.rag.sources import source_for_race

        year = year or self._now().year
        if year not in self.seasons():
            return _error(out_of_scope_message(year, self._now().date()))
        ref = date(year, 12, 31) if year != self._now().year else self._now().date()
        if race:
            session = self._race(year, race)
            if isinstance(session, str):
                return session
            ref = date.fromisoformat(session.date_start[:10])
        try:
            source = source_for_race(year, ref)
        except ValueError as e:
            return _error(str(e))
        if self._regulations is None:
            if self._make_regulations is None:
                return _error("regulation search isn't configured on this server")
            self._regulations = self._make_regulations()
        hits = self._regulations.search(query, source.season, source.issue, k=6)
        return _json(
            {
                "source": f"FIA {source.season} Sporting Regulations, issue {source.issue}",
                "clauses": [
                    {
                        "article": h.chunk.article,
                        "citation": h.chunk.citation,
                        "title": h.chunk.article_title,
                        "text": h.chunk.text,
                    }
                    for h in hits
                ],
            }
        )

    def stewards_decisions(self, query: str, year: int | None = None) -> str:
        """Search the FIA stewards' decisions (penalties, investigations, rulings) for the
        supported seasons, or one season."""
        seasons = [year] if year else self.seasons()
        if year and year not in self.seasons():
            return _error(out_of_scope_message(year, self._now().date()))
        if self._decisions is None:
            if self._make_decisions is None:
                return _error("stewards' decisions aren't configured on this server")
            self._decisions = self._make_decisions()
        found = self._decisions.search(query, seasons, k=5)
        return _json(
            {
                "source": "FIA stewards' decisions (fia.com)",
                "decisions": [_decision(d) for d in found],
            }
        )

    # --- models -----------------------------------------------------------------------------

    def review_pit_stop(
        self, year: int, race: str, driver: str, lap: int | None = None, compound: str | None = None
    ) -> str:
        """Simulate a pit decision with traffic, from the tyre wear and pit loss measured in
        that race (rivals stop when they really did). On a lap the driver pitted: was it right,
        vs staying out 1-5 laps / no more stops. On any other lap: what if they had pitted
        there (e.g. under a SC/VSC), with a time ledger. Without a lap: their stops and their
        laps under SC/VSC/red flag. Takes 20-60 s."""
        session = self._race(year, race)
        if isinstance(session, str):
            return session
        d = self._driver(session, driver)
        if isinstance(d, str):
            return d
        args: dict[str, Any] = {"driver": str(d.driver_number)}
        if lap is not None:
            args["lap"] = lap
        if compound:
            args["compound"] = compound
        out = self._tool(session, "review_pit_stop").invoke(args)
        if out.startswith("ERROR"):
            return out
        return _json({"source": f"Pit Wall simulator on the {ARCHIVE}"} | json.loads(out))

    def who_wins_from_here(self, year: int, race: str, lap: int) -> str:
        """Win / podium chances and expected positions for every car from the race state at
        the end of a lap (the leader's), simulated forward with each car's pace, tyres and
        stops still to make. Safety cars to come are left out (luck)."""
        from src.livetiming.archive import STRATEGY_TOPICS
        from src.livetiming.monitor import RaceMonitor, _circuit_for
        from src.models.tyre_curves import season_curves
        from src.sim.inrace import NotEnoughData, race_state, simulate_from
        from src.sim.pit_review import archive_race

        session = self._race(year, race)
        if isinstance(session, str):
            return session
        try:
            archived = archive_race(year, session.location, session.date_start[:10])
            messages = list(archived.messages(STRATEGY_TOPICS))
        except (LookupError, httpx.HTTPError) as e:
            return _error(
                f"{session.label} isn't available from the live-timing archive ({e}); "
                "this tool needs the race's archived timing"
            )
        circuit = _circuit_for(messages)
        loss = (
            (circuit.pit_loss.green, circuit.pit_loss.safety_car)
            if circuit and circuit.pit_loss
            else (22.0, 13.5)
        )
        monitor = RaceMonitor(pit_loss=circuit.pit_loss if circuit else None)
        reached = False
        for m in messages:
            if m.topic == "LapCount" and int(m.data.get("CurrentLap") or 0) > lap:
                reached = True
                break
            monitor.feed(m)
        if not reached:
            return _error(f"{session.label} has no lap {lap + 1}: pick a lap before the last")
        try:
            state = race_state(
                monitor, monitor.snapshot(), pit_loss=loss, curves=season_curves(year)
            )
            out = simulate_from(state, sims=3000, seed=7)
        except NotEnoughData as e:
            return _error(str(e))
        return _json(
            {
                "source": f"Pit Wall in-race simulator on the {ARCHIVE}",
                "race": session.label,
                "after_lap": out["lap"],
                "laps_remaining": out["laps_remaining"],
                "track_status": out["status"],
                "pit_loss_s": {"green": loss[0], "safety_car": loss[1]},
                "table": out["table"],
                "notes": out["notes"]
                + ["no future safety cars or retirements (luck); calibrated on 2026 races"],
            }
        )

    def predict_race(self, year: int, race: str, laps: int | None = None) -> str:
        """Pre-race prediction from the weekend's qualifying and recent race pace: expected
        position and win / podium / points chances per driver (calibrated; no safety-car or
        retirement luck). Needs qualifying to be in the warehouse; `laps` = race distance when
        the race hasn't run."""
        from src.sim.calibrate import DEFAULT_CALIBRATION, apply
        from src.sim.inputs import build_inputs
        from src.sim.race import simulate
        from src.warehouse.queries import connect

        session = self._race(year, race, allow_future=True)
        if isinstance(session, str):
            return session
        sims = 4000
        try:
            con = connect()
            try:
                inputs = build_inputs(con, year, session.location, laps=laps)
            finally:
                con.close()
            table = apply(simulate(inputs, sims=sims).table(), DEFAULT_CALIBRATION, sims)
        except Exception as e:  # noqa: BLE001 - the reason goes back to the model
            return _error(
                f"no prediction for {session.label}: {type(e).__name__}: {e}. The predictor "
                "needs the weekend's qualifying in the warehouse (and `laps` before race day)."
            )
        grid = {d.tla: d.grid for d in inputs.drivers}
        return _json(
            {
                "source": f"Pit Wall pre-race simulator ({WAREHOUSE})",
                "race": session.label,
                "table": [row | {"grid": grid.get(row["tla"])} for row in table],
                "notes": list(inputs.notes)
                + ["every car finishes, no SC/VSC luck; probabilities calibrated on 2026"],
            }
        )

    def luck_adjusted_result(self, year: int, race: str) -> str:
        """The finishing order with safety car / VSC / red flag luck removed, and each
        driver's seconds gained (+) or lost (-) to neutralisations."""
        from src.sim.inputs import weekend_sessions
        from src.sim.luck import neutralisation_luck
        from src.warehouse.queries import connect

        session = self._race(year, race)
        if isinstance(session, str):
            return session
        pit_loss_green = 22.0  # the scorecard's assumption; the order is insensitive to it
        try:
            con = connect()
            try:
                _, sessions = weekend_sessions(con, year, session.location)
                report = neutralisation_luck(con, sessions["Race"], pit_loss_green)
            finally:
                con.close()
        except (LookupError, KeyError) as e:
            return _error(f"{session.label} isn't in the warehouse yet ({e})")
        tla = {d.driver_number: d.name_acronym for d in self._drivers(session)}
        actual = [
            r.driver_number for r in self.client.get_results(session.session_key) if r.position
        ]
        return _json(
            {
                "source": WAREHOUSE,
                "race": session.label,
                "neutralisations": report.windows,
                "actual_order": [tla.get(n, n) for n in actual],
                "luck_adjusted_order": [tla.get(n, n) for n in report.adjusted_order],
                "luck_s": {tla.get(n, n): round(s, 1) for n, s in report.luck_s.items()},
                "notes": [f"green-flag pit loss assumed {pit_loss_green:.0f} s"],
            }
        )

    def tyre_life(self, year: int) -> str:
        """Season tyre curves by Pirelli compound (C1-C6): seconds lost vs a fresh set by tyre
        age, and the proven life (how far it ran in races without dropping off)."""
        from src.models.tyre_curves import late_slope, season_curves

        if year not in self.seasons():
            return _error(out_of_scope_message(year, self._now().date()))
        curves = season_curves(year)
        if not curves:
            return _error(f"no tyre curves for {year}: the warehouse isn't available")
        return _json(
            {
                "source": WAREHOUSE,
                "season": year,
                "compounds": {
                    c: {
                        "life_laps": curve.life_laps,
                        "drop_off_seen_at": curve.drop_off_age,
                        "late_wear_s_per_lap": (
                            round(s, 3) if (s := late_slope(curve)) is not None else None
                        ),
                        "loss_vs_fresh_s_by_age_band": [
                            [a, round(v, 2)] for a, v, _ in curve.bands
                        ],
                    }
                    for c, curve in sorted(curves.items())
                },
                "notes": "map a race's SOFT/MEDIUM/HARD to C-numbers with the event's nomination",
            }
        )


def _seconds(value: float) -> timedelta:
    return timedelta(seconds=value)


def _decision(d: Any) -> dict:
    """A stewards' decision, cited as "Doc <number>, <competition>"."""
    return {
        "cite_as": f"Doc {d.number}, {d.competition}",
        "title": d.title,
        "driver": d.driver,
        "session": d.session,
        "fact": re.sub(r"\s+", " ", d.fact)[:600],
        "infringement": re.sub(r"\s+", " ", d.infringement)[:400],
        "decision": d.decision,
        "reason": re.sub(r"\s+", " ", d.reason)[:800],
        "url": d.url,
    }


TOOLS = (
    "list_races",
    "find_driver",
    "race_results",
    "race_summary",
    "lap_times",
    "pace_summary",
    "tyre_stints",
    "pit_stops",
    "positions_by_lap",
    "gap_between",
    "race_control",
    "race_incidents",
    "weather",
    "search_regulations",
    "stewards_decisions",
    "review_pit_stop",
    "who_wins_from_here",
    "predict_race",
    "luck_adjusted_result",
    "tyre_life",
)


def build_server(tools: PitwallTools):
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("pitwall", instructions=INSTRUCTIONS)
    for name in TOOLS:
        # Results are JSON text already; a structured copy would double what the client reads.
        server.add_tool(getattr(tools, name), name=name, structured_output=False)

    @server.prompt(name="pitwall_rules", description="How to answer with Pit Wall's tools")
    def pitwall_rules() -> str:
        return INSTRUCTIONS

    return server


def http_app(server):
    """The streamable-HTTP app (stateless, JSON responses: any replica can answer any call).
    Public hostnames go in PITWALL_MCP_HOSTS (comma-separated); without it only localhost is
    accepted, which guards a local server against DNS rebinding."""
    import os

    from mcp.server.transport_security import TransportSecuritySettings

    hosts = [h.strip() for h in os.getenv("PITWALL_MCP_HOSTS", "").split(",") if h.strip()]
    security = None
    if hosts:
        local = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
        security = TransportSecuritySettings(
            allowed_hosts=[*hosts, *local],
            allowed_origins=[f"https://{h}" for h in hosts] + [f"http://{h}" for h in local],
        )
    return server.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        transport_security=security,
    )


def default_tools(regulations: Callable[[], Any] | None = None) -> PitwallTools:
    from src.rag.decisions import DecisionIndex
    from src.rag.index import RegulationIndex

    return PitwallTools(
        HttpOpenF1Client(), regulations=regulations or RegulationIndex, decisions=DecisionIndex
    )


def main() -> None:
    import logging

    logging.getLogger("httpx").setLevel(logging.WARNING)
    build_server(default_tools()).run("stdio")


if __name__ == "__main__":
    main()
