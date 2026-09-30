"""OpenF1 API clients.

`OpenF1Client` holds all query logic; subclasses only supply `_fetch`, which returns the raw
JSON rows for an endpoint filtered by exact-match params. `HttpOpenF1Client` hits the real API
with a disk cache; `MockOpenF1Client` serves recorded fixtures so tests never touch the network.
"""

import hashlib
import json
import re
import time
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import httpx

from src.tools.models import (
    Driver,
    Interval,
    Lap,
    PitStop,
    Position,
    RaceControlMessage,
    Session,
    SessionResult,
    Stint,
)

BASE_URL = "https://api.openf1.org/v1"
DEFAULT_CACHE_DIR = Path("data/cache/openf1")

Params = dict[str, str | int]


class OpenF1Error(RuntimeError):
    pass


def _normalize(name: str) -> str:
    """Case- and accent-insensitive form, so "montreal" matches "Montréal"."""
    decomposed = unicodedata.normalize("NFKD", name.strip().casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


class OpenF1Client(ABC):
    @abstractmethod
    def _fetch(self, endpoint: str, params: Params) -> list[dict[str, Any]]: ...

    def get_session(self, year: int, place: str, session_name: str = "Race") -> Session:
        """Find a session by country, location or circuit name (case-insensitive).

        Several countries host more than one race a season (2026: Spain has Barcelona and
        Madrid), so an ambiguous country raises instead of guessing.
        """
        rows = self._fetch("sessions", {"year": year, "session_name": session_name})
        sessions = [Session.model_validate(r) for r in rows]
        needle = _normalize(place)
        names = {
            s.session_key: [
                _normalize(n) for n in (s.country_name, s.location, s.circuit_short_name)
            ]
            for s in sessions
        }
        matches = [s for s in sessions if needle in names[s.session_key]]
        # Short forms: a whole word first ("Spa" -> Spa-Francorchamps, not Spain), then a prefix.
        for short in (
            lambda n: needle in re.split(r"[^a-z0-9]+", n),
            lambda n: n.startswith(needle),
        ):
            if matches or not needle:
                break
            matches = [s for s in sessions if any(short(n) for n in names[s.session_key])]
        if len(matches) == 1:
            return matches[0]
        if matches:
            options = "; ".join(s.label for s in matches)
            raise OpenF1Error(f"{place!r} matches several {year} races, specify one: {options}")
        known = ", ".join(sorted({s.location for s in sessions})) or "none"
        raise OpenF1Error(f"No {year} {session_name} found for {place!r}. Known locations: {known}")

    def get_latest_session(
        self, now: datetime, session_name: str = "Race", settle: timedelta = timedelta(minutes=30)
    ) -> Session:
        """Most recent session that has finished (and whose data is free: OpenF1 releases it
        ~30 minutes after the end). Looks back into the previous season early in the year."""
        for year in (now.year, now.year - 1):
            rows = self._fetch("sessions", {"year": year, "session_name": session_name})
            done = [
                Session.model_validate(r)
                for r in rows
                if r.get("date_end")
                and datetime.fromisoformat(r["date_end"]) + settle <= now
                and not r.get("is_cancelled")
            ]
            if done:
                return max(done, key=lambda s: s.date_start)
        raise OpenF1Error(f"No finished {session_name} found in {now.year - 1}-{now.year}.")

    def get_results(self, session_key: int) -> list[SessionResult]:
        rows = self._fetch("session_result", {"session_key": session_key})
        return sorted(
            (SessionResult.model_validate(r) for r in rows), key=lambda r: r.position or 99
        )

    def get_all_laps(self, session_key: int) -> list[Lap]:
        """Every driver's laps in one request (vs one request per driver)."""
        laps = [Lap.model_validate(r) for r in self._fetch("laps", {"session_key": session_key})]
        return sorted(laps, key=lambda lap: (lap.driver_number, lap.lap_number))

    def get_session_by_key(self, session_key: int) -> Session:
        rows = self._fetch("sessions", {"session_key": session_key})
        if not rows:
            raise OpenF1Error(f"No session with key {session_key}.")
        return Session.model_validate(rows[0])

    def get_drivers(self, session_key: int, team_name: str | None = None) -> list[Driver]:
        params: Params = {"session_key": session_key}
        if team_name:
            params["team_name"] = team_name
        return [Driver.model_validate(r) for r in self._fetch("drivers", params)]

    def get_stints(self, session_key: int, driver_number: int | None = None) -> list[Stint]:
        params: Params = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        stints = [Stint.model_validate(r) for r in self._fetch("stints", params)]
        return sorted(stints, key=lambda s: (s.driver_number, s.stint_number))

    def get_laps(
        self,
        session_key: int,
        driver_number: int,
        lap_start: int | None = None,
        lap_end: int | None = None,
    ) -> list[Lap]:
        # Range filtering is done locally: a driver's full race is < 100 rows and it keeps
        # the fetch cacheable regardless of the requested window.
        rows = self._fetch("laps", {"session_key": session_key, "driver_number": driver_number})
        laps = [Lap.model_validate(r) for r in rows]
        laps = [
            lap
            for lap in laps
            if (lap_start is None or lap.lap_number >= lap_start)
            and (lap_end is None or lap.lap_number <= lap_end)
        ]
        return sorted(laps, key=lambda lap: lap.lap_number)

    def get_race_control(
        self,
        session_key: int,
        lap_start: int | None = None,
        lap_end: int | None = None,
        category: str | None = None,
    ) -> list[RaceControlMessage]:
        rows = self._fetch("race_control", {"session_key": session_key})
        msgs = [RaceControlMessage.model_validate(r) for r in rows]
        return [
            m
            for m in msgs
            if (category is None or m.category == category)
            and (lap_start is None or (m.lap_number is not None and m.lap_number >= lap_start))
            and (lap_end is None or (m.lap_number is not None and m.lap_number <= lap_end))
        ]

    # Time-series endpoints. `until`/`since` are ISO timestamps; OpenF1 filters them
    # server-side via `date<`/`date>` params, so we never download a whole race of intervals.

    def get_positions(self, session_key: int, until: str | None = None) -> list[Position]:
        params: Params = {"session_key": session_key}
        if until:
            params["date<"] = until
        return [Position.model_validate(r) for r in self._fetch("position", params)]

    def get_intervals(self, session_key: int, since: str, until: str) -> list[Interval]:
        params: Params = {"session_key": session_key, "date>": since, "date<": until}
        return [Interval.model_validate(r) for r in self._fetch("intervals", params)]

    def get_pit_stops(self, session_key: int, until: str | None = None) -> list[PitStop]:
        params: Params = {"session_key": session_key}
        if until:
            params["date<"] = until
        return [PitStop.model_validate(r) for r in self._fetch("pit", params)]


class HttpOpenF1Client(OpenF1Client):
    """Real API client with a disk cache that knows when data can still change.

    - Data for a session is cached forever only once the session ended SETTLE_AFTER ago; during
      and just after a race weekend it's refetched, so partial data never gets frozen.
    - Empty results are never cached (a 404 during a live window isn't a permanent answer).
    - Calendar queries (sessions by year) expire after CALENDAR_TTL: races get added/moved.
    """

    SETTLE_AFTER = timedelta(hours=6)
    CALENDAR_TTL = timedelta(hours=12)

    def __init__(
        self,
        cache_dir: Path | None = DEFAULT_CACHE_DIR,
        timeout: float = 30.0,
        max_retries: int = 4,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._http = httpx.Client(base_url=BASE_URL, timeout=timeout)
        self._cache_dir = cache_dir
        self._max_retries = max_retries
        self._now = now
        self._session_ends: dict[Any, str | None] = {}  # avoids a sessions request per call

    def _cache_path(self, endpoint: str, params: Params) -> Path | None:
        if self._cache_dir is None:
            return None
        key = json.dumps([endpoint, sorted(params.items())], default=str)
        return self._cache_dir / f"{endpoint}-{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"

    def _read_cache(self, path: Path | None) -> list[dict[str, Any]] | None:
        if not path or not path.exists():
            return None
        entry = json.loads(path.read_text())
        if isinstance(entry, list):  # legacy format: cached forever
            return entry
        expires = entry["expires_at"]
        if expires and datetime.fromisoformat(expires) <= self._now():
            return None
        return entry["rows"]

    def _expiry(
        self, endpoint: str, params: Params, rows: list[dict[str, Any]]
    ) -> datetime | None | Literal["never-cache"]:
        """When a response stops being trustworthy: None = never expires."""
        if not rows:
            return "never-cache"
        now = self._now()
        if endpoint == "sessions" and "session_key" not in params:
            return now + self.CALENDAR_TTL
        session_key = params.get("session_key")
        if session_key is None:
            return now + self.CALENDAR_TTL
        if endpoint == "sessions":
            self._session_ends[session_key] = rows[0].get("date_end")
        elif session_key not in self._session_ends:
            session_rows = self._fetch("sessions", {"session_key": session_key})
            self._session_ends[session_key] = (
                session_rows[0].get("date_end") if session_rows else None
            )
        date_end = self._session_ends[session_key]
        if date_end and datetime.fromisoformat(date_end) + self.SETTLE_AFTER <= now:
            return None
        return now + self.CALENDAR_TTL if endpoint == "sessions" else "never-cache"

    def _fetch(self, endpoint: str, params: Params) -> list[dict[str, Any]]:
        path = self._cache_path(endpoint, params)
        cached = self._read_cache(path)
        if cached is not None:
            return cached

        for attempt in range(self._max_retries + 1):
            resp = self._http.get(f"/{endpoint}", params=params)
            if resp.status_code == 429 and attempt < self._max_retries:
                # Free tier: 3 req/s and 30 req/min, so back off in seconds, not milliseconds.
                retry_after = resp.headers.get("retry-after", "")
                time.sleep(float(retry_after) if retry_after.isdigit() else 2 ** (attempt + 2))
                continue
            # OpenF1 returns 404 with a JSON detail when a filter matches nothing.
            if resp.status_code == 404:
                rows: list[dict[str, Any]] = []
                break
            if resp.status_code in (401, 403):
                # Live data (30 min before to 30 min after a session) needs a paid account.
                raise OpenF1Error(
                    f"OpenF1 refused {endpoint} (HTTP {resp.status_code}). If a session is live, "
                    "its data is paid-only until about 30 minutes after it ends."
                )
            if resp.is_error:
                raise OpenF1Error(f"OpenF1 {endpoint} {params} -> HTTP {resp.status_code}")
            rows = resp.json()
            break

        expiry = self._expiry(endpoint, params, rows)
        if path and expiry != "never-cache":
            path.parent.mkdir(parents=True, exist_ok=True)
            entry = {"expires_at": expiry.isoformat() if expiry else None, "rows": rows}
            path.write_text(json.dumps(entry))
        return rows


class MockOpenF1Client(OpenF1Client):
    """Serves rows from `<fixtures_dir>/<endpoint>.json`, filtered by exact-match params."""

    def __init__(self, fixtures_dir: Path) -> None:
        self._fixtures_dir = fixtures_dir
        self.calls: list[tuple[str, Params]] = []

    def _fetch(self, endpoint: str, params: Params) -> list[dict[str, Any]]:
        self.calls.append((endpoint, params))
        path = self._fixtures_dir / f"{endpoint}.json"
        if not path.exists():
            return []
        rows = json.loads(path.read_text())
        return [r for r in rows if all(_matches(r, k, v) for k, v in params.items())]


def _matches(row: dict[str, Any], key: str, value: str | int) -> bool:
    """Exact match, or OpenF1's `field<` / `field>` range filters (the mock's version)."""
    if key.endswith("<"):
        return row.get(key[:-1], "") < value
    if key.endswith(">"):
        return row.get(key[:-1], "") > value
    return row.get(key) == value
