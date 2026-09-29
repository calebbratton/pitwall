"""OpenF1 API clients.

`OpenF1Client` holds all query logic; subclasses only supply `_fetch`, which returns the raw
JSON rows for an endpoint filtered by exact-match params. `HttpOpenF1Client` hits the real API
with a disk cache; `MockOpenF1Client` serves recorded fixtures so tests never touch the network.
"""

import hashlib
import json
import time
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx

from src.tools.models import Driver, Lap, RaceControlMessage, Session, Stint

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
        matches = [
            s
            for s in sessions
            if needle in {_normalize(n) for n in (s.country_name, s.location, s.circuit_short_name)}
        ]
        if len(matches) == 1:
            return matches[0]
        if matches:
            options = "; ".join(s.label for s in matches)
            raise OpenF1Error(f"{place!r} matches several {year} races, specify one: {options}")
        known = ", ".join(sorted({s.location for s in sessions})) or "none"
        raise OpenF1Error(f"No {year} {session_name} found for {place!r}. Known locations: {known}")

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


class HttpOpenF1Client(OpenF1Client):
    """Real API client. Responses are cached to disk forever: historical data doesn't change."""

    def __init__(
        self,
        cache_dir: Path | None = DEFAULT_CACHE_DIR,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._http = httpx.Client(base_url=BASE_URL, timeout=timeout)
        self._cache_dir = cache_dir
        self._max_retries = max_retries

    def _cache_path(self, endpoint: str, params: Params) -> Path | None:
        if self._cache_dir is None:
            return None
        key = json.dumps([endpoint, sorted(params.items())], default=str)
        return self._cache_dir / f"{endpoint}-{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"

    def _fetch(self, endpoint: str, params: Params) -> list[dict[str, Any]]:
        path = self._cache_path(endpoint, params)
        if path and path.exists():
            return json.loads(path.read_text())

        for attempt in range(self._max_retries + 1):
            resp = self._http.get(f"/{endpoint}", params=params)
            if resp.status_code == 429 and attempt < self._max_retries:
                time.sleep(2**attempt)
                continue
            # OpenF1 returns 404 with a JSON detail when a filter matches nothing.
            if resp.status_code == 404:
                rows: list[dict[str, Any]] = []
                break
            if resp.is_error:
                raise OpenF1Error(f"OpenF1 {endpoint} {params} -> HTTP {resp.status_code}")
            rows = resp.json()
            break

        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows))
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
        return [r for r in rows if all(r.get(k) == v for k, v in params.items())]
