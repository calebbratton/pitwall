"""Rain outlook for a race window from Open-Meteo (free, no key).

`race_rain(location, start)` returns the forecast chance of rain over the race (about two
hours from lights out): the hourly precipitation probabilities combined as "rain in at least
one hour" would overstate it for showers that span hours, so the peak hourly probability is
used, alongside the expected millimetres. Forecasts only reach ~16 days ahead.

Coordinates come from reference/circuit_coords.json, else Open-Meteo's geocoder (city-level:
fine for a probability, not for "which corner gets wet first").
"""

import json
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

COORDS = Path("reference/circuit_coords.json")
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
RACE_HOURS = 2


@dataclass(frozen=True)
class RainOutlook:
    location: str
    start: datetime
    p_rain: float  # 0..1, peak hourly precipitation probability over the race window
    mm: float  # forecast precipitation over the window
    hourly: list[tuple[str, int, float]]  # (UTC hour, probability %, mm)
    coords_source: str  # "circuit" | "geocoded"


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return text.casefold().strip()


def coordinates(location: str, client: httpx.Client | None = None) -> tuple[float, float, str]:
    table = {_norm(k): v for k, v in json.loads(COORDS.read_text()).items() if k[0] != "_"}
    if _norm(location) in table:
        lat, lon = table[_norm(location)]
        return lat, lon, "circuit"
    client = client or httpx.Client(timeout=10)
    results = client.get(GEOCODE_URL, params={"name": location, "count": 1}).json().get("results")
    if not results:
        raise ValueError(f"no coordinates for {location!r}")
    return results[0]["latitude"], results[0]["longitude"], "geocoded"


def race_rain(
    location: str, start: datetime, hours: int = RACE_HOURS, client: httpx.Client | None = None
) -> RainOutlook:
    client = client or httpx.Client(timeout=10)
    lat, lon, source = coordinates(location, client)
    start = start.astimezone(UTC)
    data = client.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation_probability,precipitation",
            "timezone": "UTC",
            "start_date": start.date().isoformat(),
            "end_date": (start + timedelta(hours=hours)).date().isoformat(),
        },
    ).json()
    if "hourly" not in data:
        raise ValueError(f"no forecast: {data.get('reason', data)}")
    first = start.replace(minute=0, second=0, microsecond=0)
    window = {(first + timedelta(hours=h)).strftime("%Y-%m-%dT%H:00") for h in range(hours + 1)}
    h = data["hourly"]
    hourly = [
        (t, int(p or 0), float(mm or 0))
        for t, p, mm in zip(
            h["time"], h["precipitation_probability"], h["precipitation"], strict=True
        )
        if t in window
    ]
    if not hourly:
        raise ValueError("race window is outside the forecast range")
    return RainOutlook(
        location=location,
        start=start,
        p_rain=max(p for _, p, _ in hourly) / 100,
        mm=round(sum(mm for _, _, mm in hourly), 1),
        hourly=hourly,
        coords_source=source,
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Rain outlook for a race window")
    ap.add_argument("location")
    ap.add_argument("start", help="race start, ISO UTC, e.g. 2026-10-04T07:00")
    args = ap.parse_args()
    o = race_rain(args.location, datetime.fromisoformat(args.start).replace(tzinfo=UTC))
    print(
        f"{o.location} from {o.start:%Y-%m-%d %H:%M} UTC: {o.p_rain:.0%} chance of rain, {o.mm} mm ({o.coords_source} coordinates)"
    )
    for t, p, mm in o.hourly:
        print(f"  {t}  {p:3d}%  {mm} mm")


if __name__ == "__main__":
    main()
