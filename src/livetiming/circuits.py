"""Static circuit data from MultiViewer's public circuit API (the source FastF1 uses too).

https://api.multiviewer.app/api/v1/circuits/<circuit_key>/<year> gives, in the same coordinate
system as the live-timing Position.z feed:
- a detailed outline (x/y), the display rotation, numbered corners and marshal sectors
- measured pit-lane time loss under green, safety car and VSC

It's an unofficial third-party API: responses are cached under data/circuits/ (not committed —
it's their data), requests identify the project, and everything degrades gracefully to the
outline traced from Position.z and the rule-of-thumb pit loss when it's unavailable.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

API = "https://api.multiviewer.app/api/v1/circuits"
DEFAULT_CACHE = Path("data/circuits")
USER_AGENT = "pitwall (personal F1 strategy project)"


@dataclass(frozen=True)
class PitLoss:
    """Seconds lost by a pit stop vs staying out."""

    green: float
    safety_car: float
    vsc: float
    source: str

    def for_status(self, track_status: str) -> float:
        if track_status == "SAFETY_CAR":
            return self.safety_car
        if track_status in ("VSC", "VSC_ENDING"):
            return self.vsc
        return self.green


@dataclass(frozen=True)
class Circuit:
    key: int
    year: int
    name: str
    points: list[list[float]]
    rotation: float
    corners: list[dict[str, Any]] = field(default_factory=list)
    marshal_sectors: list[dict[str, Any]] = field(default_factory=list)
    pit_loss: PitLoss | None = None

    def track_event(self) -> dict[str, Any]:
        xs, ys = [p[0] for p in self.points], [p[1] for p in self.points]
        return {
            "type": "track",
            "source": "multiviewer",
            "name": self.name,
            "points": self.points,
            "bounds": [min(xs), min(ys), max(xs), max(ys)],
            "rotation": self.rotation,
            "corners": self.corners,
            "marshal_sectors": self.marshal_sectors,
        }


def _marker(item: dict[str, Any]) -> dict[str, Any]:
    pos = item.get("trackPosition") or {}
    return {"number": item.get("number"), "x": pos.get("x"), "y": pos.get("y")}


def parse_circuit(raw: dict[str, Any], key: int, year: int) -> Circuit:
    loss = raw.get("pitLoss") or {}
    try:
        pit_loss = PitLoss(
            float(loss["normal"]), float(loss["sc"]), float(loss["vsc"]), "measured (MultiViewer)"
        )
    except (KeyError, TypeError, ValueError):
        pit_loss = None
    return Circuit(
        key=key,
        year=year,
        name=raw.get("circuitName", ""),
        points=[[x, y] for x, y in zip(raw["x"], raw["y"], strict=True)],
        rotation=float(raw.get("rotation") or 0),
        corners=[_marker(c) for c in raw.get("corners") or []],
        marshal_sectors=[_marker(s) for s in raw.get("marshalSectors") or []],
        pit_loss=pit_loss,
    )


def fetch_circuit(key: int, year: int, cache_dir: Path | None = DEFAULT_CACHE) -> Circuit | None:
    """Circuit data for a circuit key (SessionInfo.Meeting.Circuit.Key) and season, or None."""
    cached = cache_dir / f"{key}_{year}.json" if cache_dir else None
    try:
        if cached and cached.exists():
            raw = json.loads(cached.read_text())
        else:
            resp = httpx.get(f"{API}/{key}/{year}", headers={"User-Agent": USER_AGENT}, timeout=10)
            resp.raise_for_status()
            raw = resp.json()
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(json.dumps(raw))
        return parse_circuit(raw, key, year)
    except (httpx.HTTPError, KeyError, ValueError):
        log.warning("circuit %s/%s unavailable; using traced outline", key, year, exc_info=True)
        return None
