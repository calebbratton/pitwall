"""Locked-in race predictions and their scorecard.

Every race weekend gets two predictions, saved the moment they're made and committed to git so
they can't be quietly revised (user standard, 2026-10-01):

  pre-Q3   as Q3 starts: practice, Q1/Q2 times and season form; the grid isn't final yet
  grid     when the race feed publishes the official grid (penalties applied), ~1 h before

predictions/<year>/<location>-<stage>.json holds the table (win / podium / points chances per
driver), the notes and the model settings. `python -m src.sim.scorecard` scores them after the
race against the luck-adjusted result (SC/VSC timing and DNFs removed) and the raw result.
"""

import json
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PREDICTIONS = Path("predictions")
STAGES = ("pre-Q3", "grid")


def _slug(location: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", location.casefold()).strip("-")


def path_for(year: int, location: str, stage: str) -> Path:
    return PREDICTIONS / str(year) / f"{_slug(location)}-{stage}.json"


def save(year: int, location: str, stage: str, prediction: dict[str, Any]) -> Path | None:
    """Write once: a stage already recorded for this weekend is never overwritten."""
    from src.sim.calibrate import DEFAULT_CALIBRATION
    from src.sim.race import SimParams

    path = path_for(year, location, stage)
    if path.exists():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "year": year,
        "location": location,
        "stage": stage,
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": {"sim": asdict(SimParams()), "calibration": asdict(DEFAULT_CALIBRATION)},
        "table": prediction.get("table", []),
        "notes": prediction.get("notes", []),
    }
    path.write_text(json.dumps(record, indent=1))
    return path


def load_all(year: int) -> list[dict[str, Any]]:
    folder = PREDICTIONS / str(year)
    return (
        [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
        if folder.exists()
        else []
    )
