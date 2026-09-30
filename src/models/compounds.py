"""Map the feeds' HARD/MEDIUM/SOFT labels to Pirelli's actual compounds (C1..C5).

The labels are relative to each weekend's nomination: a SOFT at Suzuka (C3) is harder than a
MEDIUM at Monaco (C4). Within one weekend the labels are consistent; anything compared ACROSS
weekends (tyre life, default degradation, season priors) must use the C-number.
"""

import functools
import json
import unicodedata
from functools import cache
from pathlib import Path

NOMINATIONS_DIR = Path(__file__).resolve().parents[2] / "reference"
DRY_LABELS = ("HARD", "MEDIUM", "SOFT")


@cache
def nominations(year: int) -> dict[str, dict[str, str]]:
    path = NOMINATIONS_DIR / f"pirelli_nominations_{year}.json"
    return json.loads(path.read_text())["races"] if path.exists() else {}


def _key(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().casefold()


@functools.cache
def _lookup(year: int) -> tuple[dict, dict]:
    """(normalised race name -> nominations, normalised alias -> race name); read once a year
    (the tyre-curve fit asks for every lap)."""
    path = NOMINATIONS_DIR / f"pirelli_nominations_{year}.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    return (
        {_key(k): v for k, v in data.get("races", {}).items()},
        {_key(k): v for k, v in data.get("aliases", {}).items()},
    )


def c_number(year: int, location: str, label: str | None) -> str | None:
    """'C4' for (2026, 'Baku', 'MEDIUM'); None for wets/inters or unknown races. Location
    names match case/accent-insensitively and through the file's aliases (the live feed and
    OpenF1 don't always name a venue the same way, e.g. Sepang / Kuala Lumpur)."""
    if label not in DRY_LABELS:
        return None
    races, aliases = _lookup(year)
    loc = _key(location or "")
    race = races.get(loc) or races.get(_key(aliases.get(loc, "")))
    return (race or {}).get(label)
