"""Map the feeds' HARD/MEDIUM/SOFT labels to Pirelli's actual compounds (C1..C5).

The labels are relative to each weekend's nomination: a SOFT at Suzuka (C3) is harder than a
MEDIUM at Monaco (C4). Within one weekend the labels are consistent; anything compared ACROSS
weekends (tyre life, default degradation, season priors) must use the C-number.
"""

import json
from functools import cache
from pathlib import Path

NOMINATIONS_DIR = Path(__file__).resolve().parents[2] / "reference"
DRY_LABELS = ("HARD", "MEDIUM", "SOFT")


@cache
def nominations(year: int) -> dict[str, dict[str, str]]:
    path = NOMINATIONS_DIR / f"pirelli_nominations_{year}.json"
    return json.loads(path.read_text())["races"] if path.exists() else {}


def c_number(year: int, location: str, label: str | None) -> str | None:
    """'C4' for (2026, 'Baku', 'MEDIUM'); None for wets/inters or unknown races."""
    if label not in DRY_LABELS:
        return None
    return nominations(year).get(location, {}).get(label)
