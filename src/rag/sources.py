"""FIA Formula One Sporting Regulations issues we index, and picking the one in force for a race.

Each season's regulations are reissued several times. A race must be judged against the issue
in force on race day, e.g. Monaco 2024 (26 May) falls under Issue 6, not the later Issue 7.
"""

from dataclasses import dataclass
from datetime import date
from typing import Literal

_OLD = "https://www.fia.com/sites/default/files"
_NEW = "https://www.fia.com/system/files/documents"


@dataclass(frozen=True)
class RegSource:
    season: int
    issue: int
    published: date
    url: str
    fmt: Literal["classic", "section_b"] = "classic"

    @property
    def filename(self) -> str:
        return f"{self.season}_iss{self.issue:02d}.pdf"


# 2026 first: it's the current season. Issues 1-4 predate the first race and are omitted.
_S26 = "fia_2026_f1_regulations_-_section_b_sporting_-_iss"
_F24 = "fia_2024_formula_1_sporting_regulations_-_issue"
SOURCES: tuple[RegSource, ...] = (
    RegSource(2026, 5, date(2026, 2, 27), f"{_NEW}/{_S26}_05_-_2026-02-27.pdf", "section_b"),
    RegSource(2026, 6, date(2026, 4, 28), f"{_NEW}/{_S26}_06_-_2026-04-28.pdf", "section_b"),
    RegSource(2026, 7, date(2026, 6, 25), f"{_NEW}/{_S26}_07_-_2026-06-25.pdf", "section_b"),
    RegSource(2026, 8, date(2026, 8, 5), f"{_NEW}/{_S26}_08_-_2026-08-05_7.pdf", "section_b"),
    RegSource(
        2025,
        5,
        date(2025, 4, 30),
        f"{_NEW}/fia_2025_formula_1_sporting_regulations_-_issue_5_-_2025-04-30.pdf",
    ),
    RegSource(2024, 5, date(2024, 2, 28), f"{_OLD}/{_F24}_5_-_2024-02-28.pdf"),
    RegSource(2024, 6, date(2024, 4, 30), f"{_OLD}/{_F24}_6_-_2024-04-30_v2.pdf"),
    RegSource(2024, 7, date(2024, 7, 31), f"{_OLD}/{_F24}_7_-_2024-07-31.pdf"),
)


def source_for_race(season: int, race_date: date) -> RegSource:
    """Latest indexed issue published on or before race day; earliest issue if none precede it."""
    issues = sorted((s for s in SOURCES if s.season == season), key=lambda s: s.published)
    if not issues:
        raise ValueError(f"No Sporting Regulations indexed for {season}.")
    in_force = [s for s in issues if s.published <= race_date]
    return in_force[-1] if in_force else issues[0]
