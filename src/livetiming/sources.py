"""Where a finished race's live-timing messages come from, best source first:

  1. our own recording of the live feed (data/livetiming/recordings/, written by `Recorder`)
  2. F1's static archive, local cache first (data/livetiming/<year>/...)
  3. rebuilt from OpenF1 (`openf1_feed`): always available ~30 min after the race, but only the
     topics the strategy models need

The archive has answered 401 since 2026-10-16, so for new races the live recording and OpenF1
are what keep the models current.
"""

import itertools
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

from src.livetiming.archive import STRATEGY_TOPICS, ArchiveSession, Message
from src.livetiming.client import DEFAULT_RECORDINGS
from src.tools.models import Session
from src.tools.openf1 import HttpOpenF1Client, OpenF1Client

SOURCES = ("recording", "archive", "openf1")


@dataclass(frozen=True)
class RaceMessages:
    messages: list[Message]
    source: str  # "recording <folder>", "archive <path>" or "OpenF1"


def _session_info(folder: Path) -> dict | None:
    path = folder / "SessionInfo.jsonStream"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        brace = line.find("{")
        if brace >= 0:
            try:
                return json.loads(line[brace:])
            except json.JSONDecodeError:
                continue
    return None


def covers_race(messages: list[Message], share: float = 0.9) -> bool:
    """A usable recording of a race: some car's completed-lap count climbs from lap 1 without
    going backwards and covers `share` of the race distance. A recorder started late, a
    reconnect that only caught the final state, or overlapping recordings fail this."""
    total = None
    laps: dict[str, list[int]] = {}
    for m in messages:
        if m.topic == "LapCount" and m.data.get("TotalLaps"):
            total = int(m.data["TotalLaps"])
        elif m.topic == "TimingData":
            for car, line in (m.data.get("Lines") or {}).items():
                if isinstance(line, dict) and "NumberOfLaps" in line:
                    try:
                        laps.setdefault(car, []).append(int(line["NumberOfLaps"]))
                    except (TypeError, ValueError):
                        continue
    if not total or not laps:
        return False
    for seen in laps.values():
        ordered = all(b >= a for a, b in itertools.pairwise(seen))
        covered = len(set(seen) & set(range(1, total + 1)))
        if ordered and min(seen) <= 2 and covered >= share * total:
            return True
    return False


def find_recordings(day: str, session: str = "Race", root: Path = DEFAULT_RECORDINGS) -> list[str]:
    """Recording folders of the `session` held on `day` (YYYY-MM-DD, one day either side: the
    feed dates sessions in local time), most timing data first. A session can have several
    (restarts, reconnects after it ended); `covers_race` tells the usable ones."""
    if not root.is_dir():
        return []
    target = date.fromisoformat(day)
    found = []
    for folder in root.iterdir():
        info = _session_info(folder) if folder.is_dir() else None
        if not info or info.get("Name") != session:
            continue
        try:
            held = date.fromisoformat(str(info.get("StartDate", ""))[:10])
        except ValueError:
            continue
        timing = folder / "TimingData.jsonStream"
        if abs((held - target).days) <= 1 and timing.exists():
            found.append((timing.stat().st_size, folder.name))
    return [name for _, name in sorted(found, reverse=True)]


def race_messages(
    year: int,
    place: str,
    day: str | None = None,
    topics: Iterable[str] = STRATEGY_TOPICS,
    sources: Iterable[str] = SOURCES,
    client: OpenF1Client | None = None,
    session: Session | None = None,
    recordings: Path = DEFAULT_RECORDINGS,
) -> RaceMessages:
    """The race's messages from the first source that has them (see module docstring).
    `day`: the race date (YYYY-MM-DD), needed to match recordings; looked up on OpenF1 when
    not given."""
    from src.sim.pit_review import archive_race

    topics = tuple(topics)
    errors = []
    for source in sources:
        if source == "recording":
            if day is None:
                client = client or HttpOpenF1Client()
                session = session or client.get_session(year, place)
                day = session.date_start[:10]
            for folder in find_recordings(day, root=recordings):
                messages = list(ArchiveSession(folder, cache_dir=recordings).messages(topics))
                if covers_race(messages):
                    return RaceMessages(messages, f"recording {folder}")
            errors.append("no complete recording")
        elif source == "archive":
            try:
                archived = archive_race(year, place, day)
                return RaceMessages(list(archived.messages(topics)), f"archive {archived.path}")
            except (LookupError, httpx.HTTPError) as e:
                errors.append(f"archive: {e}")
        elif source == "openf1":
            from src.livetiming.openf1_feed import openf1_messages

            client = client or HttpOpenF1Client()
            session = session or client.get_session(year, place)
            return RaceMessages(openf1_messages(client, session), "OpenF1")
    raise LookupError(f"no source has {place} {year}: {'; '.join(errors)}")
