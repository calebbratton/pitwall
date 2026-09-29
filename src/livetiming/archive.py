"""F1's public live-timing archive: the same messages the live feed sent, with timestamps.

https://livetiming.formula1.com/static/<year>/<meeting>/<session>/<Topic>.jsonStream holds one
message per line as `HH:MM:SS.mmm{json}`, where the time is the offset from the stream start.
Replaying these through the state tracker is indistinguishable from the live feed, so live
features are built and tested on finished races. Files are cached under data/livetiming/.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote

import httpx

STATIC = "https://livetiming.formula1.com/static"
DEFAULT_CACHE = Path("data/livetiming")

# Topics a strategy call needs. Car telemetry/GPS (CarData.z, Position.z) are left out: they're
# large, compressed and gated behind an F1 TV login on the live feed.
STRATEGY_TOPICS = (
    "SessionInfo",
    "DriverList",
    "TimingData",
    "TimingAppData",
    "TrackStatus",
    "LapCount",
    "RaceControlMessages",
    "PitLaneTimeCollection",
)


@dataclass(frozen=True)
class Message:
    offset: timedelta  # since the start of the recording
    topic: str
    data: dict


def _get_text(url: str) -> str:
    resp = httpx.get(url, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    return resp.content.decode("utf-8-sig")  # files start with a BOM


def _parse_offset(stamp: str) -> timedelta:
    h, m, s = stamp.split(":")
    return timedelta(hours=int(h), minutes=int(m), seconds=float(s))


def find_session(year: int, meeting: str, session: str = "Race") -> str:
    """Archive path for e.g. (2024, "Miami"), matched case-insensitively against the meeting
    folder name. The yearly index isn't always complete; pass a full path to `ArchiveSession`
    when it misses a session."""
    index = json.loads(_get_text(f"{STATIC}/{year}/Index.json"))
    needle = meeting.casefold().replace(" ", "_")
    matches = [
        s["Path"]
        for m in index["Meetings"]
        for s in m["Sessions"]
        if s.get("Name") == session and needle in s["Path"].casefold()
    ]
    if len(matches) != 1:
        raise LookupError(f"{len(matches)} archived {session} sessions match {meeting!r} {year}")
    return matches[0]


class ArchiveSession:
    def __init__(self, path: str, cache_dir: Path = DEFAULT_CACHE) -> None:
        self.path = path.strip("/") + "/"
        self._cache = cache_dir / self.path

    def _stream(self, topic: str) -> str:
        cached = self._cache / f"{topic}.jsonStream"
        if cached.exists():
            return cached.read_text()
        text = _get_text(f"{STATIC}/{quote(self.path)}{topic}.jsonStream")
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(text)
        return text

    def messages(self, topics: tuple[str, ...] = STRATEGY_TOPICS) -> Iterator[Message]:
        """All messages for `topics`, merged in time order."""
        merged: list[Message] = []
        for topic in topics:
            for line in self._stream(topic).splitlines():
                brace = line.find("{")
                if brace <= 0:
                    continue
                merged.append(Message(_parse_offset(line[:brace]), topic, json.loads(line[brace:])))
        merged.sort(key=lambda m: m.offset)  # stable: keeps per-topic order on ties
        return iter(merged)
