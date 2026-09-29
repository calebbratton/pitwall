"""Track map data from the `Position.z` topic: the circuit outline and car coordinates.

The feed has no map, so the outline is traced from the race leader's own path over one early
lap (the leader rarely pits then, so the path stays on the racing line, not the pit lane).
"""

from typing import Any

from src.livetiming.archive import Message
from src.livetiming.state import TimingState

OUTLINE_LAP = 3  # early, usually green, past the lap-1 chaos
MIN_POINT_SPACING = 40.0  # feed units; keeps the outline to a few hundred points


def latest_positions(message: Message) -> tuple[str, dict[str, dict[str, Any]]] | None:
    """(timestamp, {number: {X, Y, Status}}) of the newest sample in a Position.z message."""
    samples = message.data.get("Position") or []
    if not samples:
        return None
    newest = samples[-1]
    return newest.get("Timestamp", ""), newest.get("Entries") or {}


def _leader_at_lap(messages: list[Message], lap: int) -> tuple[str, float, float] | None:
    """(leader's number when `lap` starts, start offset of `lap`, start of lap + 1) in seconds.
    The leader comes from merged TimingData state: position updates only arrive on changes."""
    timing = TimingState()
    starts: dict[int, float] = {}
    leader: str | None = None
    for m in messages:
        if m.topic == "TimingData":
            timing.apply(m.topic, m.data)
        elif m.topic == "LapCount" and "CurrentLap" in m.data:
            current = int(m.data["CurrentLap"])
            starts.setdefault(current, m.offset.total_seconds())
            if current == lap:
                lines = timing.topics.get("TimingData", {}).get("Lines", {})
                leader = next(
                    (n for n, line in lines.items() if str(line.get("Position")) == "1"), None
                )
            if current > lap:
                break
    if leader is None or lap not in starts or lap + 1 not in starts:
        return None
    return leader, starts[lap], starts[lap + 1]


def track_outline(messages: list[Message], lap: int = OUTLINE_LAP) -> dict[str, Any] | None:
    """{"points": [[x, y], ...], "bounds": [min_x, min_y, max_x, max_y]} or None."""
    found = _leader_at_lap(messages, lap)
    if not found:
        return None
    leader, start, end = found
    points: list[tuple[float, float]] = []
    for m in messages:
        if m.topic != "Position.z" or not start <= m.offset.total_seconds() <= end:
            continue
        for sample in m.data.get("Position") or []:
            car = (sample.get("Entries") or {}).get(leader)
            if not car or car.get("Status") != "OnTrack":
                continue
            x, y = float(car["X"]), float(car["Y"])
            if x == 0 and y == 0:  # no fix yet
                continue
            last = points[-1] if points else None
            if last is None or abs(x - last[0]) + abs(y - last[1]) >= MIN_POINT_SPACING:
                points.append((x, y))
    if len(points) < 20:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return {"points": [list(p) for p in points], "bounds": [min(xs), min(ys), max(xs), max(ys)]}
