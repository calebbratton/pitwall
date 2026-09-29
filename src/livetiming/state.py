"""Merge F1 live-timing messages into the current state of every topic.

The feed sends each topic's full state once, then partial updates:
- nested dicts are merged key by key: {"Lines": {"4": {"InPit": true}}}
- lists are updated by index, sent as a dict with string keys: {"Stints": {"1": {...}}}
  (an index equal to the list length appends)
- {"_deleted": [keys]} removes entries (e.g. PitTimes rows disappear after a few seconds)

Transient data that the feed deletes but strategy needs later (pit-lane times) is kept in a
history on the tracker.
"""

import copy
from dataclasses import dataclass, field


def merge(target: dict | list, update: dict) -> None:
    """Apply one partial update to `target` in place."""
    for key, value in update.items():
        if key == "_deleted":
            for gone in value:
                if isinstance(target, dict):
                    target.pop(gone, None)
            continue
        if isinstance(target, list):
            index = int(key)
            while len(target) <= index:
                target.append({})
            current = target[index]
            if isinstance(value, dict) and isinstance(current, dict | list):
                merge(current, value)
            else:
                target[index] = copy.deepcopy(value)
            continue
        current = target.get(key)
        if isinstance(value, dict) and isinstance(current, dict | list):
            merge(current, value)
        else:
            target[key] = copy.deepcopy(value)


@dataclass(frozen=True)
class PitLaneTime:
    racing_number: str
    duration_s: float
    lap: int


@dataclass
class TimingState:
    topics: dict[str, dict] = field(default_factory=dict)
    pit_lane_times: list[PitLaneTime] = field(default_factory=list)
    clock: str = ""  # offset (archive) or UTC timestamp (live) of the last message applied

    def apply(self, topic: str, data: dict, clock: str = "") -> None:
        merge(self.topics.setdefault(topic, {}), data)
        if clock:
            self.clock = clock
        if topic == "PitLaneTimeCollection":
            self._record_pit_times(data)

    def _record_pit_times(self, data: dict) -> None:
        for number, row in (data.get("PitTimes") or {}).items():
            if number == "_deleted" or not isinstance(row, dict):
                continue
            try:
                entry = PitLaneTime(row["RacingNumber"], float(row["Duration"]), int(row["Lap"]))
            except (KeyError, ValueError):
                continue
            if entry not in self.pit_lane_times:
                self.pit_lane_times.append(entry)
