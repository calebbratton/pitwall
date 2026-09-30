"""Record trimmed OpenF1 responses as test fixtures for MockOpenF1Client.

Usage: python scripts/record_fixtures.py --year 2024 --place Monaco --drivers 4 81 16
"""

import argparse
import json
from pathlib import Path

from src.tools.models import (
    Driver,
    Lap,
    PitStop,
    Position,
    RaceControlMessage,
    Session,
    SessionResult,
    Stint,
)
from src.tools.openf1 import HttpOpenF1Client

FIXTURES_ROOT = Path("tests/fixtures/openf1")


def _dump(rows: list[dict], model: type, path: Path) -> None:
    # Keep only modelled fields (plus filter keys) so fixtures stay small and readable.
    keep = set(model.model_fields) | {"session_key", "meeting_key"}
    trimmed = [{k: v for k, v in r.items() if k in keep} for r in rows]
    path.write_text(json.dumps(trimmed, indent=1) + "\n")
    print(f"  {path} ({len(trimmed)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--place", required=True, help="country, location or circuit")
    ap.add_argument(
        "--drivers", type=int, nargs="+", required=True, help="drivers to record laps for"
    )
    args = ap.parse_args()

    client = HttpOpenF1Client()
    session = client.get_session(args.year, args.place)
    sk = session.session_key
    out = FIXTURES_ROOT / f"{args.year}_{args.place.lower().replace(' ', '_')}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Recording {args.place} {args.year} race (session_key={sk}) -> {out}")

    fetch = client._fetch
    # All of the season's races, so get_session's ambiguity handling is testable.
    _dump(
        fetch("sessions", {"year": args.year, "session_name": "Race"}),
        Session,
        out / "sessions.json",
    )
    _dump(fetch("drivers", {"session_key": sk}), Driver, out / "drivers.json")
    _dump(fetch("stints", {"session_key": sk}), Stint, out / "stints.json")
    _dump(fetch("race_control", {"session_key": sk}), RaceControlMessage, out / "race_control.json")
    _dump(fetch("session_result", {"session_key": sk}), SessionResult, out / "session_result.json")
    _dump(fetch("pit", {"session_key": sk}), PitStop, out / "pit.json")
    # Positions only change a few hundred times a race; the full series is small.
    _dump(fetch("position", {"session_key": sk}), Position, out / "position.json")
    laps = [
        row for d in args.drivers for row in fetch("laps", {"session_key": sk, "driver_number": d})
    ]
    _dump(laps, Lap, out / "laps.json")


if __name__ == "__main__":
    main()
