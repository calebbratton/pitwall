"""Ingest OpenF1 sessions into the warehouse's raw layer (JSON Lines, one file per table/session).

The raw layer is append-only and cheap to rebuild from; `build.py` turns it into query-ready
DuckDB tables. OpenF1's free tier allows 30 requests/min, so a full season takes ~10 minutes the
first time; everything is cached by the OpenF1 client, and sessions already ingested are skipped.

Usage: python -m src.warehouse.ingest --years 2026 [2025 2024] [--sessions Race "Practice 2"]
"""

import argparse
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.seasons import supported_seasons
from src.tools.models import Lap, RaceControlMessage
from src.tools.openf1 import HttpOpenF1Client, OpenF1Client, Params
from src.tools.telemetry import neutralised_windows

RAW_DIR = Path("data/warehouse/raw")
# table name -> (OpenF1 endpoint, extra params). One request per session each: the whole-session
# responses are small except intervals (~25K rows), which is still a single request.
ENDPOINTS: dict[str, str] = {
    "drivers": "drivers",
    "results": "session_result",
    "laps": "laps",
    "stints": "stints",
    "pits": "pit",
    "race_control": "race_control",
    "weather": "weather",
    "positions": "position",
    "intervals": "intervals",
}
SETTLE = timedelta(hours=6)  # only ingest sessions whose data has stopped changing

log = logging.getLogger(__name__)


def _write(table: str, session_key: int, rows: list[dict], raw_dir: Path) -> None:
    path = raw_dir / table / f"{session_key}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(path)  # atomic: a half-written session never looks complete


def ingest_session(client: OpenF1Client, session: dict, raw_dir: Path = RAW_DIR) -> None:
    sk = session["session_key"]
    fetched: dict[str, list[dict]] = {}
    for table, endpoint in ENDPOINTS.items():
        params: Params = {"session_key": sk}
        fetched[table] = client._fetch(endpoint, params)

    # Derived in Python (shared with the live tools): SC / VSC / red flag windows.
    laps = [Lap.model_validate(r) for r in fetched["laps"]]
    messages = [RaceControlMessage.model_validate(r) for r in fetched["race_control"]]
    windows = [
        {"session_key": sk, "start": s.isoformat(), "end": e.isoformat(), "kind": k}
        for s, e, k in neutralised_windows(messages, laps)
    ]
    for table, rows in fetched.items():
        _write(table, sk, rows, raw_dir)
    _write("neutralised", sk, windows, raw_dir)
    _write("sessions", sk, [session], raw_dir)  # written last: marks the session complete


def ingest(
    years: list[int],
    session_names: list[str],
    client: OpenF1Client | None = None,
    raw_dir: Path = RAW_DIR,
    now: datetime | None = None,
) -> list[int]:
    client = client or HttpOpenF1Client()
    now = now or datetime.now(UTC)
    done = []
    for year in years:
        for name in session_names:
            for session in client._fetch("sessions", {"year": year, "session_name": name}):
                sk = session["session_key"]
                end = session.get("date_end")
                if session.get("is_cancelled") or not end:
                    continue
                if datetime.fromisoformat(end) + SETTLE > now:
                    continue  # future or too recent
                if (raw_dir / "sessions" / f"{sk}.jsonl").exists():
                    continue
                log.warning("ingesting %s %s %s", year, session.get("location"), name)
                ingest_session(client, session, raw_dir)
                done.append(sk)
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=supported_seasons())
    ap.add_argument("--sessions", nargs="+", default=["Race"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    new = ingest(args.years, args.sessions)
    print(f"Ingested {len(new)} new sessions.")


if __name__ == "__main__":
    main()
