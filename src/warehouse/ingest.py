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


def _windows(session_key: int, race_control: list[dict], laps: list[dict]) -> list[dict]:
    """SC / VSC / red-flag windows, derived with the same code the live tools use."""
    windows = neutralised_windows(
        [RaceControlMessage.model_validate(r) for r in race_control],
        [Lap.model_validate(r) for r in laps],
    )
    return [
        {"session_key": session_key, "start": s.isoformat(), "end": e.isoformat(), "kind": k}
        for s, e, k in windows
    ]


def _read(raw_dir: Path, table: str, session_key: int) -> list[dict]:
    path = raw_dir / table / f"{session_key}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def recompute_derived(raw_dir: Path = RAW_DIR) -> int:
    """Re-derive the neutralised windows for every ingested session from its stored race control
    and laps, so detection fixes apply to past sessions without re-downloading."""
    count = 0
    for path in sorted((raw_dir / "sessions").glob("*.jsonl")):
        sk = int(path.stem)

        race_control, laps = (_read(raw_dir, table, sk) for table in ("race_control", "laps"))
        _write("neutralised", sk, _windows(sk, race_control, laps), raw_dir)
        count += 1
    return count


def ingest_session(client: OpenF1Client, session: dict, raw_dir: Path = RAW_DIR) -> None:
    sk = session["session_key"]
    fetched: dict[str, list[dict]] = {}
    for table, endpoint in ENDPOINTS.items():
        params: Params = {"session_key": sk}
        fetched[table] = client._fetch(endpoint, params)

    for table, rows in fetched.items():
        _write(table, sk, rows, raw_dir)
    _write("neutralised", sk, _windows(sk, fetched["race_control"], fetched["laps"]), raw_dir)
    _write("sessions", sk, [session], raw_dir)  # written last: marks the session complete


def ingest(
    years: list[int],
    session_names: list[str],
    client: OpenF1Client | None = None,
    raw_dir: Path = RAW_DIR,
    now: datetime | None = None,
    settle: timedelta = SETTLE,
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
                if datetime.fromisoformat(end) + settle > now:
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
    ap.add_argument(
        "--settle-minutes",
        type=int,
        default=int(SETTLE.total_seconds() // 60),
        help="how long after a session ends before ingesting it (OpenF1 publishes free data "
        "~30 min after the end; use ~45 on race weekends)",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    new = ingest(args.years, args.sessions, settle=timedelta(minutes=args.settle_minutes))
    print(f"Ingested {len(new)} new sessions.")


if __name__ == "__main__":
    main()
