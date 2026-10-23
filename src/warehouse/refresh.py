"""Keep the data store current: one idempotent job for cron or the server.

Each run ingests every finished session (practice to race) that isn't in the warehouse yet,
from OpenF1 once its data has settled, rebuilds the warehouse if anything was added, and fetches
new stewards' decisions. Running it again with nothing new does nothing.

Race timing for the strategy models doesn't need this job: live sessions are recorded from F1's
feed by the server (`Recorder`), and a finished race is rebuilt from OpenF1 on demand when no
recording or archive copy exists (src/livetiming/sources.py).

Usage: python -m src.warehouse.refresh [--years 2026] [--settle-minutes 45] [--no-decisions]
Server: PITWALL_REFRESH_MINUTES=30 runs it on that interval in the API process.
"""

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from src.tools.openf1 import OpenF1Error

ALL_SESSIONS = (
    "Practice 1",
    "Practice 2",
    "Practice 3",
    "Sprint Qualifying",
    "Sprint",
    "Qualifying",
    "Race",
)
# OpenF1 frees a session's data ~30 min after it ends.
DEFAULT_SETTLE = timedelta(minutes=45)

log = logging.getLogger(__name__)


@dataclass
class RefreshReport:
    sessions: list[int] = field(default_factory=list)  # newly ingested session keys
    rebuilt: bool = False
    decisions: int | None = None  # stewards' documents on file after the run; None = failed
    skipped: str = ""  # why OpenF1 wasn't read this run (e.g. a session is live)
    errors: list[str] = field(default_factory=list)


def _ingest(years: list[int], sessions: list[str], settle: timedelta) -> list[int]:
    from src.warehouse.ingest import ingest

    return ingest(years, sessions, settle=settle)


def _build() -> None:
    from src.warehouse.build import build

    build()


def _decisions(seasons: list[int]) -> list:
    from src.rag.decisions import ingest

    return ingest(seasons)


def refresh(
    years: list[int] | None = None,
    settle: timedelta = DEFAULT_SETTLE,
    decisions: bool = True,
    ingest_fn: Callable[..., list[int]] = _ingest,
    build_fn: Callable[[], None] = _build,
    decisions_fn: Callable[[list[int]], list] = _decisions,
) -> RefreshReport:
    years = years or [datetime.now(UTC).year]
    report = RefreshReport()
    try:
        report.sessions = list(ingest_fn(years, list(ALL_SESSIONS), settle))
    except OpenF1Error as e:
        # OpenF1 refuses every request while a session is live: try again next run.
        report.skipped = str(e)
        log.warning("refresh: OpenF1 skipped: %s", e)
    if report.sessions:
        build_fn()
        report.rebuilt = True
    if decisions:
        try:
            report.decisions = len(decisions_fn(years))
        except Exception as e:  # noqa: BLE001 - a stewards' site outage mustn't stop the job
            report.errors.append(f"stewards' decisions: {type(e).__name__}: {e}")
            log.warning("refresh: decisions failed: %s", e)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", help="default: the current season")
    ap.add_argument("--settle-minutes", type=int, default=int(DEFAULT_SETTLE.total_seconds() // 60))
    ap.add_argument("--no-decisions", action="store_true", help="skip the stewards' documents")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    report = refresh(
        args.years, timedelta(minutes=args.settle_minutes), decisions=not args.no_decisions
    )
    print(
        f"ingested {len(report.sessions)} sessions, rebuilt: {report.rebuilt}, "
        f"stewards' documents: {report.decisions}"
        + (f", OpenF1 skipped: {report.skipped}" if report.skipped else "")
        + "".join(f"\nerror: {e}" for e in report.errors)
    )


if __name__ == "__main__":
    main()
