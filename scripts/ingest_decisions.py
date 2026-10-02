"""Fetch the FIA stewards' per-car rulings for the supported seasons (or --seasons) into
data/decisions/. Incremental: documents already parsed are skipped. Safe to run while the API
is up (it reloads the records on the next question).

Usage: python scripts/ingest_decisions.py [--seasons 2026 2025]
"""

import argparse
import logging
from collections import Counter

from src.rag.decisions import ingest
from src.seasons import supported_seasons

logging.getLogger("pypdf").setLevel(logging.ERROR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=supported_seasons())
    args = ap.parse_args()
    records = ingest(args.seasons)
    by = Counter((r.season, r.competition) for r in records)
    for (season, comp), n in sorted(by.items()):
        print(f"{season} {comp}: {n}")
    print(f"{len(records)} rulings")


if __name__ == "__main__":
    main()
