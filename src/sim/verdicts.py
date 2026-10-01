"""Post-race strategy verdicts: for each top finisher's pit stops, was it the right call?

Runs the pit review (src/sim/pit_review.py) on every stop of the luck-adjusted top N and labels
it: right call / <alternative> was better (by how much) / unlucky (a rival near them got a cheap
SC/VSC stop just before). Cached in data/verdicts/<year>-<location>.json because each review
replays the race and simulates the alternatives.

Usage: python -m src.sim.verdicts --year 2026 --place Madrid [--top 10]
"""

import argparse
import json
import re
from pathlib import Path

from src.sim.pit_review import review
from src.sim.scorecard import _orders
from src.warehouse.queries import connect

CACHE = Path("data/verdicts")


def _stop_laps(con, year: int, location: str, tla: str) -> list[int]:
    rows = con.execute(
        """SELECT p.lap_number FROM raw_pits p
           JOIN raw_drivers d ON d.session_key = p.session_key AND d.driver_number = p.driver_number
           JOIN races r ON r.session_key = p.session_key
           WHERE r.year = ? AND r.location ILIKE ? AND r.session_name = 'Race'
             AND d.name_acronym = ? ORDER BY 1""",
        [year, location, tla],
    ).fetchall()
    return [int(r[0]) for r in rows if r[0]]


def verdict_for(r) -> dict:
    actual, best = r.branches[0], min(r.branches, key=lambda b: b.expected)
    unlucky = next((n for n in r.notes if "neutralisation's timing" in n), None)
    margin = actual.expected - best.expected
    if best is actual or margin < 0.15:
        label = "unlucky" if unlucky else "right call"
        summary = (
            "Right call - but a rival's cheap SC/VSC stop just before cost them"
            if unlucky
            else "Right call (or as good as any alternative)"
        )
    else:
        label = "better option"
        summary = f"{best.label.capitalize()} would likely have been better (P{best.expected:.1f} vs P{actual.expected:.1f})"
    return {
        "driver": r.driver,
        "lap": r.lap,
        "label": label,
        "summary": summary,
        "tyres": f"{r.old_compound} ({r.old_tyre_age} laps) -> {r.new_compound}",
        "status": r.status,
        "expected_actual": actual.expected,
        "best_option": best.label,
        "expected_best": best.expected,
        "context": unlucky,
    }


def verdicts(year: int, place: str, top: int = 10, refresh: bool = False) -> list[dict]:
    path = CACHE / f"{year}-{re.sub(r'[^a-z0-9]+', '-', place.casefold())}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    con = connect()
    try:
        orders = _orders(con, year, place)
        if orders is None:
            raise LookupError(f"no {year} race at {place!r} in the warehouse")
        adjusted, _ = orders
        out = []
        for tla in adjusted[:top]:
            for lap in _stop_laps(con, year, place, tla):
                try:
                    out.append(verdict_for(review(year, place, tla, lap, sims=1500)))
                except Exception as e:  # noqa: BLE001 - one stop failing shouldn't lose the rest
                    out.append(
                        {"driver": tla, "lap": lap, "label": "unavailable", "summary": str(e)}
                    )
    finally:
        con.close()
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--place", required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    for v in verdicts(args.year, args.place, args.top, args.refresh):
        print(f"{v['driver']:4} L{v['lap']:<3} {v['label']:13} {v['summary']}")


if __name__ == "__main__":
    main()
