"""Score the locked-in predictions (prediction_log.py) against what happened.

For each recorded prediction: rank correlation of expected position vs the result, winner hit,
podium overlap, and the log-loss of the winner's predicted chance - against the luck-adjusted
result (the target the model is built for) and the raw classification.

Usage: python -m src.sim.scorecard [--year 2026]
"""

import argparse
import math

import numpy as np

from src.sim.luck import neutralisation_luck
from src.sim.prediction_log import load_all
from src.warehouse.queries import connect


def _orders(con, year: int, location: str) -> tuple[list[str], list[str]] | None:
    """(luck-adjusted order, raw classified order) as TLAs, or None if the race isn't in the
    warehouse yet."""
    row = con.execute(
        """SELECT session_key FROM races WHERE year = ? AND session_name = 'Race'
           AND (location ILIKE ? OR country_name ILIKE ?)""",
        [year, location, location],
    ).fetchone()
    if not row:
        return None
    sk = row[0]
    names = dict(
        con.execute(
            "SELECT driver_number, name_acronym FROM raw_drivers WHERE session_key = ?", [sk]
        ).fetchall()
    )
    adjusted = [names.get(d, str(d)) for d in neutralisation_luck(con, sk, 22.0).adjusted_order]
    raw = [
        names.get(d, str(d))
        for (d,) in con.execute(
            """SELECT driver_number FROM raw_results WHERE session_key = ? AND position IS NOT NULL
               AND NOT coalesce(dnf, FALSE) AND NOT coalesce(dns, FALSE)
               AND NOT coalesce(dsq, FALSE) ORDER BY position""",
            [sk],
        ).fetchall()
    ]
    return adjusted, raw


def score(table: list[dict], order: list[str]) -> dict:
    """Compare a prediction table with a finishing order (finishers only)."""
    pred = {r["tla"]: r for r in table}
    finishers = [t for t in order if t in pred]
    if len(finishers) < 3:
        return {}
    expected = np.array([pred[t]["expected"] for t in finishers])
    pred_rank = np.argsort(np.argsort(expected))
    actual_rank = np.arange(len(finishers))
    predicted_podium = [r["tla"] for r in sorted(table, key=lambda r: r["expected"])[:3]]
    favourite = max(table, key=lambda r: r["p_win"])["tla"]
    winner = finishers[0]
    return {
        "rank_corr": round(float(np.corrcoef(pred_rank, actual_rank)[0, 1]), 3),
        "winner": winner,
        "favourite": favourite,
        "winner_hit": favourite == winner,
        "podium_overlap": len(set(predicted_podium) & set(finishers[:3])),
        "winner_log_loss": round(-math.log(max(pred[winner]["p_win"], 1e-3)), 3),
    }


def scorecard(year: int) -> list[dict]:
    con = connect()
    rows = []
    for rec in load_all(year):
        orders = _orders(con, year, rec["location"])
        if orders is None:
            rows.append({**rec, "result": "pending"})
            continue
        adjusted, raw = orders
        rows.append(
            {
                "location": rec["location"],
                "stage": rec["stage"],
                "luck_adjusted": score(rec["table"], adjusted),
                "raw": score(rec["table"], raw),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()
    rows = scorecard(args.year)
    if not rows:
        print("no recorded predictions yet")
    for r in rows:
        if r.get("result") == "pending":
            print(f"{r['location']:16} {r['stage']:7} pending (race not in the warehouse yet)")
            continue
        for kind in ("luck_adjusted", "raw"):
            s = r[kind]
            if s:
                print(
                    f"{r['location']:16} {r['stage']:7} {kind:13} rank corr {s['rank_corr']:+.2f}  "
                    f"favourite {s['favourite']} / winner {s['winner']}  podium {s['podium_overlap']}/3  "
                    f"winner log-loss {s['winner_log_loss']:.2f}"
                )


if __name__ == "__main__":
    main()
