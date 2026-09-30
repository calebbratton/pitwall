"""Predict a race's finishing order from the weekend's practice and qualifying.

Usage (after qualifying, once the sessions are in the warehouse):
  python -m src.warehouse.ingest --years 2026 --sessions "Practice 1" "Practice 2" \\
      "Practice 3" Qualifying "Sprint Qualifying" Sprint --settle-minutes 45
  python -m src.warehouse.build
  python -m src.sim.predict --year 2026 --place "Kuala Lumpur" --laps 56
"""

import argparse
import json

from src.sim.calibrate import DEFAULT_CALIBRATION, apply
from src.sim.inputs import _grid, build_inputs, penalised_grid, weekend_sessions
from src.sim.race import simulate
from src.warehouse.queries import connect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--place", required=True)
    ap.add_argument("--laps", type=int, help="race distance (required before the race)")
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument(
        "--penalty",
        nargs="*",
        default=[],
        metavar="TLA=N",
        help="grid-place penalties from the FIA documents, e.g. VER=10 HAM=5",
    )
    ap.add_argument("--pit-lane", nargs="*", default=[], metavar="TLA", help="pit-lane starters")
    ap.add_argument("--raw", action="store_true", help="uncalibrated simulator probabilities")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = ap.parse_args()

    con = connect()
    grid = None
    if args.penalty or args.pit_lane:
        base = build_inputs(con, args.year, args.place, laps=args.laps)
        number = {d.tla: d.number for d in base.drivers}
        unknown = [t for t in [p.split("=")[0] for p in args.penalty] + args.pit_lane if t.upper() not in number]
        if unknown:
            ap.error(f"unknown driver(s) {unknown}; choose from {sorted(number)}")
        _, sessions = weekend_sessions(con, args.year, args.place)
        grid = penalised_grid(
            _grid(con, None, sessions.get("Qualifying")),
            {number[t.upper()]: int(n) for t, n in (p.split("=") for p in args.penalty)},
            {number[t.upper()] for t in args.pit_lane},
        )
    inputs = build_inputs(con, args.year, args.place, laps=args.laps, grid=grid)
    prediction = simulate(inputs, sims=args.sims)
    table = prediction.table()
    if not args.raw:
        table = apply(table, DEFAULT_CALIBRATION, args.sims)
    notes = list(inputs.notes)
    if grid:
        notes.append(
            "grid penalties applied: "
            + ", ".join([*args.penalty, *(f"{t} pit lane" for t in args.pit_lane)])
        )
    elif inputs.race_session_key is None:
        notes.append("grid = qualifying order (no penalties given: use --penalty / --pit-lane)")
    if args.laps:
        notes.append(f"race distance assumed: {args.laps} laps")
    missing_pace = [d.tla for d in inputs.drivers if d.quali_delta_s is None]
    if missing_pace:
        notes.append(f"no qualifying lap for {', '.join(missing_pace)}: placed by grid slot")

    if args.json:
        print(
            json.dumps(
                {"meeting": inputs.meeting, "year": inputs.year, "table": table, "notes": notes},
                indent=1,
            )
        )
        return
    print(f"{inputs.meeting} {inputs.year} — predicted finishing order ({args.sims} simulations)")
    print(f"{'':3} {'driver':6} {'grid':>4} {'exp. pos':>8} {'win':>6} {'podium':>7} {'points':>7}")
    grid = {d.tla: d.grid for d in inputs.drivers}
    for i, row in enumerate(table, 1):
        print(
            f"{i:>2}. {row['tla']:6} {grid[row['tla']]:>4} {row['expected']:>8.1f} "
            f"{row['p_win']:>6.0%} {row['p_podium']:>7.0%} {row['p_points']:>7.0%}"
        )
    print("\nNotes:")
    for note in notes:
        print(f"- {note}")
    print(
        "- backtest (2026, out-of-sample, vs luck-adjusted results): rank corr. 0.86 "
        "(qualifying order 0.86, grid 0.83); winner log-loss 1.66 vs 1.76 for grid win rates. "
        "Positions assume every car finishes and no SC/VSC luck. Probabilities are calibrated "
        "on 2026 results (the raw simulator is overconfident; --raw shows it)."
    )


if __name__ == "__main__":
    main()
