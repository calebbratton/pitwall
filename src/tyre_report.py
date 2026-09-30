"""Print the fitted tyre model for a race from the warehouse.

Usage: python -m src.tyre_report --year 2026 --place Baku [--session Race]
"""

import argparse

from src.models.tyres import fit_tyre_model
from src.warehouse.queries import clean_laps, connect, find_session


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--place", required=True)
    ap.add_argument("--session", default="Race")
    args = ap.parse_args()

    con = connect()
    session_key = find_session(con, args.year, args.place, args.session)
    laps = clean_laps(con, session_key)
    model = fit_tyre_model(laps)
    basis = (
        "fuel + track evolution from lap effects"
        if model.method == "panel"
        else f"fuel gain assumed {model.fuel_gain_s_per_lap} s/lap"
    )
    print(
        f"{args.place} {args.year} {args.session}: {len(laps)} clean laps, {model.method} method "
        f"({basis}); reference {model.reference_compound}"
    )
    if model.note:
        print(f"  note: {model.note}")
    print(
        "  fresh-tyre offsets are UNVALIDATED (race-only fits are biased; needs practice/quali data)"
    )
    print(f"{'compound':10} {'deg s/lap':>10} {'80% range':>18} {'fresh vs ref':>13} stints laps")
    for fit in sorted(model.compounds.values(), key=lambda f: f.deg_s_per_lap):
        offset = "-" if fit.offset_s is None else f"{fit.offset_s:+.2f}s"
        band = f"{fit.deg_low:+.3f}..{fit.deg_high:+.3f}"
        print(
            f"{fit.compound:10} {fit.deg_s_per_lap:>+10.3f} {band:>18} {offset:>13} "
            f"{fit.n_stints:>6} {fit.n_laps:>4}"
            + ("  (few stints: low confidence)" if fit.n_stints < 3 else "")
        )


if __name__ == "__main__":
    main()
