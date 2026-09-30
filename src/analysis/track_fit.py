"""Which kinds of circuit suit a team: measured circuit features vs the team's luck-free form.

Circuit features (all measured, no hand labels):
  length_km, corners, corners_per_km   MultiViewer outline/corners (else traced from Position.z)
  avg_speed_kmh                        length / pole lap time
  top_speed_kmh                        90th percentile qualifying speed trap
  compound_softness                    mean Pirelli C-number nominated (1 hard .. 5 soft)
Team form per race (luck-free by construction):
  quali_gap_pct                        team's best qualifying lap vs pole, % of lap time
  race_pace_gap_pct                    team's median clean race lap vs the fastest team's, %
  adjusted_finish                      mean luck-adjusted finishing position of its cars

Usage: python -m src.analysis.track_fit --team McLaren [--year 2026]
"""

import argparse
import math
import statistics
from dataclasses import asdict, dataclass

import numpy as np

from src.livetiming.archive import STRATEGY_TOPICS, ArchiveSession, list_sessions
from src.livetiming.circuits import fetch_circuit
from src.livetiming.track import track_outline
from src.models.compounds import nominations
from src.sim.luck import neutralisation_luck
from src.warehouse.queries import connect

FEATURES = ("avg_speed_kmh", "top_speed_kmh", "corners_per_km", "length_km", "compound_softness")
FEATURE_LABELS = {
    "avg_speed_kmh": "average lap speed",
    "top_speed_kmh": "top speed (speed trap)",
    "corners_per_km": "corner density",
    "length_km": "lap length",
    "compound_softness": "softer Pirelli compounds (lower tyre severity)",
}


@dataclass(frozen=True)
class RaceFit:
    location: str
    avg_speed_kmh: float | None
    top_speed_kmh: float | None
    corners_per_km: float | None
    length_km: float | None
    compound_softness: float | None
    quali_gap_pct: float | None
    race_pace_gap_pct: float | None
    adjusted_finish: float | None


def _outline_length_km(points) -> float:
    pts = [tuple(p) for p in points]
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:] + pts[:1], strict=True)) / 10_000


def circuit_shape(year: int, location: str, circuit_key: int | None, race_date: str):
    """(length_km, corners) from MultiViewer, else a traced outline (length only)."""
    circuit = fetch_circuit(int(circuit_key), year) if circuit_key else None
    if circuit:
        return _outline_length_km(circuit.points), len(circuit.corners) or None
    for race in list_sessions(year):
        if race["date"] == race_date:
            messages = list(ArchiveSession(race["path"]).messages((*STRATEGY_TOPICS, "Position.z")))
            outline = track_outline(messages)
            return (_outline_length_km(outline["points"]) if outline else None), None
    return None, None


def season_fit(con, year: int, team: str) -> list[RaceFit]:
    races = con.execute(
        """SELECT r.location, r.meeting_key, any_value(r.circuit_key),
                  strftime(min(r.date_start) FILTER (r.session_name = 'Race'), '%Y-%m-%d'),
                  any_value(r.session_key) FILTER (r.session_name = 'Race'),
                  any_value(r.session_key) FILTER (r.session_name = 'Qualifying')
           FROM races r WHERE r.year = ? GROUP BY 1, 2 ORDER BY 4""",
        [year],
    ).fetchall()
    noms = nominations(year)
    out = []
    for location, _, circuit_key, race_date, race_sk, quali_sk in races:
        if race_sk is None or quali_sk is None:
            continue
        length, corners = circuit_shape(year, location, circuit_key, race_date)
        pole, top = con.execute(
            """SELECT min(lap_time), quantile_cont(speed_trap, 0.9) FROM laps
               WHERE session_key = ? AND lap_time IS NOT NULL""",
            [quali_sk],
        ).fetchone()
        teams = dict(
            con.execute(
                "SELECT driver_number, team_name FROM raw_drivers WHERE session_key = ?", [race_sk]
            ).fetchall()
        )
        mine = {d for d, t in teams.items() if t and team.casefold() in t.casefold()}
        best = con.execute(
            f"""SELECT min(lap_time) FROM laps WHERE session_key = ?
                AND driver_number IN ({",".join("?" * len(mine)) or "NULL"})""",
            [quali_sk, *mine],
        ).fetchone()[0]
        paces = dict(
            con.execute(
                """SELECT d.team_name, median(l.lap_time) FROM clean_laps l
                   JOIN raw_drivers d ON d.session_key = l.session_key
                    AND d.driver_number = l.driver_number
                   WHERE l.session_key = ? GROUP BY 1 HAVING count(*) >= 20""",
                [race_sk],
            ).fetchall()
        )
        my_pace = next((p for t, p in paces.items() if t and team.casefold() in t.casefold()), None)
        luck = neutralisation_luck(con, race_sk, 22.0)
        finishes = [i + 1 for i, d in enumerate(luck.adjusted_order) if d in mine]
        c_numbers = [int(c[1]) for c in noms.get(location, {}).values()]
        out.append(
            RaceFit(
                location=location,
                avg_speed_kmh=length / pole * 3600 if length and pole else None,
                top_speed_kmh=float(top) if top else None,
                corners_per_km=corners / length if corners and length else None,
                length_km=length,
                compound_softness=statistics.mean(c_numbers) if c_numbers else None,
                quali_gap_pct=100 * (best - pole) / pole if best and pole else None,
                race_pace_gap_pct=(
                    100 * (my_pace - min(paces.values())) / min(paces.values())
                    if my_pace and paces
                    else None
                ),
                adjusted_finish=statistics.mean(finishes) if finishes else None,
            )
        )
    return out


def correlations(fits: list[RaceFit], metric: str = "quali_gap_pct") -> list[dict]:
    """Per feature: Pearson r with the team's gap (negative r = the team is closer to the front
    as the feature increases), how many races, and the best/worst races."""
    rows = []
    for feature in FEATURES:
        pairs = [
            (getattr(f, feature), getattr(f, metric))
            for f in fits
            if getattr(f, feature) is not None and getattr(f, metric) is not None
        ]
        if len(pairs) < 6:
            continue
        x, y = np.array(pairs, dtype=float).T
        r = float(np.corrcoef(x, y)[0, 1])
        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "r": round(r, 2),
                "n": len(pairs),
            }
        )
    return sorted(rows, key=lambda row: -abs(row["r"]))


def team_track_profile(team: str, year: int) -> dict:
    fits = season_fit(connect(), year, team)
    by_quali = sorted(
        [f for f in fits if f.quali_gap_pct is not None], key=lambda f: f.quali_gap_pct
    )
    return {
        "team": team,
        "year": year,
        "races": len(fits),
        "quali_correlations": correlations(fits, "quali_gap_pct"),
        "race_pace_correlations": correlations(fits, "race_pace_gap_pct"),
        "best_quali": [(f.location, round(f.quali_gap_pct, 3)) for f in by_quali[:4]],
        "worst_quali": [(f.location, round(f.quali_gap_pct, 3)) for f in by_quali[-4:]],
        "per_race": [
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(f).items()}
            for f in fits
        ],
        "note": "r < 0: the team is closer to pole / the fastest race pace as the feature grows. "
        "15 races: treat |r| < 0.5 as weak.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()
    profile = team_track_profile(args.team, args.year)
    print(f"{args.team} {args.year}: {profile['races']} races")
    print(
        f"{'race':18} {'speed':>6} {'top':>5} {'crn/km':>6} {'soft':>4} {'Q gap%':>7} {'R gap%':>7} {'adj fin':>7}"
    )
    for r in profile["per_race"]:
        fmt = lambda v, s: "   -" if v is None else format(v, s)
        print(
            f"{r['location']:18} {fmt(r['avg_speed_kmh'], '6.1f')} {fmt(r['top_speed_kmh'], '5.0f')} "
            f"{fmt(r['corners_per_km'], '6.2f')} {fmt(r['compound_softness'], '4.1f')} "
            f"{fmt(r['quali_gap_pct'], '7.3f')} {fmt(r['race_pace_gap_pct'], '7.3f')} {fmt(r['adjusted_finish'], '7.1f')}"
        )
    for name in ("quali_correlations", "race_pace_correlations"):
        print(f"\n{name}:")
        for c in profile[name]:
            print(f"  {c['label']:48} r {c['r']:+.2f}  (n={c['n']})")
    print(
        "\nbest qualifying:", profile["best_quali"], "\nworst qualifying:", profile["worst_quali"]
    )


if __name__ == "__main__":
    main()
