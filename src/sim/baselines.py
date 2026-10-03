"""Is the simulator worth it? Score it against simple baselines on a season's races.

At the grid stage (official starting grid known), for each race:
  model  the pre-race simulator, calibrated (what the "grid" prediction logs)
  grid   P(win / podium | starting slot) from earlier seasons' results
  llm    an LLM (Groq, the analyst model) given the same per-driver numbers the simulator gets
         (grid, qualifying gap, practice long-run pace, season race pace, team) plus the last
         three results, asked for win / podium probabilities. Its training predates the season,
         so it can't know the results.
Scored against the luck-adjusted result (and raw): winner log-loss, podium log-loss (per car),
favourite = winner rate. LLM answers are cached in data/baselines/ (rerun with --refresh).

Usage: python -m src.sim.baselines --year 2026 [--refresh]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from src.sim.backtest import actual_order
from src.sim.calibrate import DEFAULT_CALIBRATION, apply
from src.sim.inputs import WeekendInputs, build_inputs
from src.sim.race import simulate
from src.warehouse.queries import connect

CACHE = Path("data/baselines")
SIMS = 4000
FLOOR = 1e-3


class DriverOdds(BaseModel):
    tla: str
    p_win: float = Field(description="Probability of winning, 0-1")
    p_podium: float = Field(description="Probability of finishing in the top 3, 0-1")


class RaceOdds(BaseModel):
    reasoning: str = Field(description="Two or three sentences on the key factors.")
    drivers: list[DriverOdds] = Field(description="Every driver on the grid.")


PROMPT = """\
You are an expert Formula 1 analyst. Predict the {year} {meeting} Grand Prix ({laps} laps) from
the data below, which is everything known before the start. Give every driver a probability of
winning (summing to 1) and of finishing on the podium (summing to 3). Ignore crashes, mechanical
retirements and safety-car luck: predict the result on pace and track position.

Columns: grid slot, driver, team, qualifying gap to pole (s), practice long-run pace vs the field
median (s/lap, negative = faster), season race pace gap to the fastest driver in earlier races
(s/lap).
{table}

Earlier {year} results (top 10, safety-car luck removed):
{recent}
"""


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:+.3f}"


def llm_odds(
    inputs: WeekendInputs, recent: list[str], model=None
) -> dict[str, tuple[float, float]]:
    from src.evals.judge import call_with_backoff
    from src.llm.factory import get_chat_model, with_schema

    rows = "\n".join(
        f"{d.grid:2} {d.tla} {d.team}: quali {_fmt(d.quali_delta_s)}, long run "
        f"{_fmt(d.long_run_delta_s)}, race pace {_fmt(d.race_pace_s)}"
        for d in sorted(inputs.drivers, key=lambda d: d.grid)
    )
    prompt = PROMPT.format(
        year=inputs.year,
        meeting=inputs.meeting,
        laps=inputs.laps,
        table=rows,
        recent="\n".join(recent) or "(none: first race)",
    )
    # 22 drivers of JSON plus hidden reasoning: give it room, and resample on Groq's
    # intermittent json_validate_failed (with_schema retries 3x; this adds rounds of that).
    runnable = with_schema(model or get_chat_model("analyst", max_tokens=6000), RaceOdds)
    for attempt in range(3):
        try:
            odds: RaceOdds = call_with_backoff(runnable.invoke, prompt)
            break
        except Exception as e:
            if "json_validate_failed" not in str(e) or attempt == 2:
                raise
    return {o.tla.upper(): (o.p_win, o.p_podium) for o in odds.drivers}


def _normalise(p: np.ndarray, total: float) -> np.ndarray:
    p = np.clip(p, 0, None)
    p = p / p.sum() * total if p.sum() > 0 else np.full(len(p), total / len(p))
    return np.clip(p, FLOOR, 1 - FLOOR)


def grid_rates(con, before_year: int) -> tuple[dict[int, float], dict[int, float]]:
    """P(win), P(podium) by starting slot from earlier seasons (classified finishers)."""
    rows = con.execute(
        """SELECT r.session_key FROM races r WHERE r.session_name = 'Race' AND r.year < ?""",
        [before_year],
    ).fetchall()
    starts: dict[int, int] = {}
    wins: dict[int, int] = {}
    pods: dict[int, int] = {}
    for (sk,) in rows:
        grid = dict(
            con.execute(
                """SELECT driver_number, arg_min(position, date::TIMESTAMPTZ) FROM raw_positions
                   WHERE session_key = ? GROUP BY 1""",
                [sk],
            ).fetchall()
        )
        finish = dict(
            con.execute(
                "SELECT driver_number, position FROM raw_results WHERE session_key = ?", [sk]
            ).fetchall()
        )
        for n, g in grid.items():
            starts[g] = starts.get(g, 0) + 1
            pos = finish.get(n)
            wins[g] = wins.get(g, 0) + (pos == 1)
            pods[g] = pods.get(g, 0) + (pos is not None and pos <= 3)
    # light smoothing so back-row slots aren't exactly 0
    win = {g: (wins.get(g, 0) + 0.05) / (n + 1) for g, n in starts.items()}
    pod = {g: (pods.get(g, 0) + 0.15) / (n + 1) for g, n in starts.items()}
    return win, pod


def _scores(p_win: np.ndarray, p_pod: np.ndarray, rank: np.ndarray) -> dict[str, float]:
    winner = int(np.argmin(rank))
    on_podium = (rank <= 3).astype(float)
    return {
        "win_ll": -math.log(p_win[winner]),
        "podium_ll": float(
            -np.mean(on_podium * np.log(p_pod) + (1 - on_podium) * np.log(1 - p_pod))
        ),
        "fav_hit": float(int(np.argmax(p_win)) == winner),
    }


def run(year: int, refresh: bool = False) -> dict[str, dict[str, list[float]]]:
    con = connect()
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / f"llm_{year}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() and not refresh else {}
    win_rate, pod_rate = grid_rates(con, year)
    locations = [
        r[0]
        for r in con.execute(
            """SELECT location FROM races WHERE year = ? AND session_name = 'Race'
               ORDER BY date_start""",
            [year],
        ).fetchall()
    ]
    results: dict[str, dict[str, list[float]]] = {}
    recent: list[str] = []
    for location in locations:
        inputs = build_inputs(con, year, location)
        ranks = {
            "luck-adjusted": actual_order(con, inputs, adjusted=True),
            "raw": actual_order(con, inputs, adjusted=False),
        }
        tlas = [d.tla for d in inputs.drivers]

        table = apply(simulate(inputs, sims=SIMS, seed=3).table(), DEFAULT_CALIBRATION, SIMS)
        by_tla = {r["tla"]: r for r in table}
        model = (
            np.array([by_tla[t]["p_win"] for t in tlas]),
            np.array([by_tla[t]["p_podium"] for t in tlas]),
        )
        grid = (
            np.array([win_rate.get(d.grid, FLOOR) for d in inputs.drivers]),
            np.array([pod_rate.get(d.grid, FLOOR) for d in inputs.drivers]),
        )
        if location not in cache:
            cache[location] = llm_odds(inputs, recent)
            cache_path.write_text(json.dumps(cache, indent=1))
        odds = cache[location]
        llm = (
            np.array([odds.get(t, (0.0, 0.0))[0] for t in tlas]),
            np.array([odds.get(t, (0.0, 0.0))[1] for t in tlas]),
        )
        for label, rank in ranks.items():
            finished = ~np.isnan(rank)
            if finished.sum() < 3:
                continue
            for name, (pw, pp) in (("model", model), ("grid", grid), ("llm", llm)):
                pw_, pp_ = _normalise(pw[finished], 1.0), _normalise(pp[finished], 3.0)
                s = _scores(pw_, pp_, rank[finished])
                bucket = results.setdefault(label, {}).setdefault(name, {})
                for k, v in s.items():
                    bucket.setdefault(k, []).append(v)
        adj = ranks["luck-adjusted"]
        order = [tlas[i] for i in np.argsort(np.where(np.isnan(adj), 99, adj))[:10]]
        recent.append(f"{location}: {' '.join(order)}")
        recent = recent[-3:]
    con.close()
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--refresh", action="store_true", help="ask the LLM again")
    args = ap.parse_args()
    results = run(args.year, args.refresh)
    for label, by_method in results.items():
        n = len(next(iter(by_method.values()))["win_ll"])
        print(f"\n{args.year}, {n} races, scored on the {label} result (lower log-loss is better)")
        print(
            f"  {'method':6} {'winner log-loss':>16} {'podium log-loss':>16} {'favourite won':>14}"
        )
        for name, s in by_method.items():
            print(
                f"  {name:6} {np.mean(s['win_ll']):16.3f} {np.mean(s['podium_ll']):16.3f} "
                f"{int(sum(s['fav_hit']))}/{n:>12}"
            )


if __name__ == "__main__":
    main()
