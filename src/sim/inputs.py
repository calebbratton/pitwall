"""Everything the race simulator needs for one weekend, from pre-race data only.

Backtests must not peek at the race: season-wide rates exclude the target race, pace comes from
practice / sprint / qualifying, and the grid is the order cars lined up in (known at lights out).
"""

import functools
import statistics
from dataclasses import dataclass, field

from src.livetiming.circuits import fetch_circuit
from src.models.tyres import CleanLap, fit_tyre_model
from src.warehouse.queries import connect

MIN_LONG_RUN_LAPS = 6
SPRINT_WEIGHT = 2.0  # a sprint is a real race: its pace counts double vs a practice long run
DEFAULT_PIT_LOSS = (22.0, 13.5, 15.5)  # green, SC, VSC — when no measured circuit data exists
DEFAULT_DEG = {"SOFT": 0.09, "MEDIUM": 0.06, "HARD": 0.045}  # 2026-ish fallback, s/lap


@dataclass
class DriverInput:
    number: int
    tla: str
    team: str
    grid: int
    quali_delta_s: float | None  # best qualifying lap minus the pole lap
    long_run_delta_s: float | None  # practice/sprint race pace vs the field median
    form_s: float = 0.0  # team's race-vs-quali pace bias from earlier races (s/lap)
    form_races: int = 0
    # The driver's race-pace gap to the fastest driver in this season's earlier races (s/lap,
    # recency-weighted, shrunk toward 0 with few races). The one pre-race signal beyond
    # qualifying that helped in every season 2023-26; per driver beat the team average
    # (2023-26: 58 vs 56 winners of 85, calibrated winner log-loss 1.085 vs 1.100).
    race_pace_s: float = 0.0
    # Qualifying looked compromised (far slower than teammate + usual gap): quali_delta_s holds
    # the teammate-based estimate instead of the lap actually set.
    quali_compromised: bool = False


@dataclass
class WeekendInputs:
    meeting: str
    year: int
    race_session_key: int | None
    laps: int
    drivers: list[DriverInput]
    deg: dict[str, float]  # s/lap of tyre age, per compound
    pit_loss_green: float
    pit_loss_sc: float
    sc_per_race: float  # full safety cars per race
    dnf_prob: float  # per car per race (the simulator ignores DNFs by default)
    notes: list[str] = field(default_factory=list)
    vsc_per_race: float = 0.0
    pit_loss_vsc: float = DEFAULT_PIT_LOSS[2]
    sc_laps: float = 6.0  # typical safety car length, laps (incl. the restart lap)
    vsc_laps: float = 1.4  # typical VSC length, laps
    # On-track pass rate at this circuit relative to the median circuit, from earlier seasons
    # (1.0 = median or unknown). The simulator scales the pass threshold by it.
    pass_rel: float = 1.0


def weekend_sessions(con, year: int, place: str) -> tuple[int, dict[str, int]]:
    """(meeting_key, {session_name: session_key}) for the weekend at `place`."""
    rows = con.execute(
        """
        SELECT meeting_key, session_name, session_key FROM races
        WHERE year = ? AND (location ILIKE ? OR country_name ILIKE ? OR circuit_short_name ILIKE ?)
        ORDER BY date_start
        """,
        [year, f"%{place}%", f"%{place}%", f"%{place}%"],
    ).fetchall()
    meetings = {r[0] for r in rows}
    if len(meetings) != 1:
        raise LookupError(f"{len(meetings)} {year} weekends match {place!r} in the warehouse")
    return rows[0][0], {name: sk for _, name, sk in rows}


def _drivers(con, session_keys: list[int]) -> dict[int, tuple[str, str]]:
    rows = con.execute(
        f"""SELECT driver_number, any_value(name_acronym), any_value(team_name) FROM raw_drivers
            WHERE session_key IN ({",".join("?" * len(session_keys))}) GROUP BY 1""",
        session_keys,
    ).fetchall()
    return {n: (tla, team or "") for n, tla, team in rows}


def _grid(con, race_sk: int | None, quali_sk: int | None) -> dict[int, int]:
    """Actual starting order when the race has data (first position reading per car, which
    includes grid penalties); otherwise the qualifying classification."""
    if race_sk is not None:
        rows = con.execute(
            """SELECT driver_number, arg_min(position, date::TIMESTAMPTZ) FROM raw_positions
               WHERE session_key = ? GROUP BY 1""",
            [race_sk],
        ).fetchall()
        if rows:
            return dict(rows)
    if quali_sk is None:
        return {}
    rows = con.execute(
        "SELECT driver_number, position FROM raw_results WHERE session_key = ? AND position IS NOT NULL",
        [quali_sk],
    ).fetchall()
    return dict(rows)


def penalised_grid(
    grid: dict[int, int], penalties: dict[int, int], pit_lane: set[int] = frozenset()
) -> dict[int, int]:
    """Apply grid-place penalties and pit-lane starts to a qualifying order (car -> slot).

    Penalised cars drop N places from their qualifying slot (never below the last non-pit-lane
    slot) and unpenalised cars move up to fill; pit-lane starters go to the back in qualifying
    order. The FIA applies penalties in the order offences were committed, which only matters
    when two penalised cars collide on the same slot (here: better qualifier first)."""
    racing = sorted((n for n in grid if n not in pit_lane), key=grid.__getitem__)
    last = len(racing)
    penalised = sorted(
        (n for n in racing if penalties.get(n, 0) > 0),
        key=lambda n: (grid[n] + penalties[n], grid[n]),
    )
    taken: dict[int, int] = {}  # slot -> car
    for n in penalised:
        slot = min(grid[n] + penalties[n], last)
        while slot in taken and slot < last:
            slot += 1
        if slot in taken:  # pushed past the back: the last free slot
            slot = max(set(range(1, last + 1)) - taken.keys())
        taken[slot] = n
    free = iter(sorted(set(range(1, last + 1)) - taken.keys()))
    for n in racing:
        if n not in penalised:
            taken[next(free)] = n
    out = {n: slot for slot, n in taken.items()}
    for i, n in enumerate(sorted((n for n in grid if n in pit_lane), key=grid.__getitem__)):
        out[n] = last + 1 + i
    return out


def _quali_deltas(con, quali_sk: int | None) -> dict[int, float]:
    if quali_sk is None:
        return {}
    best = dict(
        con.execute(
            """SELECT driver_number, min(lap_time) FROM laps
               WHERE session_key = ? AND lap_time IS NOT NULL GROUP BY 1""",
            [quali_sk],
        ).fetchall()
    )
    if not best:
        return {}
    pole = min(best.values())
    # Drop laps >7% off pole: aborted laps, not pace.
    return {n: t - pole for n, t in best.items() if t <= pole * 1.07}


def _clean_laps(con, session_key: int) -> list[CleanLap]:
    rows = con.execute(
        """SELECT driver_number::VARCHAR, stint, compound, lap_number, tyre_age, lap_time
           FROM clean_laps WHERE session_key = ? AND tyre_age IS NOT NULL""",
        [session_key],
    ).fetchall()
    return [CleanLap(*r) for r in rows]


MAX_LONG_RUN_DELTA = 3.0  # beyond this a run is a different programme (fuel, modes), not pace


def _long_run_deltas(
    sessions: dict[int, float], laps_by_session: dict[int, list[CleanLap]], deg: dict[str, float]
) -> dict[int, float]:
    """Per driver, the weighted mean of (their long-run pace − the median long-run pace of
    drivers on the SAME compound in the SAME session). Comparing within compound removes the
    compound pace gap; laps are corrected for tyre age with `deg`."""
    totals: dict[int, list[tuple[float, float]]] = {}
    for sk, weight in sessions.items():
        by_stint: dict[tuple[str, int], list[CleanLap]] = {}
        for lap in laps_by_session.get(sk, []):
            by_stint.setdefault((lap.driver, lap.stint), []).append(lap)
        # (compound -> driver -> best long-run pace on that compound)
        groups: dict[str, dict[int, float]] = {}
        for (driver, _), stint in by_stint.items():
            if len(stint) < MIN_LONG_RUN_LAPS:
                continue
            median = statistics.median(l.lap_time for l in stint)
            kept = [l for l in stint if l.lap_time <= median * 1.03]
            corrected = statistics.median(
                l.lap_time - deg.get(l.compound, 0.06) * l.tyre_age for l in kept
            )
            by_driver = groups.setdefault(stint[0].compound, {})
            by_driver[int(driver)] = min(corrected, by_driver.get(int(driver), corrected))
        for by_driver in groups.values():
            if len(by_driver) < 4:
                continue
            reference = statistics.median(by_driver.values())
            for d, pace in by_driver.items():
                delta = pace - reference
                if abs(delta) <= MAX_LONG_RUN_DELTA:
                    totals.setdefault(d, []).append((delta, weight))
    return {d: sum(v * w for v, w in vals) / sum(w for _, w in vals) for d, vals in totals.items()}


@dataclass(frozen=True)
class SeasonRates:
    sc_per_race: float
    vsc_per_race: float
    sc_laps: float
    vsc_laps: float
    dnf_prob: float
    races: int


PASS_PRIOR_SEASONS = 2  # shrinkage: pseudo-seasons at the median pass rate


@functools.cache
def _pass_rates_by_race() -> list[tuple[int, int, float]]:
    """(year, circuit_key, on-track passes per 1000 green car-laps) for every warehouse race:
    order swaps between consecutive green laps of cars that didn't pit on either lap (lap 1-2
    excluded). Counts depend on pace spread as well as the circuit, hence the shrinkage."""
    con = connect()
    try:
        races = con.execute(
            "SELECT session_key, year, circuit_key FROM races WHERE session_name = 'Race'"
        ).fetchall()
        out = []
        for sk, year, ck in races:
            rows = con.execute(
                """SELECT driver_number, lap_number, position,
                          coalesce(pit_in, false) OR coalesce(pit_out, false),
                          coalesce(neutralised, '') NOT IN ('', 'false', 'False')
                   FROM laps WHERE session_key = ? AND position IS NOT NULL""",
                [sk],
            ).fetchall()
            by_lap: dict[int, dict[int, tuple]] = {}
            for d, n, pos, pit, neut in rows:
                by_lap.setdefault(n, {})[d] = (pos, pit, neut)
            passes = car_laps = 0
            for n, cur in by_lap.items():
                prev = by_lap.get(n - 1)
                if n < 3 or not prev:
                    continue
                ok = [
                    d
                    for d in cur
                    if d in prev and not (cur[d][1] or prev[d][1] or cur[d][2] or prev[d][2])
                ]
                if len(ok) < 10:
                    continue
                car_laps += len(ok)
                for i, a in enumerate(ok):
                    for b in ok[i + 1 :]:
                        if (prev[a][0] - prev[b][0]) * (cur[a][0] - cur[b][0]) < 0:
                            passes += 1
            if car_laps and ck is not None:
                out.append((year, int(ck), 1000 * passes / car_laps))
        return out
    finally:
        con.close()


def circuit_pass_rel(circuit_key: int | None, year: int) -> float:
    """This circuit's pass rate relative to the median, from seasons before `year`, shrunk
    toward 1 (PASS_PRIOR_SEASONS pseudo-seasons at the median). 1.0 for new circuits."""
    rows = [(y, ck, r) for y, ck, r in _pass_rates_by_race() if y < year]
    if circuit_key is None or not rows:
        return 1.0
    median = statistics.median(r for _, _, r in rows)
    mine = [r / median for _, ck, r in rows if ck == int(circuit_key)]
    return (sum(mine) + PASS_PRIOR_SEASONS) / (len(mine) + PASS_PRIOR_SEASONS)


def _season_rates(con, year: int, exclude_meeting: int) -> SeasonRates:
    """SC / VSC frequency and length, and DNF probability, from the season's OTHER races."""
    races = [
        r[0]
        for r in con.execute(
            "SELECT session_key FROM races WHERE year = ? AND session_name = 'Race' AND meeting_key != ?",
            [year, exclude_meeting],
        ).fetchall()
    ]
    if not races:
        return SeasonRates(0.7, 1.2, 6.0, 1.4, 0.08, 0)
    marks = ",".join("?" * len(races))
    sc, vsc, sc_s, vsc_s = con.execute(
        f"""SELECT count(*) FILTER (kind = 'SC'), count(*) FILTER (kind = 'VSC'),
                   median(epoch("end"::TIMESTAMPTZ - start::TIMESTAMPTZ)) FILTER (kind = 'SC' AND "end" < '9999'),
                   median(epoch("end"::TIMESTAMPTZ - start::TIMESTAMPTZ)) FILTER (kind = 'VSC' AND "end" < '9999')
            FROM raw_neutralised WHERE session_key IN ({marks})""",
        races,
    ).fetchone()
    lap_s = (
        con.execute(
            f"SELECT median(lap_time) FROM clean_laps WHERE session_key IN ({marks})", races
        ).fetchone()[0]
        or 90.0
    )
    dnf, starters = con.execute(
        f"""SELECT count(*) FILTER (dnf), count(*) FILTER (NOT dns)
            FROM raw_results WHERE session_key IN ({marks})""",
        races,
    ).fetchone()
    return SeasonRates(
        sc_per_race=sc / len(races),
        vsc_per_race=vsc / len(races),
        sc_laps=(sc_s or 540) / lap_s,
        vsc_laps=(vsc_s or 120) / lap_s,
        dnf_prob=dnf / max(starters, 1),
        races=len(races),
    )


def _race_minus_quali(con, race_sk: int, quali_sk: int) -> dict[str, float]:
    """Per team: race pace gap to the field (median clean lap, tyre-age corrected) minus the
    qualifying gap to the field. Positive = worse in the race than qualifying suggested."""
    race = dict(
        con.execute(
            """SELECT l.driver_number, median(l.lap_time - 0.05 * l.tyre_age)
               FROM clean_laps l WHERE l.session_key = ? GROUP BY 1 HAVING count(*) >= 10""",
            [race_sk],
        ).fetchall()
    )
    quali = _quali_deltas(con, quali_sk)
    teams = dict(
        con.execute(
            "SELECT driver_number, team_name FROM raw_drivers WHERE session_key = ?", [race_sk]
        ).fetchall()
    )
    common = [d for d in race if d in quali and teams.get(d)]
    if len(common) < 8:
        return {}
    race_med = statistics.median(race[d] for d in common)
    quali_med = statistics.median(quali[d] for d in common)
    by_team: dict[str, list[float]] = {}
    for d in common:
        by_team.setdefault(teams[d], []).append((race[d] - race_med) - (quali[d] - quali_med))
    return {team: statistics.mean(v) for team, v in by_team.items()}


RACE_PACE_SHRINK = 2  # pseudo-races of "no gap" in the running mean
# races; None = plain season mean. 3 races: the front converges within a season (2026: top-two
# gap 0.53% -> 0.15%), and recency-weighting fixed the overrated early leaders.
RACE_PACE_HALF_LIFE: float | None = 3.0


def season_race_pace(
    con, year: int, before, half_life: float | None = None, by: str = "team"
) -> dict:
    """`by="team"` keys by team name (both cars averaged per race); `by="driver"` keys by car
    number (each driver's own race pace).

    Per team: race-pace gap to the fastest driver (% of a lap) over this season's earlier
    races, shrunk toward 0: sum / (races + RACE_PACE_SHRINK). Clean green laps only (tyre-age
    corrected), so SC/VSC luck doesn't enter."""
    races = con.execute(
        """SELECT session_key FROM races WHERE year = ? AND session_name = 'Race'
           AND date_start < ? ORDER BY date_start""",
        [year, before],
    ).fetchall()
    gaps: dict[str, list[float]] = {}
    for (race_sk,) in races:
        rows = con.execute(
            f"""SELECT {"d.team_name" if by == "team" else "l.driver_number"},
                      median(l.lap_time - 0.05 * l.tyre_age) AS pace
               FROM clean_laps l JOIN raw_drivers d
                 ON d.session_key = l.session_key AND d.driver_number = l.driver_number
               WHERE l.session_key = ? GROUP BY l.driver_number, d.team_name
               HAVING count(*) >= 10""",
            [race_sk],
        ).fetchall()
        if not rows:
            continue
        best = min(p for _, p in rows)
        by_team: dict[str, list[float]] = {}
        for team, pace in rows:
            if team:
                by_team.setdefault(team, []).append(100 * (pace - best) / best)
        for team, pcts in by_team.items():
            gaps.setdefault(team, []).append(statistics.mean(pcts))
    if half_life is None:
        return {team: sum(v) / (len(v) + RACE_PACE_SHRINK) for team, v in gaps.items()}
    # Recency-weighted: the field converges during a season (2026: the top-two gap fell from
    # 0.53% to 0.15% of a lap), so older races count less. Shrinkage uses the same weights.
    out = {}
    for team, v in gaps.items():
        w = [0.5 ** ((len(v) - 1 - i) / half_life) for i in range(len(v))]
        out[team] = sum(x * wi for x, wi in zip(v, w, strict=True)) / (sum(w) + RACE_PACE_SHRINK)
    return out


def season_form(con, year: int, before, max_abs: float = 1.5) -> dict[str, tuple[float, int]]:
    """Per team: mean race-vs-quali pace bias over this season's earlier races, and how many."""
    weekends = con.execute(
        """SELECT meeting_key,
                  any_value(session_key) FILTER (session_name = 'Race'),
                  any_value(session_key) FILTER (session_name = 'Qualifying')
           FROM races WHERE year = ? GROUP BY meeting_key
           HAVING min(date_start) < ?""",
        [year, before],
    ).fetchall()
    samples: dict[str, list[float]] = {}
    for _, race_sk, quali_sk in weekends:
        if race_sk is None or quali_sk is None:
            continue
        for team, bias in _race_minus_quali(con, race_sk, quali_sk).items():
            if abs(bias) <= max_abs:
                samples.setdefault(team, []).append(bias)
    return {team: (statistics.mean(v), len(v)) for team, v in samples.items()}


TEAMMATE_SHRINK = 3  # pseudo-qualifyings at "same pace as the teammate"
SEGMENT_BREAK_S = 240  # no lap started for this long = a break between Q1 / Q2 / Q3


def _segment_bests(con, quali_sk: int) -> dict[int, dict[int, float]]:
    """Per driver: best lap per qualifying segment (0 = Q1 ...). Segments are split at breaks in
    running, so teammates can be compared on the same track state (track evolution makes Q3
    laps faster than Q1 laps)."""
    rows = con.execute(
        """SELECT driver_number, epoch(lap_start), lap_time FROM laps
           WHERE session_key = ? AND lap_time IS NOT NULL AND lap_start IS NOT NULL
           ORDER BY 2""",
        [quali_sk],
    ).fetchall()
    out: dict[int, dict[int, float]] = {}
    seg, last = 0, None
    for d, t, lt in rows:
        if last is not None and t - last > SEGMENT_BREAK_S:
            seg += 1
        last = t
        best = out.setdefault(d, {})
        best[seg] = min(lt, best.get(seg, lt))
    return out


def _teammate_gap(bests: dict[int, dict[int, float]], d: int, mate: int) -> float | None:
    """d's best minus the teammate's in the latest segment both set a time in (None if
    none; laps over 7% off the segment's teammate time are aborted laps, ignored)."""
    common = sorted(set(bests.get(d, {})) & set(bests.get(mate, {})))
    for seg in reversed(common):
        a, b = bests[d][seg], bests[mate][seg]
        if a <= b * 1.07 and b <= a * 1.07:
            return a - b
    return None


def teammate_offsets(con, year: int, before) -> dict[int, float]:
    """Per driver: median qualifying gap to their teammate (s, + = slower), segment-matched,
    over this season's earlier qualifyings, shrunk toward 0."""
    rows = con.execute(
        """SELECT session_key FROM races WHERE year = ? AND session_name = 'Qualifying'
           AND date_start < ? ORDER BY date_start""",
        [year, before],
    ).fetchall()
    gaps: dict[int, list[float]] = {}
    for (sk,) in rows:
        bests = _segment_bests(con, sk)
        teams = dict(
            con.execute(
                "SELECT driver_number, team_name FROM raw_drivers WHERE session_key = ?", [sk]
            ).fetchall()
        )
        by_team: dict[str, list[int]] = {}
        for d in bests:
            if teams.get(d):
                by_team.setdefault(teams[d], []).append(d)
        for pair in by_team.values():
            if len(pair) == 2:
                a, b = pair
                g = _teammate_gap(bests, a, b)
                if g is not None:
                    gaps.setdefault(a, []).append(g)
                    gaps.setdefault(b, []).append(-g)
    return {d: statistics.median(v) * len(v) / (len(v) + TEAMMATE_SHRINK) for d, v in gaps.items()}


def build_inputs(
    con,
    year: int,
    place: str,
    laps: int | None = None,
    long_run_level: str = "driver",
    sprint_weight: float = SPRINT_WEIGHT,
    grid: dict[int, int] | None = None,
    compromised_threshold: float | None = None,
) -> WeekendInputs:
    """`compromised_threshold` (s): a driver whose qualifying lap is this much slower than
    teammate + usual gap (or who set no time) gets the teammate-based estimate instead; None =
    use qualifying as set.

    `long_run_level`: "driver" (each driver's own long runs) or "team" (both cars' average,
    less sensitive to one driver's fuel load / programme). `sprint_weight`: how much a sprint's
    race pace counts relative to a practice long run. `grid` (car number -> slot) overrides the
    starting grid, e.g. the official one from the live feed or `penalised_grid`."""
    meeting_key, sessions = weekend_sessions(con, year, place)
    race_sk = sessions.get("Race")
    quali_sk = sessions.get("Qualifying")
    notes: list[str] = []

    pace_sessions = {
        sessions[s]: 1.0 for s in ("Practice 1", "Practice 2", "Practice 3") if s in sessions
    }
    if "Sprint" in sessions:
        pace_sessions[sessions["Sprint"]] = sprint_weight
    laps_by_session = {sk: _clean_laps(con, sk) for sk in pace_sessions}

    # Degradation from this weekend's running (practice + sprint), per compound.
    pooled = [lap for laps_ in laps_by_session.values() for lap in laps_]
    model = fit_tyre_model(pooled, method="stint") if pooled else None
    deg = dict(DEFAULT_DEG)
    for compound, fit in (model.compounds if model else {}).items():
        if fit.n_stints >= 3 and 0 <= fit.deg_s_per_lap <= 0.3:
            deg[compound] = fit.deg_s_per_lap
    missing = [c for c in DEFAULT_DEG if not model or c not in model.compounds]
    if missing:
        notes.append(f"default degradation used for {', '.join(missing)}")

    grid = grid or _grid(con, race_sk, quali_sk)
    quali = _quali_deltas(con, quali_sk)
    long_run = _long_run_deltas(pace_sessions, laps_by_session, deg)
    names = _drivers(con, [sk for sk in sessions.values()])
    if long_run_level == "team":
        by_team: dict[str, list[float]] = {}
        for n, delta in long_run.items():
            by_team.setdefault(names.get(n, ("", ""))[1], []).append(delta)
        long_run = {
            n: statistics.mean(by_team[names.get(n, ("", ""))[1]])
            for n in long_run
            if names.get(n, ("", ""))[1]
        }
    weekend_start = con.execute(
        "SELECT min(date_start) FROM races WHERE meeting_key = ?", [meeting_key]
    ).fetchone()[0]
    form = season_form(con, year, weekend_start)
    race_pace = season_race_pace(
        con, year, weekend_start, half_life=RACE_PACE_HALF_LIFE, by="driver"
    )
    pole = (
        con.execute("SELECT min(lap_time) FROM laps WHERE session_key = ?", [quali_sk]).fetchone()[
            0
        ]
        if quali_sk
        else None
    ) or 90.0
    compromised: set[int] = set()
    if compromised_threshold is not None and quali:
        offsets = teammate_offsets(con, year, weekend_start)
        teams: dict[str, list[int]] = {}
        for n in grid:
            if names.get(n, ("", ""))[1]:
                teams.setdefault(names[n][1], []).append(n)
        original = dict(quali)
        bests = _segment_bests(con, quali_sk)
        for pair in teams.values():
            if len(pair) != 2:
                continue
            for d, mate in (pair, pair[::-1]):
                if mate not in original:
                    continue
                gap = _teammate_gap(bests, d, mate)  # same segment, so same track state
                usual = offsets.get(d, 0.0)
                if (d not in original and d in bests) or (
                    gap is not None and gap - usual > compromised_threshold
                ):
                    quali[d] = original[mate] + usual
                    compromised.add(d)
        if compromised:
            tlas = ", ".join(names.get(d, (str(d), ""))[0] for d in sorted(compromised))
            notes.append(f"qualifying looked compromised for {tlas}: teammate-based pace used")
    drivers = []
    for n, pos in sorted(grid.items(), key=lambda kv: kv[1]):
        tla, team = names.get(n, (str(n), ""))
        bias, n_races = form.get(team, (0.0, 0))
        drivers.append(
            DriverInput(
                n,
                tla,
                team,
                pos,
                quali.get(n),
                long_run.get(n),
                bias,
                n_races,
                race_pace_s=race_pace.get(n, 0.0) * pole / 100,
                quali_compromised=n in compromised,
            )
        )

    if laps is None:
        if race_sk is None:
            raise ValueError("race laps unknown before the race: pass laps=")
        laps = con.execute(
            "SELECT max(lap_number) FROM laps WHERE session_key = ?", [race_sk]
        ).fetchone()[0]

    circuit_key = con.execute(
        "SELECT any_value(circuit_key) FROM races WHERE meeting_key = ?", [meeting_key]
    ).fetchone()[0]
    pass_rel = circuit_pass_rel(circuit_key, year)
    circuit = fetch_circuit(int(circuit_key), year) if circuit_key else None
    if circuit and circuit.pit_loss:
        pit_green, pit_sc, pit_vsc = (
            circuit.pit_loss.green,
            circuit.pit_loss.safety_car,
            circuit.pit_loss.vsc,
        )
    else:
        pit_green, pit_sc, pit_vsc = DEFAULT_PIT_LOSS
        notes.append("default pit loss (no measured circuit data)")

    rates = _season_rates(con, year, meeting_key)
    notes.append(
        f"{rates.sc_per_race:.2f} SC + {rates.vsc_per_race:.2f} VSC per race "
        f"from {rates.races} other {year} races"
    )
    meeting = con.execute(
        "SELECT any_value(location) FROM races WHERE meeting_key = ?", [meeting_key]
    ).fetchone()[0]
    return WeekendInputs(
        meeting,
        year,
        race_sk,
        int(laps),
        drivers,
        deg,
        pit_green,
        pit_sc,
        rates.sc_per_race,
        rates.dnf_prob,
        notes,
        vsc_per_race=rates.vsc_per_race,
        pit_loss_vsc=pit_vsc,
        sc_laps=rates.sc_laps,
        vsc_laps=rates.vsc_laps,
        pass_rel=pass_rel,
    )
