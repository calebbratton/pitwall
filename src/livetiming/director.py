"""The Director: which battle to watch, from the live timing (no video, no F1 TV automation).

Each completed lap, every close pair (chaser within battles.MAX_GAP_S of the car ahead) is scored
with the learned battle models (src/sim/battles.py): P(on-track pass within 5 laps) and P(within
1 s within 3 laps). The top battle becomes "watch <chaser>'s onboard"; a `battle` alert goes out
when the top battle changes and is likely enough. The viewer switches onboards themselves.

Event: {"type": "director", "lap", "watch": {...} | None, "battles": [{chaser, ahead, gap,
closing, p_pass, p_close}, ...]}.
"""

import itertools
import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Notify vs actual overtake (backtest(), 2025-26, 38 races, models fitted without each race):
# threshold 0.6 -> 7.5 alerts/race, 75% followed by that pass on track within 5 laps (median 1
# lap's notice), 10% of all on-track passes flagged (one top battle at a time, and the total
# includes lapping and recovery drives). 0.5 -> 10.8/race, 73%; 0.4 -> 14.6/race, 67%.
ALERT_P_PASS = 0.6
TOP = 3


RESOLVE_LAPS = 5  # an alerted battle "worked" if the chaser passes on track within this many laps


class Director:
    def __init__(self, battle_models=None, alert_p_pass: float | None = None) -> None:
        """`battle_models`: (pass, close) models - the live server fits them on every race; the
        backtest passes ones fitted without the race being replayed."""
        self._gaps: dict[int, dict[tuple[str, str], float]] = {}  # lap -> (chaser, ahead) -> gap
        self._watching: tuple[str, str] | None = None
        self._models = battle_models
        self.alert_p_pass = ALERT_P_PASS if alert_p_pass is None else alert_p_pass
        self.open_alerts: dict[str, dict[str, Any]] = {}  # id -> alert awaiting its outcome
        self._pits: dict[str, int] = {}  # car number -> pit stops at the last check

    def _resolve(self, snap) -> list[dict[str, Any]]:
        """Close alerted battles: a pass on track (no stop by either car) within RESOLVE_LAPS
        laps -> "worked"; a stop by either car, or time up -> "failed" (no pass)."""
        lap = snap.current_lap or 0
        by_tla = {d.tla: d for d in snap.drivers}
        events = []
        for aid, alert in list(self.open_alerts.items()):
            chaser, ahead = by_tla.get(alert["drivers"][0]), by_tla.get(alert["drivers"][1])
            if chaser is None or ahead is None or chaser.retired or ahead.retired:
                outcome, note = "failed", "a car retired"
            elif chaser.pit_stops != alert["_stops"][0] or ahead.pit_stops != alert["_stops"][1]:
                outcome, note = "failed", "a pit stop ended the battle"
            elif (chaser.position or 99) < (ahead.position or 0):
                outcome, note = "worked", f"{chaser.tla} passed {ahead.tla} by lap {lap}"
            elif lap - alert["lap"] >= RESOLVE_LAPS:
                outcome, note = "failed", f"no pass within {RESOLVE_LAPS} laps"
            else:
                continue
            del self.open_alerts[aid]
            done = {k: v for k, v in alert.items() if not k.startswith("_")}
            done |= {
                "status": "resolved",
                "outcome": outcome,
                "detail": f"{alert['detail']} {note[0].upper() + note[1:]}.",
            }
            events.append(done)
        return events

    def on_lap(self, monitor) -> list[dict[str, Any]]:
        from src.sim.battles import MAX_GAP_S, models, pair_features
        from src.sim.inputs import circuit_pass_rel

        snap = monitor.snapshot()
        lap = snap.current_lap or 0
        if not snap.is_race or not snap.total_laps or lap < 4:
            return []
        resolved = self._resolve(snap)
        running = sorted(
            (d for d in snap.drivers if d.position and not d.retired and not d.in_pit),
            key=lambda d: d.position,
        )
        gaps = {}
        pairs = []
        for ahead, chaser in itertools.pairwise(running):
            gap = chaser.interval_s
            if gap is None or not (0 < gap <= MAX_GAP_S) or chaser.laps_down:
                continue
            gaps[(chaser.tla, ahead.tla)] = gap
            pairs.append((chaser, ahead, gap))
        self._gaps[lap] = gaps
        self._gaps = {k: v for k, v in self._gaps.items() if k >= lap - 4}
        if snap.track_status != "GREEN" or not pairs:
            return [*resolved, {"type": "director", "lap": lap, "watch": None, "battles": []}]

        def laps(number: str) -> list[float]:
            return [
                r.time_s
                for r in monitor.laps.get(number, [])[-3:]
                if r.time_s and not (r.pit_in or r.pit_out or r.neutralised)
            ]

        rel = getattr(self, "_pass_rel", None)
        if rel is None:
            rel = self._pass_rel = _circuit_rel(snap, circuit_pass_rel)
        X = np.array(
            [
                pair_features(
                    gap,
                    self._gaps.get(lap - 3, {}).get((c.tla, a.tla)),
                    laps(c.number),
                    laps(a.number),
                    c.tyre_age_laps or 0,
                    a.tyre_age_laps or 0,
                    c.compound,
                    a.compound,
                    rel,
                    a.position,
                    lap / snap.total_laps,
                    snap.year or 0,
                )
                for c, a, gap in pairs
            ],
            dtype=float,
        )
        try:
            pass_model, close_model = self._models or models()
        except Exception:  # noqa: BLE001 — no warehouse / dataset: no director
            log.warning("battle models unavailable")
            return []
        p_pass, p_close = pass_model.predict(X), close_model.predict(X)
        battles = sorted(
            (
                {
                    "chaser": c.tla,
                    "ahead": a.tla,
                    "_stops": (c.pit_stops, a.pit_stops),
                    "position": a.position,
                    "gap": round(gap, 2),
                    "closing": round(x[1], 2),
                    "p_pass": round(float(pp), 2),
                    "p_close": round(float(pc), 2),
                }
                for (c, a, gap), x, pp, pc in zip(pairs, X, p_pass, p_close, strict=True)
            ),
            key=lambda b: b["p_pass"] + 0.5 * b["p_close"],
            reverse=True,
        )[:TOP]
        public = [{k: v for k, v in b.items() if not k.startswith("_")} for b in battles]
        top = battles[0]
        events: list[dict[str, Any]] = [
            *resolved,
            {"type": "director", "lap": lap, "watch": public[0], "battles": public},
        ]
        key = (top["chaser"], top["ahead"])
        if key != self._watching and top["p_pass"] >= self.alert_p_pass:
            self._watching = key
            closing = (
                f", closing {top['closing']:.1f} s over 3 laps" if top["closing"] > 0.05 else ""
            )
            events.append(
                {
                    "type": "alert",
                    "id": f"battle-{top['chaser']}-{top['ahead']}-{lap}",
                    "kind": "battle",
                    "lap": lap,
                    "headline": f"Battle for P{top['position']}: watch {top['chaser']}'s onboard",
                    "detail": f"{top['chaser']} is {top['gap']:.1f} s behind {top['ahead']}{closing}. "
                    f"Within 1 s in the next 3 laps: {top['p_close']:.0%}; a pass in 5 laps: "
                    f"{top['p_pass']:.0%}.",
                    "drivers": [top["chaser"], top["ahead"]],
                    "p": top["p_pass"],
                    "status": "live",
                    "outcome": None,
                }
            )
            self.open_alerts[events[-1]["id"]] = {**events[-1], "_stops": top["_stops"]}
        return events


def _circuit_rel(snap, circuit_pass_rel) -> float:
    try:
        from src.warehouse.queries import connect

        con = connect()
        try:
            row = con.execute(
                "SELECT circuit_key FROM races WHERE year = ? AND location ILIKE ? LIMIT 1",
                [snap.year, snap.location],
            ).fetchone()
        finally:
            con.close()
        return circuit_pass_rel(row[0] if row else None, snap.year or 0)
    except Exception:  # noqa: BLE001
        return 1.0


def backtest(years: tuple[int, ...] = (2025, 2026), thresholds=(0.3, 0.4, 0.5, 0.6)) -> dict:
    """Notify vs actual overtakes. Each archived race is replayed with battle models fitted
    without it; every lap's top battle and running order are recorded, then each alert threshold
    is applied offline:
      precision  alerts followed by that pass on track within RESOLVE_LAPS laps
      recall     on-track passes (no stop by either car, green lap) preceded by an alert for
                 that pair in the RESOLVE_LAPS laps before
      lead       laps from alert to pass, for alerts that worked"""
    from src.livetiming.archive import ArchiveSession, list_sessions
    from src.livetiming.monitor import STRATEGY_TOPICS, RaceMonitor
    from src.sim.battles import FEATURES, load
    from src.sim.pit_hazard import Blend
    from src.warehouse.queries import connect

    rows = load()
    con = connect()
    locations = dict(
        con.execute(
            "SELECT strftime(date_start, '%Y-%m-%d'), location FROM races WHERE session_name = 'Race'"
        ).fetchall()
    )
    races = []
    for year in years:
        for race in list_sessions(year):
            loc = locations.get(race["date"])
            if loc is None:
                continue
            keep = np.array([rows.races[i] != (year, loc) for i in rows.race])
            cols = list(range(len(FEATURES)))
            fitted = (
                Blend(rows.X[keep][:, cols], rows.y_pass[keep]),
                Blend(rows.X[keep][:, cols], rows.y_close[keep]),
            )
            director = Director(battle_models=fitted, alert_p_pass=2.0)  # record only, no alerts
            monitor = RaceMonitor()
            laps: dict[int, dict] = {}
            last = None
            for m in ArchiveSession(race["path"]).messages(STRATEGY_TOPICS):
                monitor.feed(m)
                if m.topic != "LapCount":
                    continue
                snap = monitor.snapshot()
                lap = snap.current_lap
                if not lap or lap == last:
                    continue
                last = lap
                pick = next((e for e in director.on_lap(monitor) if e["type"] == "director"), None)
                laps[lap] = {
                    "green": snap.track_status == "GREEN",
                    "top": pick["watch"] if pick else None,
                    "cars": {
                        d.tla: (d.position, d.pit_stops, d.retired)
                        for d in snap.drivers
                        if d.position
                    },
                }
            races.append((year, race["meeting"], laps))
            print(f"  {year} {race['meeting']:28} {len(laps)} laps", flush=True)

    def passes(laps):
        out = []
        for n in sorted(laps):
            prev, cur = laps.get(n - 1), laps[n]
            if not prev or not cur["green"] or not prev["green"]:
                continue
            for x, (px, sx, rx) in cur["cars"].items():
                for y, (py, sy, ry) in cur["cars"].items():
                    if x == y or rx or ry or px >= py:
                        continue
                    before = prev["cars"].get(x), prev["cars"].get(y)
                    if not all(before):
                        continue
                    (bx, bsx, _), (by, bsy, _) = before
                    if bx > by and bsx == sx and bsy == sy:
                        out.append((n, x, y))
            # x passed y on lap n (was behind, now ahead, no stops)
        return out

    results = {}
    for thr in thresholds:
        alerts, worked, leads, flagged, total_passes = 0, 0, [], 0, 0
        for _, _, laps in races:
            truth = passes(laps)
            total_passes += len(truth)
            watching, issued = None, []
            for n in sorted(laps):
                top = laps[n]["top"]
                if not top:
                    continue
                key = (top["chaser"], top["ahead"])
                if key != watching and top["p_pass"] >= thr:
                    watching = key
                    issued.append((n, *key))
            alerts += len(issued)
            for n, c, a in issued:
                hit = [p for p in truth if p[1] == c and p[2] == a and 0 < p[0] - n <= RESOLVE_LAPS]
                if hit:
                    worked += 1
                    leads.append(hit[0][0] - n)
            for n, x, y in truth:
                if any(c == x and a == y and 0 < n - k <= RESOLVE_LAPS for k, c, a in issued):
                    flagged += 1
        results[thr] = {
            "alerts_per_race": round(alerts / max(len(races), 1), 1),
            "precision": round(worked / alerts, 3) if alerts else None,
            "recall": round(flagged / total_passes, 3) if total_passes else None,
            "median_lead_laps": float(np.median(leads)) if leads else None,
            "passes_per_race": round(total_passes / max(len(races), 1), 1),
        }
    for thr, r in results.items():
        print(f"alert at p_pass >= {thr}: {r}")
    return results


if __name__ == "__main__":
    backtest()
