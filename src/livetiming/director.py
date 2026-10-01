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

# Alert when a pass is more likely than not (battles.py, leave-one-race-out: predicted 0.67 ->
# observed 0.66 in that band). "Within 1 s soon" is too common (~half of close pairs) to alert on.
ALERT_P_PASS = 0.5
TOP = 3


class Director:
    def __init__(self) -> None:
        self._gaps: dict[int, dict[tuple[str, str], float]] = {}  # lap -> (chaser, ahead) -> gap
        self._watching: tuple[str, str] | None = None

    def on_lap(self, monitor) -> list[dict[str, Any]]:
        from src.sim.battles import MAX_GAP_S, models, pair_features
        from src.sim.inputs import circuit_pass_rel

        snap = monitor.snapshot()
        lap = snap.current_lap or 0
        if not snap.is_race or not snap.total_laps or lap < 4:
            return []
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
            return [{"type": "director", "lap": lap, "watch": None, "battles": []}]

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
            pass_model, close_model = models()
        except Exception:  # noqa: BLE001 — no warehouse / dataset: no director
            log.warning("battle models unavailable")
            return []
        p_pass, p_close = pass_model.predict(X), close_model.predict(X)
        battles = sorted(
            (
                {
                    "chaser": c.tla,
                    "ahead": a.tla,
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
        top = battles[0]
        events: list[dict[str, Any]] = [
            {"type": "director", "lap": lap, "watch": top, "battles": battles}
        ]
        key = (top["chaser"], top["ahead"])
        if key != self._watching and top["p_pass"] >= ALERT_P_PASS:
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
                    "status": "resolved",
                    "outcome": None,
                }
            )
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
