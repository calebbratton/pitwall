"""Strategy alert stream: a short list of calls at the moments that matter (push, don't pull).

Alerts are event-driven: the engine runs when a lap completes or the track status changes, not
on a timer. Each alert is a situation with an id; it fires once, updates in place, and is
resolved afterwards ("it worked: +1"). The simulator decides; text is templated (no LLM), so an
alert can't describe something that isn't in the data.

Kinds (v1):
  sc_call   a safety car / VSC: who should pit and who should stay out (from strategy.pit_calls)
  undercut  car A, close behind B, gains B's position by pitting now (B covering a lap later)
            vs letting B stop first. Fires when P(A ahead at the flag) >= UNDERCUT_MIN_P and
            beats the alternative by UNDERCUT_MIN_GAIN, two laps running.

Event: {"type": "alert", "id", "kind", "lap", "headline", "detail", "drivers", "p", "status":
"live" | "resolved", "outcome": None | "worked" | "failed" | "not taken"}.
"""

import itertools
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.livetiming.monitor import RaceMonitor
from src.livetiming.snapshot import RaceSnapshot
from src.livetiming.strategy import PitCallReport
from src.sim.inrace import NotEnoughData, RaceState, StopPlan, race_state, simulate_from

log = logging.getLogger(__name__)

UNDERCUT_MAX_GAP_S = 4.0  # beyond this a fresh-tyre out-lap can't close it
UNDERCUT_MIN_P = 0.6
UNDERCUT_MIN_GAIN = 0.2  # vs letting the car ahead stop first
UNDERCUT_MIN_LAPS_LEFT = 8
PERSIST_LAPS = 2  # the condition must hold on consecutive laps before an alert fires
MAX_PAIRS = 6  # evaluated per lap, front of the field first
SIMS = 600
RESOLVE_LAPS_AFTER_STOP = 2  # judge an undercut this many laps after the later of the two stops
GIVE_UP_LAPS = 8  # an alert whose cars never both stopped is closed as "not taken"


@dataclass
class Alert:
    id: str
    kind: str
    lap: int | None
    headline: str
    detail: str
    drivers: list[str]
    p: float | None = None
    status: str = "live"
    outcome: str | None = None
    # bookkeeping for resolution (not sent)
    numbers: list[str] = field(default_factory=list)
    stops_at_alert: dict[str, int] = field(default_factory=dict)

    def event(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("numbers")
        d.pop("stops_at_alert")
        return {"type": "alert", **d}


def sc_call_alert(report: PitCallReport, top: int = 10) -> Alert | None:
    """One alert summarising the SC/VSC calls that matter: the top cars' PIT vs STAY OUT."""
    calls = [c for c in report.calls if c.position and c.position <= top]
    pit = [c.tla for c in calls if c.call == "PIT"]
    stay = [c.tla for c in calls if c.call == "STAY OUT"]
    if not pit and not stay:
        return None
    kind = "SC" if report.track_status == "SAFETY_CAR" else "VSC"
    parts = []
    if pit:
        parts.append(f"pit: {', '.join(pit)}")
    if stay:
        parts.append(f"stay out: {', '.join(stay)}")
    return Alert(
        id=f"sc-{report.lap}",
        kind="sc_call",
        lap=report.lap,
        headline=f"{kind}: " + "; ".join(parts),
        detail=(
            f"A stop costs {report.pit_loss_now_s:.0f} s now vs {report.green_pit_loss_s:.0f} s "
            "under green. " + " ".join(c.reason for c in calls[:3] if c.reason)
        ),
        drivers=pit + stay,
        status="resolved",  # informational: the pit-call panel carries the per-car reasons
    )


class AlertEngine:
    """Per live session. `on_lap` after each completed leader lap, `on_pit_calls` at SC/VSC."""

    def __init__(self, curves: dict | None = None) -> None:
        """`curves`: season tyre-age curves (fitted without this race in backtests); loaded for
        the session's year on first use otherwise. They carry the fresh-tyre advantage that
        makes an undercut work."""
        self.curves = curves
        self.alerts: dict[str, Alert] = {}
        self._streak: dict[str, int] = {}  # candidate id -> consecutive laps the condition held

    # --- safety car ----------------------------------------------------------------------------

    def on_pit_calls(self, report: PitCallReport) -> list[dict[str, Any]]:
        alert = sc_call_alert(report)
        if alert is None or alert.id in self.alerts:
            return []
        self.alerts[alert.id] = alert
        return [alert.event()]

    # --- every lap -----------------------------------------------------------------------------

    def on_lap(self, monitor: RaceMonitor, pit_loss: tuple[float, float]) -> list[dict[str, Any]]:
        snap = monitor.snapshot()
        events = self._resolve(snap, monitor)
        if snap.track_status != "GREEN" or not snap.is_race:
            self._streak.clear()
            return events
        if self.curves is None and snap.year:
            from src.models.tyre_curves import season_curves

            self.curves = season_curves(snap.year)
        try:
            state = race_state(
                monitor, snap, pit_loss=pit_loss, curves=self.curves, age_curves=bool(self.curves)
            )
        except NotEnoughData:
            return events
        if state.laps_remaining < UNDERCUT_MIN_LAPS_LEFT:
            return events
        seen = set()
        for a, b in self._pairs(state)[:MAX_PAIRS]:
            cid = f"undercut-{a.tla}-{b.tla}"
            seen.add(cid)
            p_now, p_base = self._undercut(state, a.number, b.number)
            if p_now >= UNDERCUT_MIN_P and p_now - p_base >= UNDERCUT_MIN_GAIN:
                self._streak[cid] = self._streak.get(cid, 0) + 1
            else:
                self._streak[cid] = 0
            live = self.alerts.get(cid)
            if live and live.status == "live":
                if abs((live.p or 0) - p_now) >= 0.2:  # update in place on a big move
                    live.p = round(p_now, 2)
                    live.detail = _undercut_detail(a.tla, b.tla, p_now, p_base, state)
                    events.append(live.event())
                continue
            if self._streak[cid] >= PERSIST_LAPS and not (live and live.status == "live"):
                alert = Alert(
                    id=f"{cid}-{state.lap}",
                    kind="undercut",
                    lap=state.lap,
                    headline=f"{a.tla}'s undercut on {b.tla} is on",
                    detail=_undercut_detail(a.tla, b.tla, p_now, p_base, state),
                    drivers=[a.tla, b.tla],
                    p=round(p_now, 2),
                    numbers=[a.number, b.number],
                    stops_at_alert={a.number: a.stops, b.number: b.stops},
                )
                self.alerts[cid] = alert
                events.append(alert.event())
        for cid in list(self._streak):
            if cid not in seen:
                self._streak[cid] = 0
        return events

    @staticmethod
    def _pairs(state: RaceState):
        """(chaser, car ahead) on the lead lap, close enough, both still needing a stop."""
        cars = [c for c in state.cars if c.gap_s < 900]
        out = []
        for b, a in itertools.pairwise(cars):
            if a.gap_s - b.gap_s > UNDERCUT_MAX_GAP_S:
                continue

            # The undercut battle is over the mandatory stop: both still owe a compound. (Tyre
            # life alone isn't a reliable "will stop": 2026 cars routinely ran past the season's
            # proven life and finished on those tyres - backtest: 24 of 25 such alerts untaken.)
            if a.owes_compound and b.owes_compound:
                out.append((a, b))
        return out

    @staticmethod
    def _undercut(state: RaceState, a: str, b: str) -> tuple[float, float]:
        """P(a finishes ahead of b): a pits now and b covers next lap, vs b first then a."""
        idx = {c.number: i for i, c in enumerate(state.cars)}
        results = []
        for plan in (
            {a: StopPlan(1), b: StopPlan(2)},
            {b: StopPlan(1), a: StopPlan(2)},
        ):
            pos = simulate_from(state, sims=SIMS, seed=5, plan=plan, return_positions=True)[
                "positions"
            ]
            results.append(float((pos[:, idx[a]] < pos[:, idx[b]]).mean()))
        return results[0], results[1]

    # --- resolution ----------------------------------------------------------------------------

    def _resolve(self, snap: RaceSnapshot, monitor: RaceMonitor) -> list[dict[str, Any]]:
        events = []
        lap = snap.current_lap or 0
        for cid, alert in list(self.alerts.items()):
            if alert.kind != "undercut" or alert.status != "live":
                continue
            a_no, b_no = alert.numbers
            stops = {d.number: d.pit_stops for d in snap.drivers}
            a_stop = _stop_lap_after(monitor, a_no, alert.lap or 0)
            b_stop = _stop_lap_after(monitor, b_no, alert.lap or 0)
            a, b = alert.drivers
            if a_stop is not None and b_stop is not None:
                if lap < max(a_stop, b_stop) + RESOLVE_LAPS_AFTER_STOP:
                    continue
                pos = {d.number: d.position for d in snap.drivers}
                taken = a_stop <= b_stop and a_stop <= (alert.lap or 0) + 2
                if not taken:
                    alert.outcome = "not taken"
                    alert.detail += f" {a} didn't stop first."
                elif (pos.get(a_no) or 99) < (pos.get(b_no) or 99):
                    alert.outcome = "worked"
                    alert.detail += f" It worked: {a} is ahead of {b} after both stops."
                else:
                    alert.outcome = "failed"
                    alert.detail += f" It didn't work: {b} stayed ahead."
                alert.status = "resolved"
                events.append(alert.event())
            elif lap >= (alert.lap or 0) + GIVE_UP_LAPS or stops.get(a_no) is None:
                alert.status, alert.outcome = "resolved", "not taken"
                events.append(alert.event())
        return events


def _stop_lap_after(monitor: RaceMonitor, number: str, lap: int) -> int | None:
    for r in monitor.laps.get(number, []):
        if r.pit_in and r.lap >= lap:
            return r.lap
    return None


def _undercut_detail(a: str, b: str, p_now: float, p_base: float, state: RaceState) -> str:
    return (
        f"If {a} pits now and {b} covers next lap, {a} finishes ahead {p_now:.0%} of the time "
        f"(vs {p_base:.0%} if {b} stops first). Pit loss {state.pit_loss_green:.0f} s, "
        f"{state.laps_remaining} laps to go."
    )


def hit_rate(alerts: list[Alert]) -> dict[str, float | int]:
    taken = [a for a in alerts if a.kind == "undercut" and a.outcome in ("worked", "failed")]
    return {
        "alerts": sum(a.kind == "undercut" for a in alerts),
        "taken": len(taken),
        "worked": sum(a.outcome == "worked" for a in taken),
        "hit_rate": float(np.mean([a.outcome == "worked" for a in taken]))
        if taken
        else float("nan"),
    }


def backtest(year: int = 2026, quiet: bool = False) -> dict:
    """Replay a season's races through the engine: alerts per race and the undercut hit rate
    (of alerts the team acted on within 2 laps, how often the chaser ended up ahead)."""
    from src.livetiming.archive import ArchiveSession, list_sessions
    from src.livetiming.monitor import STRATEGY_TOPICS, _circuit_for
    from src.livetiming.strategy import pit_calls
    from src.models.tyre_curves import fit_curves
    from src.sim.inrace_backtest import _race_keys
    from src.warehouse.queries import connect

    con = connect()
    keys = _race_keys(con, year)
    every: list[Alert] = []
    per_race = {}
    for race in list_sessions(year):
        race_sk = keys.get(race["date"])
        messages = list(ArchiveSession(race["path"]).messages(STRATEGY_TOPICS))
        circuit = _circuit_for(messages)
        loss = (
            (circuit.pit_loss.green, circuit.pit_loss.safety_car)
            if circuit and circuit.pit_loss
            else (22.0, 13.5)
        )
        monitor = RaceMonitor(pit_loss=circuit.pit_loss if circuit else None)
        # Curves fitted without this race: no hindsight about its tyres.
        engine = AlertEngine(curves=fit_curves(con, year, exclude=(race_sk,)) or {})
        last_lap = None
        for m in messages:
            for e in monitor.feed(m):
                if e["type"] == "pit_calls":
                    snap = monitor.snapshot()
                    engine.on_pit_calls(pit_calls(snap, monitor.pit_loss))
            if m.topic == "LapCount":
                lap = monitor.snapshot().current_lap
                if lap and lap != last_lap:
                    last_lap = lap
                    engine.on_lap(monitor, loss)
        alerts = list(engine.alerts.values())
        every += alerts
        per_race[race["meeting"]] = {
            "sc_calls": sum(a.kind == "sc_call" for a in alerts),
            **hit_rate(alerts),
        }
        if not quiet:
            r = per_race[race["meeting"]]
            print(
                f"{race['meeting']:28} sc {r['sc_calls']}  undercut alerts {r['alerts']:2}  "
                f"taken {r['taken']}  worked {r['worked']}"
            )
            for a in alerts:
                if a.kind == "undercut":
                    print(f"    L{a.lap} {a.headline} ({a.p:.0%}) -> {a.outcome}")
    total = hit_rate(every)
    if not quiet:
        print("\nall:", total)
    return {"per_race": per_race, "total": total}


def undercut_attempts(year: int = 2026, quiet: bool = False) -> dict:
    """Calibration on real attempts: every green-flag first stop where a car pitted while within
    UNDERCUT_MAX_GAP_S behind the car ahead, which hadn't stopped yet. The engine's P(chaser
    ahead) is taken the lap before (curves fitted without that race); the outcome is the order
    RESOLVE_LAPS_AFTER_STOP laps after the car ahead stopped."""
    from src.livetiming.archive import ArchiveSession, list_sessions
    from src.livetiming.monitor import STRATEGY_TOPICS, _circuit_for
    from src.models.tyre_curves import fit_curves
    from src.sim.inrace_backtest import _race_keys
    from src.warehouse.queries import connect

    con = connect()
    keys = _race_keys(con, year)
    rows = []  # (race, chaser, ahead, lap, p_now, p_base, worked)
    for race in list_sessions(year):
        messages = list(ArchiveSession(race["path"]).messages(STRATEGY_TOPICS))
        circuit = _circuit_for(messages)
        loss = (
            (circuit.pit_loss.green, circuit.pit_loss.safety_car)
            if circuit and circuit.pit_loss
            else (22.0, 13.5)
        )
        curves = fit_curves(con, year, exclude=(keys.get(race["date"]),)) or None
        monitor = RaceMonitor(pit_loss=circuit.pit_loss if circuit else None)
        engine = AlertEngine(curves=curves or {})
        candidates: dict[str, tuple] = {}  # chaser number -> (state, a, b) at the lap before
        last_lap = None
        pending = []  # (a, b, lap, p_now, p_base)
        order_at: dict[int, dict[str, int]] = {}  # leader's lap -> running order as it started
        for m in messages:
            monitor.feed(m)
            if m.topic != "LapCount":
                continue
            snap = monitor.snapshot()
            lap = snap.current_lap
            if not lap or lap == last_lap:
                continue
            last_lap = lap
            order_at[lap] = {d.number: d.position for d in snap.drivers if d.position}
            # Did a car from last lap's close pairs pit (in-lap = the lap just completed)?
            for a_no, (p_now, p_base, b_no, a_tla, b_tla) in list(candidates.items()):
                a_stop = _stop_lap_after(monitor, a_no, lap - 2)
                b_stop = _stop_lap_after(monitor, b_no, lap - 2)
                if a_stop is not None and b_stop is None:
                    pending.append((a_no, b_no, a_tla, b_tla, a_stop, p_now, p_base))
            candidates = {}
            if snap.track_status != "GREEN":
                continue
            try:
                state = race_state(
                    monitor, snap, pit_loss=loss, curves=curves, age_curves=bool(curves)
                )
            except NotEnoughData:
                continue
            for a, b in engine._pairs(state):
                p_now, p_base = engine._undercut(state, a.number, b.number)
                candidates[a.number] = (p_now, p_base, b.number, a.tla, b.tla)
        # Outcomes, from the finished race.
        final = {}
        for a_no, b_no, a_tla, b_tla, a_stop, p_now, p_base in pending:
            b_stop = _stop_lap_after(monitor, b_no, a_stop)
            if b_stop is None:
                continue
            judge = b_stop + RESOLVE_LAPS_AFTER_STOP
            pos = order_at.get(judge) or order_at.get(max(order_at)) or {}
            if a_no in pos and b_no in pos:
                final[(a_no, b_no)] = (a_tla, b_tla, a_stop, p_now, p_base, pos[a_no] < pos[b_no])
        for a_tla, b_tla, a_stop, p_now, p_base, worked in final.values():
            rows.append((race["meeting"], a_tla, b_tla, a_stop, p_now, p_base, worked))
            if not quiet:
                print(
                    f"  {race['meeting']:26} L{a_stop} {a_tla} on {b_tla}: p {p_now:.0%} -> {'worked' if worked else 'failed'}"
                )
    p = np.array([r[4] for r in rows])
    y = np.array([r[6] for r in rows], dtype=float)
    out = {"attempts": len(rows), "worked": int(y.sum()) if len(rows) else 0}
    if len(rows):
        out["brier"] = float(np.mean((p - y) ** 2))
        out["brier_base_rate"] = float(np.mean((y.mean() - y) ** 2))
        out["reliability"] = [
            (lo, hi, int(m.sum()), float(p[m].mean()), float(y[m].mean()))
            for lo, hi in ((0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01))
            if (m := (p >= lo) & (p < hi)).any()
        ]
    if not quiet:
        print(out)
    return out


if __name__ == "__main__":
    import sys

    undercut_attempts() if "--attempts" in sys.argv else backtest()
