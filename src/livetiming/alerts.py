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

import functools
import itertools
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.livetiming.monitor import RaceMonitor
from src.livetiming.snapshot import RaceSnapshot
from src.livetiming.strategy import PitCallReport
from src.sim.inrace import NotEnoughData, RaceState, StopPlan, race_state, simulate_from

log = logging.getLogger(__name__)

# On with the learned model (src/sim/undercut_model.py), which beats the same-season base rate
# leave-one-race-out in every season 2023-26. (The simulator version did not: 2026 Brier 0.234
# vs 0.217 for the base rate.)
UNDERCUT_ALERTS = True
UNDERCUT_MAX_GAP_S = 4.0  # beyond this a fresh-tyre out-lap can't close it
UNDERCUT_MIN_P = 0.6  # the learned model's 0.6-0.75 band worked 57-66% of the time
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
        self._stops_seen: set[tuple[str, int]] = set()  # (car, out-lap) already announced
        self._fastest: float | None = None
        self._raining: bool | None = None

    # --- safety car ----------------------------------------------------------------------------

    def on_pit_calls(self, report: PitCallReport) -> list[dict[str, Any]]:
        alert = sc_call_alert(report)
        if alert is None or alert.id in self.alerts:
            return []
        self.alerts[alert.id] = alert
        return [alert.event()]

    # --- every lap -----------------------------------------------------------------------------

    # --- facts (no prediction: straight from timing) --------------------------------------------

    def _facts(self, snap: RaceSnapshot, monitor: RaceMonitor) -> list[dict[str, Any]]:
        events = []
        lap = snap.current_lap or 0
        by_number = {d.number: d for d in snap.drivers}
        ordered = [d for d in snap.drivers if d.position and not d.retired]
        ordered.sort(key=lambda d: d.position)
        # Completed stops (on the out-lap, with where the car rejoined). Individual alerts only for
        # green-flag stops into the top 10; busy laps (SC/VSC, 4+ stops) and the rest of the field
        # are one summary line each, so a safety car doesn't produce a dozen alerts.
        stops = []
        for car, recs in monitor.laps.items():
            if not recs or not recs[-1].pit_out or (car, recs[-1].lap) in self._stops_seen:
                continue
            self._stops_seen.add((car, recs[-1].lap))
            d = by_number.get(car)
            if d is not None and d.position is not None and not d.retired:
                stops.append(d)
        stops.sort(key=lambda d: d.position)
        neutralised = snap.track_status in ("SAFETY_CAR", "VSC", "VSC_ENDING", "RED_FLAG")
        busy = neutralised or len(stops) > 3
        singles = [] if busy else [d for d in stops if d.position <= 10]
        grouped = [d for d in stops if d not in singles]
        for d in singles:
            ahead = next((o for o in ordered if o.position == d.position - 1), None)
            behind_gap = (
                f", {d.interval_s:.1f} s behind {ahead.tla}" if ahead and d.interval_s else ""
            )
            events.append(
                Alert(
                    id=f"stop-{d.tla}-{lap}",
                    kind="pit_stop",
                    lap=lap,
                    headline=f"{d.tla} pitted: rejoined P{d.position}{behind_gap}",
                    detail=f"On {(d.compound or '?').lower()}s (stop {d.pit_stops}).",
                    drivers=[d.tla] + ([ahead.tla] if ahead else []),
                    status="resolved",
                ).event()
            )
        if grouped:
            why = {"SAFETY_CAR": " (SC)", "VSC": " (VSC)", "VSC_ENDING": " (VSC)"}.get(
                snap.track_status, ""
            )
            listed = ", ".join(f"{d.tla} P{d.position}" for d in grouped[:10])
            more = f" +{len(grouped) - 10} more" if len(grouped) > 10 else ""
            events.append(
                Alert(
                    id=f"stops-{lap}",
                    kind="pit_stop",
                    lap=lap,
                    headline=f"Lap {lap} stops{why}: {listed}{more}",
                    detail="Positions after rejoining.",
                    drivers=[d.tla for d in grouped],
                    status="resolved",
                ).event()
            )
        # Fastest lap: second half of the race only (early ones change every few laps).
        if snap.total_laps and lap > snap.total_laps / 2:
            for car, recs in monitor.laps.items():
                r = recs[-1] if recs else None
                clean = (
                    r is not None
                    and r.time_s
                    and r.lap == lap - 1
                    and not (r.pit_in or r.pit_out or r.neutralised)
                )
                if not clean or (self._fastest is not None and r.time_s >= self._fastest - 1e-6):
                    continue
                d = by_number.get(car)
                if self._fastest is not None and d is not None:
                    m, sec = divmod(r.time_s, 60)
                    events.append(
                        Alert(
                            id=f"fastest-{d.tla}-{r.lap}",
                            kind="fastest_lap",
                            lap=lap,
                            headline=f"Fastest lap: {d.tla} {int(m)}:{sec:06.3f}",
                            detail=f"Lap {r.lap}, on {(r.compound or '?').lower()}s "
                            f"{r.tyre_age} laps old.",
                            drivers=[d.tla],
                            status="resolved",
                        ).event()
                    )
                self._fastest = r.time_s
        elif snap.total_laps:
            for recs in monitor.laps.values():
                r = recs[-1] if recs else None
                if r and r.time_s and not (r.pit_in or r.pit_out or r.neutralised):
                    self._fastest = min(self._fastest or r.time_s, r.time_s)
        # Rain starting or stopping at the circuit.
        weather = monitor.state.topics.get("WeatherData", {})
        if "Rainfall" in weather:
            raining = str(weather.get("Rainfall")) not in ("0", "")
            if self._raining is not None and raining != self._raining:
                events.append(
                    Alert(
                        id=f"rain-{lap}",
                        kind="weather",
                        lap=lap,
                        headline="Rain at the circuit" if raining else "Rain has stopped",
                        detail="From the circuit's weather station (WeatherData).",
                        drivers=[],
                        status="resolved",
                    ).event()
                )
            self._raining = raining
        return events

    def on_lap(self, monitor: RaceMonitor, pit_loss: tuple[float, float]) -> list[dict[str, Any]]:
        snap = monitor.snapshot()
        events = self._resolve(snap, monitor)
        if snap.is_race:
            events += self._facts(snap, monitor)
        if snap.track_status != "GREEN" or not snap.is_race:
            self._streak.clear()
            return events
        if not UNDERCUT_ALERTS:
            return events
        model = undercut_model()
        lap = snap.current_lap or 0
        if model is None or not snap.total_laps or snap.total_laps - lap < UNDERCUT_MIN_LAPS_LEFT:
            return events
        seen = set()
        for pair, p_now in undercut_chances(snap, monitor, pit_loss, model)[:MAX_PAIRS]:
            a, b = pair["chaser"], pair["ahead"]
            cid = f"undercut-{a.tla}-{b.tla}"
            seen.add(cid)
            self._streak[cid] = self._streak.get(cid, 0) + 1 if p_now >= UNDERCUT_MIN_P else 0
            live = self.alerts.get(cid)
            if live and live.status == "live":
                if abs((live.p or 0) - p_now) >= 0.15:  # update in place on a big move
                    live.p = round(p_now, 2)
                    live.detail = _undercut_detail(a.tla, b.tla, p_now, pair["gap"], snap, pit_loss)
                    events.append(live.event())
                continue
            if self._streak[cid] >= PERSIST_LAPS:
                alert = Alert(
                    id=f"{cid}-{lap}",
                    kind="undercut",
                    lap=lap,
                    headline=f"{a.tla}'s undercut on {b.tla} is on",
                    detail=_undercut_detail(a.tla, b.tla, p_now, pair["gap"], snap, pit_loss),
                    drivers=[a.tla, b.tla],
                    p=round(p_now, 2),
                    numbers=[a.number, b.number],
                    stops_at_alert={a.number: a.pit_stops, b.number: b.pit_stops},
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


def _undercut_detail(
    a: str, b: str, p: float, gap: float, snap: RaceSnapshot, pit_loss: tuple[float, float]
) -> str:
    left = (snap.total_laps or 0) - (snap.current_lap or 0)
    return (
        f"{a} is {gap:.1f} s behind {b}. Pitting now, {a} comes out ahead after both stop about "
        f"{p:.0%} of the time (learned from 570 undercuts, 2023-26; similar calls were right "
        f"about that often). Pit loss {pit_loss[0]:.0f} s, {left} laps to go."
    )


def undercut_chances(snap, monitor, pit_loss: tuple[float, float], model=None) -> list[tuple]:
    """(pair, P(chaser ends up ahead if it pits now)) for every close pair before its first stop,
    from the learned undercut model (same features as its training data)."""
    from src.sim.undercut_data import Attempt, candidate_pairs
    from src.sim.undercut_model import features

    model = model or undercut_model()
    if model is None or not snap.total_laps:
        return []
    out = []
    for pair in candidate_pairs(snap, monitor):
        a, b = pair["chaser"], pair["ahead"]
        attempt = Attempt(
            year=snap.year or 0,
            race=snap.meeting,
            lap=snap.current_lap or 0,
            total_laps=snap.total_laps,
            chaser=a.tla,
            ahead=b.tla,
            gap_s=pair["gap"],
            chaser_age=a.tyre_age_laps or 0,
            ahead_age=b.tyre_age_laps or 0,
            chaser_compound=a.compound or "",
            ahead_compound=b.compound or "",
            new_compound="HARD" if a.compound != "HARD" else "MEDIUM",
            pace_delta_s=pair["pace_delta"],
            pit_loss_s=pit_loss[0],
            response_laps=None,
            worked=None,
        )
        out.append((pair, float(model.predict(np.array([features(attempt)], dtype=float))[0])))
    return out


@functools.cache
def undercut_model():
    """The learned undercut model fitted on every cached attempt (src/sim/undercut_data.py), or
    None if the dataset hasn't been built."""
    from src.sim.undercut_data import CACHE, Attempt
    from src.sim.undercut_model import Logistic, features

    if not CACHE.exists():
        log.warning("no undercut dataset (python -m src.sim.undercut_data): undercut alerts off")
        return None
    rows = [Attempt(**json.loads(line)) for line in CACHE.read_text().splitlines()]
    rows = [r for r in rows if r.worked is not None]
    X = np.array([features(r) for r in rows], dtype=float)
    y = np.array([r.worked for r in rows], dtype=float)
    return Logistic().fit(X, y)


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
