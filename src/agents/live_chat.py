"""Race-engineer chat about the race as it stands right now.

Deterministic first: the current snapshot, the who-wins-from-here simulation, tyre outlooks and
(under SC/VSC) the pit calls are computed in code; one LLM call turns them into an answer. The
model is told to use only those numbers.
"""

import json
import logging
import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.livetiming.monitor import RaceMonitor
from src.livetiming.strategy import pit_calls
from src.llm.factory import get_chat_model
from src.models.tyre_curves import season_curves
from src.sim.inrace import race_state, simulate_from

log = logging.getLogger(__name__)

PROMPT = """\
You are a Formula 1 race strategy engineer on the pit wall, answering questions about the race
AS IT STANDS RIGHT NOW (lap {lap}/{total}, track status {status}).

Rules:
- Use only the numbers in RACE STATE below. Don't invent lap times, gaps or probabilities.
- "p_win"/"p_podium" come from simulating the remaining laps {sims} times from this state
  (pace from recent clean laps, tyre wear measured in this race, tyre life from 2026 stints,
  owed compound stops). They exclude future safety cars and retirements — say so if relevant.
- For "can X make it to the end?" use tyre_age, laps_remaining and tyre_note. Cars with
  stopped=false and old tyres are the ones that stayed out; if every car has fresh tyres, say
  that nobody stayed out rather than describing stay-outs.
- Tyres: HARD/MEDIUM/SOFT are this weekend's labels; compound_c is Pirelli's actual compound
  (C1 hardest .. C5 softest). Mention it when comparing tyres.
- "pit_within_3_laps": each car's chance of pitting in the next 3 laps from a model learned on
  2023-26 races (tyre age, stint, rivals' and teammate's stops, SC). Reliable up to ~35%; above
  that it overstates - say "likely soon", not "will pit".
- "undercuts": for close pairs before their first stop, the chance the chaser comes out ahead if
  it pits now (learned from 570 real undercuts; 2026 undercuts work only ~1/3 of the time).
- "battles": close pairs' chance of a pass on track within 5 laps (p_pass) and of being within
  1 s within 3 laps (p_close), learned from 2023-26 races.
- Refer to drivers by the three-letter codes exactly as given (VER, PIA, ...). Never expand a
  code into a name or guess who it is.
- Be direct and brief, like radio to the pit wall: verdict first, then 2-4 supporting points.

RACE STATE (JSON):
{state}
"""


def _strategy_models(monitor: RaceMonitor, snapshot, loss: tuple[float, float]) -> dict:
    """What the learned models say right now: who's likely to pit, undercuts, battles. Each part
    is skipped (not guessed) if its model or data isn't available."""
    out: dict[str, Any] = {}
    if os.getenv("PITWALL_LIVE_MODELS", "1") == "0":  # tests: no warehouse-trained models
        return out
    try:
        from src.sim.pit_hazard import pit_within

        likely = sorted(pit_within(snapshot, monitor).items(), key=lambda kv: -kv[1])
        tyres = {d.tla: (d.compound, d.tyre_age_laps) for d in snapshot.drivers}
        out["pit_within_3_laps"] = [
            {
                "tla": t,
                "p": p,
                "compound": tyres.get(t, (None, None))[0],
                "tyre_age": tyres.get(t, (None, None))[1],
            }
            for t, p in likely[:8]
            if p >= 0.05
        ]
    except Exception:  # noqa: BLE001
        log.info("pit-timing model unavailable for the live chat")
    try:
        from src.livetiming.alerts import undercut_chances

        out["undercuts"] = [
            {
                "chaser": pair["chaser"].tla,
                "ahead": pair["ahead"].tla,
                "gap_s": round(pair["gap"], 2),
                "p_ahead_if_pits_now": round(p, 2),
            }
            for pair, p in undercut_chances(snapshot, monitor, loss)
        ]
    except Exception:  # noqa: BLE001
        log.info("undercut model unavailable for the live chat")
    try:
        from src.livetiming.director import _circuit_rel, close_pairs, score_pairs

        out["battles"] = [
            {k: v for k, v in b.items() if not k.startswith("_") and k != "closing"}
            for b in score_pairs(
                snapshot, monitor, close_pairs(snapshot), {}, _circuit_rel(snapshot)
            )[:5]
        ]
    except Exception:  # noqa: BLE001
        log.info("battle models unavailable for the live chat")
    return out


def race_context(monitor: RaceMonitor, sims: int = 2000) -> dict[str, Any]:
    snapshot = monitor.snapshot()
    pit_loss = monitor.pit_loss
    loss = (pit_loss.green, pit_loss.safety_car) if pit_loss else (22.0, 13.5)
    curves = season_curves(snapshot.year) if snapshot.year else {}
    state = race_state(monitor, snapshot, pit_loss=loss, curves=curves, age_curves=bool(curves))
    prediction = simulate_from(state, sims=sims)
    gaps = {d.tla: d.gap_to_leader_s for d in snapshot.drivers}
    context: dict[str, Any] = {
        "lap": snapshot.current_lap,
        "total_laps": snapshot.total_laps,
        "laps_remaining": state.laps_remaining,
        "status": snapshot.track_status,
        "cars": [
            {
                k: row[k]
                for k in (
                    "position",
                    "tla",
                    "compound",
                    "compound_c",
                    "tyre_age",
                    "stopped",
                    "p_win",
                    "p_podium",
                    "tyre_risk",
                    "tyre_note",
                )
            }
            | {"gap_to_leader_s": gaps.get(row["tla"])}
            for row in prediction["table"]
        ],
        "notes": prediction["notes"],
    }
    context |= _strategy_models(monitor, snapshot, loss)
    if snapshot.track_status in ("SAFETY_CAR", "VSC", "VSC_ENDING"):
        report = pit_calls(snapshot, pit_loss)
        context["pit_calls"] = [
            {"tla": c.tla, "call": c.call, "reason": c.reason} for c in report.calls[:12]
        ]
    return {"context": context, "prediction": prediction}


def answer(
    monitor: RaceMonitor, message: str, model: BaseChatModel | None = None, sims: int = 2000
) -> dict[str, Any]:
    computed = race_context(monitor, sims=sims)
    ctx = computed["context"]
    prompt = PROMPT.format(
        lap=ctx["lap"],
        total=ctx["total_laps"],
        status=ctx["status"],
        sims=sims,
        state=json.dumps(ctx, separators=(",", ":")),
    )
    reply = (model or get_chat_model("analyst")).invoke(
        [SystemMessage(prompt), HumanMessage(message)]
    )
    return {
        "answer": reply.content if isinstance(reply.content, str) else str(reply.content),
        "lap": ctx["lap"],
        "status": ctx["status"],
        "prediction": computed["prediction"],
        "steps": [
            f"state: lap {ctx['lap']}/{ctx['total_laps']}, {ctx['status']}",
            f"simulated {sims} finishes from this state",
            *(["pit calls computed (neutralisation)"] if "pit_calls" in ctx else []),
        ],
    }


def prediction_event(monitor: RaceMonitor, sims: int = 2000) -> dict[str, Any] | None:
    """`prediction` SSE event (who wins from here), or None if there isn't enough data yet."""
    try:
        prediction = race_context(monitor, sims=sims)["prediction"]
    except ValueError:
        return None
    return {"type": "prediction", **prediction}
