"""Safety car / VSC pit calls: who should pit and who should stay out.

Deterministic on purpose: under a safety car there's about a lap to decide, so the call is
computed from the snapshot in milliseconds and the LLM only explains it. Every rule of thumb is a
named constant and is reported with the calls, so they can be tuned against past races.
"""

from dataclasses import dataclass

from src.livetiming.snapshot import DriverState, RaceSnapshot

# Assumptions (rules of thumb, not measurements):
PIT_LANE_BYPASS_S = 5.0  # track time the pit lane replaces; loss = lane time - this
DEFAULT_PIT_LANE_S = 22.0  # used until the session's own pit-lane times are known
NEUTRALISED_LOSS_FACTOR = {"SAFETY_CAR": 0.5, "VSC": 0.65, "VSC_ENDING": 0.8}
FRESH_TYRE_LAPS = 5  # stopped this recently: no point stopping again
OLD_TYRE_LAPS = 15  # old enough that fresh tyres are worth a couple of places
MAX_PLACES_FOR_OLD_TYRES = 2
MIN_LAPS_TO_BENEFIT = 5  # fresh tyres can't pay back a stop in fewer laps than this


@dataclass(frozen=True)
class PitCall:
    tla: str
    position: int | None
    call: str  # "PIT" | "STAY OUT" | "IN PIT" | "RETIRED"
    reason: str
    places_at_risk: int | None
    compound: str | None
    tyre_age_laps: int | None


@dataclass(frozen=True)
class PitCallReport:
    track_status: str
    lap: int | None
    laps_remaining: int | None
    green_pit_loss_s: float
    pit_loss_now_s: float
    calls: tuple[PitCall, ...]
    assumptions: tuple[str, ...]


def _places_at_risk(driver: DriverState, snapshot: RaceSnapshot, loss_s: float) -> int | None:
    """Cars behind within `loss_s` that would pass if this driver pits and they don't."""
    if driver.gap_to_leader_s is None:
        return None
    return sum(
        1
        for other in snapshot.drivers
        if other is not driver
        and not other.retired
        and not other.in_pit
        and other.gap_to_leader_s is not None
        and (driver.position or 99) < (other.position or 99)
        and other.gap_to_leader_s - driver.gap_to_leader_s <= loss_s
    )


def _call(
    driver: DriverState, snapshot: RaceSnapshot, loss_s: float, green_loss_s: float
) -> PitCall:
    def make(call: str, reason: str, at_risk: int | None = None) -> PitCall:
        return PitCall(
            driver.tla,
            driver.position,
            call,
            reason,
            at_risk,
            driver.compound,
            driver.tyre_age_laps,
        )

    if driver.retired:
        return make("RETIRED", "Out of the race.")
    if driver.in_pit:
        return make("IN PIT", "Already in the pit lane.")

    at_risk = _places_at_risk(driver, snapshot, loss_s)
    remaining = snapshot.laps_remaining
    age = driver.tyre_age_laps
    risk_text = (
        "unknown gaps behind"
        if at_risk is None
        else f"{at_risk} car(s) within {loss_s:.1f}s behind"
    )

    if driver.needs_second_compound and (remaining is None or remaining > 0):
        return make(
            "PIT",
            f"Still needs a second dry compound. A stop now costs ~{loss_s:.1f}s instead of "
            f"~{green_loss_s:.1f}s under green ({risk_text}).",
            at_risk,
        )
    if remaining is not None and remaining < MIN_LAPS_TO_BENEFIT:
        return make(
            "STAY OUT", f"Only {remaining} laps left: fresh tyres can't pay back a stop.", at_risk
        )
    if age is not None and age <= FRESH_TYRE_LAPS:
        return make("STAY OUT", f"Tyres only {age} laps old.", at_risk)
    if at_risk == 0:
        return make("PIT", f"Free stop: no car within {loss_s:.1f}s behind.", at_risk)
    if (
        age is not None
        and age >= OLD_TYRE_LAPS
        and at_risk is not None
        and at_risk <= MAX_PLACES_FOR_OLD_TYRES
    ):
        return make(
            "PIT",
            f"{age}-lap-old tyres; fresh ones are worth the {at_risk} place(s) at risk.",
            at_risk,
        )
    return make("STAY OUT", f"Track position: pitting risks {risk_text}.", at_risk)


def pit_calls(snapshot: RaceSnapshot) -> PitCallReport:
    lane_s = snapshot.median_pit_lane_s or DEFAULT_PIT_LANE_S
    green_loss = lane_s - PIT_LANE_BYPASS_S
    factor = NEUTRALISED_LOSS_FACTOR.get(snapshot.track_status, 1.0)
    loss_now = green_loss * factor
    lane_source = (
        f"median of {len(snapshot.pit_lane_times_s)} pit-lane times this session"
        if snapshot.pit_lane_times_s
        else "default (no stops yet this session)"
    )
    return PitCallReport(
        track_status=snapshot.track_status,
        lap=snapshot.current_lap,
        laps_remaining=snapshot.laps_remaining,
        green_pit_loss_s=round(green_loss, 1),
        pit_loss_now_s=round(loss_now, 1),
        calls=tuple(_call(d, snapshot, loss_now, green_loss) for d in snapshot.drivers),
        assumptions=(
            (
                f"Pit lane time {lane_s:.1f}s ({lane_source}) minus {PIT_LANE_BYPASS_S}s bypass "
                f"= {green_loss:.1f}s loss under green."
            ),
            f"{snapshot.track_status} multiplies the loss by {factor} -> {loss_now:.1f}s.",
            "Places at risk assumes the cars behind stay out.",
            f"Fresh tyres: <= {FRESH_TYRE_LAPS} laps; old tyres: >= {OLD_TYRE_LAPS} laps.",
        ),
    )
