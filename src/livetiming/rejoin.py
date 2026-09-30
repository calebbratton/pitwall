"""Where would a car rejoin if it pitted now? (the pit wall's "15s / 30s" markers)

A stop costs the circuit's measured pit loss: green, under a safety car (the field is slow, so
it costs less), or under a VSC. The car's time behind the leader grows by that loss; its
rejoin position is where that time falls in the current running order, assuming nobody else
pits on the same lap. Cars a lap or more down (no gap to the leader in the feed) are left out.
"""

from dataclasses import dataclass

from src.livetiming.circuits import PitLoss
from src.livetiming.snapshot import RaceSnapshot

DEFAULT_LOSS = {"green": 22.0, "sc": 13.5, "vsc": 15.5}  # when the circuit has no measurement
TRAFFIC_S = 1.5  # rejoining this close behind a car means running in its dirty air


@dataclass(frozen=True)
class Rejoin:
    kind: str  # "green" | "sc" | "vsc"
    loss_s: float
    position: int  # running position after the stop
    ahead: str | None  # car it would come out behind
    gap_ahead_s: float | None
    behind: str | None  # car it would come out ahead of
    gap_behind_s: float | None

    @property
    def traffic(self) -> bool:
        return self.gap_ahead_s is not None and self.gap_ahead_s < TRAFFIC_S


def losses(pit_loss: PitLoss | None) -> dict[str, float]:
    if pit_loss is None:
        return dict(DEFAULT_LOSS)
    return {"green": pit_loss.green, "sc": pit_loss.safety_car, "vsc": pit_loss.vsc}


def rejoin(snap: RaceSnapshot, tla: str, pit_loss: PitLoss | None = None) -> list[Rejoin]:
    """Rejoin projections for one car (empty if it isn't running on the lead lap)."""
    me = next((d for d in snap.drivers if d.tla == tla), None)
    if me is None or me.retired or me.in_pit or me.laps_down:
        return []
    my_gap = me.gap_to_leader_s if me.gap_to_leader_s is not None else 0.0
    if me.position != 1 and me.gap_to_leader_s is None:
        return []
    others = sorted(
        (
            (d.gap_to_leader_s if d.position != 1 else 0.0, d.tla)
            for d in snap.drivers
            if d.tla != tla
            and not d.retired
            and not d.laps_down
            and (d.gap_to_leader_s is not None or d.position == 1)
        ),
    )
    out = []
    for kind, loss in losses(pit_loss).items():
        t = my_gap + loss
        ahead = [(g, name) for g, name in others if g <= t]
        behind = [(g, name) for g, name in others if g > t]
        out.append(
            Rejoin(
                kind=kind,
                loss_s=round(loss, 1),
                position=len(ahead) + 1,
                ahead=ahead[-1][1] if ahead else None,
                gap_ahead_s=round(t - ahead[-1][0], 1) if ahead else None,
                behind=behind[0][1] if behind else None,
                gap_behind_s=round(behind[0][0] - t, 1) if behind else None,
            )
        )
    return out


def rejoin_table(snap: RaceSnapshot, pit_loss: PitLoss | None = None) -> dict[str, list[dict]]:
    """Every running car's projections, for the snapshot event."""
    table = {}
    for d in snap.drivers:
        rows = rejoin(snap, d.tla, pit_loss)
        if rows:
            table[d.tla] = [
                {
                    "kind": r.kind,
                    "loss": r.loss_s,
                    "position": r.position,
                    "ahead": r.ahead,
                    "gap_ahead": r.gap_ahead_s,
                    "behind": r.behind,
                    "gap_behind": r.gap_behind_s,
                    "traffic": r.traffic,
                }
                for r in rows
            ]
    return table
