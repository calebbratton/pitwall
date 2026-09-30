"""Safety car / VSC / red flag luck, measured from timing data — and results with it removed.

Neutralisation timing isn't predictable, so predictions ignore it and are scored against the
race with its effects taken out ("Norris would have won Madrid 2026").

For each neutralisation window and each driver, from the lap before it starts to the lap after
the restart:
    expected elapsed = laps × their clean green-flag pace (+ green pit loss if they pitted)
    luck             = expected − actual, relative to the field median (positive = gained)
Pitting under a VSC/SC shows up as losing less than a green stop; bunching up behind an SC shows
up as the leader losing time relative to the cars behind. Adjusted finish = actual + luck.
"""

import statistics
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class LuckReport:
    luck_s: dict[int, float]  # driver -> seconds gained from neutralisations (+ = helped)
    adjusted_order: list[int]  # classified drivers in luck-adjusted finishing order
    windows: int


def neutralisation_luck(con, race_sk: int, pit_loss_green: float) -> LuckReport:
    laps = con.execute(
        """SELECT driver_number, lap_number, lap_end, lap_time FROM laps
           WHERE session_key = ? AND lap_end IS NOT NULL ORDER BY 1, 2""",
        [race_sk],
    ).fetchall()
    crossings: dict[int, list[tuple[int, object]]] = defaultdict(list)
    for driver, lap, end, _ in laps:
        crossings[driver].append((lap, end))
    pace = dict(
        con.execute(
            "SELECT driver_number, median(lap_time) FROM clean_laps WHERE session_key = ? GROUP BY 1",
            [race_sk],
        ).fetchall()
    )
    field_pace = statistics.median(pace.values()) if pace else 90.0
    pits: dict[int, set[int]] = defaultdict(set)
    for driver, lap in con.execute(
        "SELECT driver_number, lap_number FROM raw_pits WHERE session_key = ?", [race_sk]
    ).fetchall():
        pits[driver].add(lap)
    windows = con.execute(
        """SELECT start::TIMESTAMPTZ, least("end", '2100-01-01')::TIMESTAMPTZ FROM raw_neutralised
           WHERE session_key = ? ORDER BY 1""",
        [race_sk],
    ).fetchall()

    luck: dict[int, float] = defaultdict(float)
    for start, end in windows:
        per_driver: dict[int, float] = {}
        for driver, laps_ in crossings.items():
            before = [(lap, t) for lap, t in laps_ if t <= start]
            after = [(lap, t) for lap, t in laps_ if t > end]
            if not before or len(after) < 2:
                continue
            (lap_b, t_b), (lap_a, t_a) = before[-1], after[1]  # +1 lap: the restart lap
            n_laps = lap_a - lap_b
            pitted = any(lap_b < lap <= lap_a for lap in pits.get(driver, ()))
            expected = n_laps * pace.get(driver, field_pace) + (pit_loss_green if pitted else 0)
            per_driver[driver] = expected - (t_a - t_b).total_seconds()
        if len(per_driver) < 5:
            continue
        reference = statistics.median(per_driver.values())
        for driver, gain in per_driver.items():
            luck[driver] += gain - reference

    results = con.execute(
        """SELECT r.driver_number, r.number_of_laps, max(l.lap_end)
           FROM raw_results r JOIN laps l
             ON l.session_key = r.session_key AND l.driver_number = r.driver_number
           WHERE r.session_key = ? AND r.position IS NOT NULL
             AND NOT coalesce(r.dnf, FALSE) AND NOT coalesce(r.dns, FALSE)
             AND NOT coalesce(r.dsq, FALSE)
           GROUP BY 1, 2""",
        [race_sk],
    ).fetchall()
    finish = {
        d: (laps_done or 0, end.timestamp() + luck.get(d, 0.0)) for d, laps_done, end in results
    }
    adjusted = sorted(finish, key=lambda d: (-finish[d][0], finish[d][1]))
    return LuckReport(dict(luck), adjusted, len(windows))
