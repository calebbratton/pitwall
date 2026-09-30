"""Monte Carlo race simulator (v1): predicts the finishing-order distribution.

All simulations run at once as numpy arrays shaped [sims, cars]. Only *relative* pace matters for
the order, so fuel burn and track evolution (shared by every car) are left out.

Per simulated race:
  start      grid spacing + a lap-1 shuffle
  each lap   lap time = pace + tyre wear × tyre age + noise; one pit stop in a window
  traffic    a car catching the one ahead passes with P(pass | pace advantage) — otherwise it's
             held behind (min gap + dirty-air loss)
  SC         random per-lap hazard, ~8 laps: the queue is compressed and every car that hasn't
             stopped (and is in its window) can pit cheaply
  VSC        random per-lap hazard, ~1-2 laps, gaps preserved: a car can only use it if it reaches
             the pit entry before it ends — cars that just passed the pit entry miss out
             (Norris, Madrid 2026)
  DNF        off by default: retirements are unpredictable noise for a pace model
"""

from dataclasses import dataclass

import numpy as np

from src.sim.inputs import WeekendInputs


@dataclass(frozen=True)
class SimParams:
    # Tuned (leave-one-race-out, 2026): long runs as extracted add noise, so quali dominates.
    quali_weight: float = 1.0  # blend of one-lap pace and long-run pace into race pace
    race_pace_scale: float = 0.9  # race-pace gaps ≈ this × qualifying gaps
    # Team race-vs-quali bias: tuned to 0 against luck-adjusted results (it was partly SC luck).
    form_weight: float = 0.0
    form_full_after: int = 5  # races of history for full weight (shrinks toward 0 before)
    missing_pace_per_grid_slot: float = 0.1  # s/lap per grid slot when a car has no pace data
    lap_noise: float = 0.35  # s, lap-to-lap variation
    # s, lap-1 shuffle. Tuned to 2.0 (grid edge): it also absorbs race randomness the model
    # doesn't represent, which keeps win probabilities calibrated.
    start_noise: float = 2.0
    grid_spacing: float = 0.3  # s per grid slot at the end of lap 1 (before the shuffle)
    pass_threshold: float = 1.2  # s/lap advantage for a 50% pass chance (tuned; grid edge)
    pass_scale: float = 0.2  # logistic width of the pass curve
    min_gap: float = 0.4  # s, closest a held car can follow
    dirty_air: float = 0.2  # s lost per lap stuck behind a car
    stop_window: tuple[float, float] = (0.3, 0.65)  # fraction of race distance
    vsc_pace_factor: float = 1.35  # VSC laps take ~35% longer (delta time to hold)
    include_dnfs: bool = False
    # SC/VSC timing is unpredictable luck: race predictions leave it out (pure pace + strategy).
    # The live pit-wall tools switch it on.
    include_neutralisations: bool = False


@dataclass(frozen=True)
class Prediction:
    tla: list[str]
    positions: np.ndarray  # [sims, cars] finishing position (1-based)

    def expected_position(self) -> np.ndarray:
        return self.positions.mean(axis=0)

    def probability(self, best: int) -> np.ndarray:
        """P(finishing in position `best` or better), per car."""
        return (self.positions <= best).mean(axis=0)

    def table(self) -> list[dict]:
        exp = self.expected_position()
        order = np.argsort(exp)
        return [
            {
                "tla": self.tla[i],
                "expected": round(float(exp[i]), 2),
                "p_win": round(float(self.probability(1)[i]), 3),
                "p_podium": round(float(self.probability(3)[i]), 3),
                "p_points": round(float(self.probability(10)[i]), 3),
            }
            for i in order
        ]


def traffic_step(
    new_T: np.ndarray,
    lap_time: np.ndarray,
    order: np.ndarray,
    running: np.ndarray,
    racing: np.ndarray,
    p: SimParams,
    rng: np.random.Generator,
) -> None:
    """Front to back (in `order`, the running order before the lap): a car that would close
    within min_gap of the car ahead passes with P(pace advantage), otherwise it's held behind
    and pays dirty air. Updates `new_T` in place; `racing` masks simulations under SC/VSC."""
    sims, n = new_T.shape
    flat = np.arange(sims)
    for k in range(1, n):
        car = order[:, k]
        ahead = order[:, k - 1]
        t_car = new_T[flat, car]
        t_ahead = new_T[flat, ahead]
        ok = running[flat, car] & running[flat, ahead] & racing
        close = ok & (t_car < t_ahead + p.min_gap)
        advantage = lap_time[flat, ahead] - lap_time[flat, car]
        p_pass = 1 / (1 + np.exp(-(advantage - p.pass_threshold) / p.pass_scale))
        held = close & (rng.random(sims) >= p_pass)
        new_T[flat, car] = np.where(held, t_ahead + p.min_gap + p.dirty_air, t_car)


def race_pace(inputs: WeekendInputs, p: SimParams) -> np.ndarray:
    """Per-car race pace offset (s/lap, lower = faster), from qualifying and long runs."""
    paces = []
    for d in inputs.drivers:
        parts = []
        if d.quali_delta_s is not None and p.quali_weight > 0:
            parts.append((d.quali_delta_s * p.race_pace_scale, p.quali_weight))
        if d.long_run_delta_s is not None and p.quali_weight < 1:
            parts.append((d.long_run_delta_s, 1 - p.quali_weight))
        if parts:
            pace = sum(v * w for v, w in parts) / sum(w for _, w in parts)
        else:
            pace = p.missing_pace_per_grid_slot * (d.grid - 1)
        shrink = min(d.form_races / p.form_full_after, 1.0)
        paces.append(pace + p.form_weight * shrink * d.form_s)
    pace = np.array(paces)
    # Centre on the median so long-run (median-relative) and quali (pole-relative) parts mix.
    return pace - np.median(pace)


def simulate(
    inputs: WeekendInputs, params: SimParams | None = None, sims: int = 2000, seed: int = 0
) -> Prediction:
    rng = np.random.default_rng(seed)
    p = params or SimParams()
    n, laps = len(inputs.drivers), inputs.laps
    pace = race_pace(inputs, p)[None, :]  # [1, n]
    grid = np.array([d.grid for d in inputs.drivers], dtype=float)
    deg = float(np.mean(list(inputs.deg.values())))
    rows = np.arange(sims)[:, None]

    # Start: grid spacing plus a lap-1 shuffle.
    T = (grid[None, :] - 1) * p.grid_spacing + rng.normal(0, p.start_noise, (sims, n))
    age = np.zeros((sims, n))
    first_window = int(laps * p.stop_window[0])
    stop_lap = rng.integers(first_window, int(laps * p.stop_window[1]) + 1, (sims, n))
    stopped = np.zeros((sims, n), dtype=bool)
    dnf_prob = inputs.dnf_prob if p.include_dnfs else 0.0
    dnf_lap = np.where(
        rng.random((sims, n)) < dnf_prob, rng.integers(1, laps + 1, (sims, n)), laps + 1
    )
    sc_hazard = inputs.sc_per_race / laps if p.include_neutralisations else 0.0
    vsc_hazard = inputs.vsc_per_race / laps if p.include_neutralisations else 0.0
    sc_left = np.zeros(sims)  # laps of SC remaining
    vsc_left = np.zeros(sims)  # laps of VSC remaining (fractional)

    for lap in range(2, laps + 1):
        running = dnf_lap > lap
        quiet = (sc_left <= 0) & (vsc_left <= 0)
        draw = rng.random(sims)
        new_sc = quiet & (draw < sc_hazard)
        new_vsc = quiet & ~new_sc & (draw < sc_hazard + vsc_hazard)
        sc_left = np.where(new_sc, np.maximum(1, rng.normal(inputs.sc_laps, 1.5, sims)), sc_left)
        vsc_left = np.where(
            new_vsc, np.clip(rng.exponential(inputs.vsc_laps, sims), 0.3, 4.0), vsc_left
        )
        under_sc = sc_left > 0
        under_vsc = vsc_left > 0

        order = np.argsort(np.where(running, T, np.inf), axis=1)  # running order before the lap
        lap_time = pace + deg * age + rng.normal(0, p.lap_noise, (sims, n))
        lap_time = np.where(under_sc[:, None], 0.0, lap_time)  # SC: everyone at the same pace

        # Who can pit cheaply: under an SC anyone; under a new VSC only cars that reach the
        # pit entry (end of the lap) before it ends. Their lap position when it's called is
        # uniform; reaching the entry takes (1 - position) laps at VSC pace.
        lap_position = rng.random((sims, n))
        reaches_pit = (1 - lap_position) * p.vsc_pace_factor <= vsc_left[:, None]
        in_window = lap >= first_window * 0.8
        cheap = in_window & (new_sc[:, None] | (new_vsc[:, None] & reaches_pit))
        pit_now = ~stopped & running & ((stop_lap == lap) | cheap)
        loss = np.where(
            under_sc[:, None],
            inputs.pit_loss_sc,
            np.where(
                under_vsc[:, None] & (new_vsc[:, None] & reaches_pit),
                inputs.pit_loss_vsc,
                inputs.pit_loss_green,
            ),
        )
        lap_time = lap_time + np.where(pit_now, loss, 0.0)
        age = np.where(pit_now, 0, age + 1)
        stopped |= pit_now

        new_T = T + lap_time
        if new_sc.any():
            # Queue behind the SC: gaps shrink to ~0.8 s in running order (order is kept).
            rank = np.empty_like(order)
            rank[rows, order] = np.arange(n)[None, :]
            leader = np.take_along_axis(new_T, order[:, :1], axis=1)
            queued = leader + rank * 0.8
            new_T = np.where(new_sc[:, None] & running, queued, new_T)
        racing = ~under_sc & ~under_vsc
        if racing.any():
            traffic_step(new_T, lap_time, order, running, racing, p, rng)
        T = np.where(running, new_T, T)
        sc_left = np.maximum(sc_left - 1, 0)
        vsc_left = np.maximum(vsc_left - 1, 0)

    # Finishing order: classified cars by time; DNFs behind, those who lasted longer ahead.
    key = np.where(dnf_lap > laps, T, 1e9 - dnf_lap)
    positions = np.empty((sims, n), dtype=int)
    positions[rows, np.argsort(key, axis=1)] = np.arange(1, n + 1)[None, :]
    return Prediction([d.tla for d in inputs.drivers], positions)
