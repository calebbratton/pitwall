"""Monte Carlo race simulator (v1): predicts the finishing-order distribution.

All simulations run at once as numpy arrays shaped [sims, cars]. Only *relative* pace matters for
the order, so fuel burn and track evolution (shared by every car) are left out.

Per simulated race:
  start      grid spacing + a lap-1 shuffle
  each lap   lap time = pace + tyre wear × tyre age + noise; one pit stop in a window
  traffic    a car catching the one ahead passes with P(pass | pace advantage) — otherwise it's
             held behind (min gap + dirty-air loss)
  SC         random per-lap hazard; the queue is compressed and cars that haven't stopped (and are
             in their window) pit cheaply
  DNF        random per car
"""

from dataclasses import dataclass

import numpy as np

from src.sim.inputs import WeekendInputs


@dataclass(frozen=True)
class SimParams:
    # Tuned (leave-one-race-out, 2026): long runs as extracted add noise, so quali dominates.
    quali_weight: float = 1.0  # blend of one-lap pace and long-run pace into race pace
    race_pace_scale: float = 0.9  # race-pace gaps ≈ this × qualifying gaps
    form_weight: float = 0.5  # how much of a team's race-vs-quali bias to apply (tuned)
    form_full_after: int = 5  # races of history for full weight (shrinks toward 0 before)
    missing_pace_per_grid_slot: float = 0.1  # s/lap per grid slot when a car has no pace data
    lap_noise: float = 0.35  # s, lap-to-lap variation
    start_noise: float = 0.6  # s, lap-1 shuffle
    grid_spacing: float = 0.3  # s per grid slot at the end of lap 1 (before the shuffle)
    pass_threshold: float = 0.5  # s/lap advantage for a 50% pass chance
    pass_scale: float = 0.2  # logistic width of the pass curve
    min_gap: float = 0.4  # s, closest a held car can follow
    dirty_air: float = 0.2  # s lost per lap stuck behind a car
    stop_window: tuple[float, float] = (0.3, 0.65)  # fraction of race distance
    sc_laps: int = 4


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

    # Start: grid spacing plus a lap-1 shuffle.
    T = (grid[None, :] - 1) * p.grid_spacing + rng.normal(0, p.start_noise, (sims, n))
    age = np.zeros((sims, n))
    stop_lap = rng.integers(
        int(laps * p.stop_window[0]), int(laps * p.stop_window[1]) + 1, (sims, n)
    )
    stopped = np.zeros((sims, n), dtype=bool)
    dnf_lap = np.where(
        rng.random((sims, n)) < inputs.dnf_prob, rng.integers(1, laps + 1, (sims, n)), laps + 1
    )
    sc_hazard = inputs.sc_per_race / laps
    sc_left = np.zeros(sims, dtype=int)
    rows = np.arange(sims)[:, None]

    for lap in range(2, laps + 1):
        running = dnf_lap > lap
        # Safety car starts
        new_sc = (sc_left == 0) & (rng.random(sims) < sc_hazard)
        sc_left = np.where(new_sc, p.sc_laps, np.maximum(sc_left - 1, 0))
        under_sc = sc_left > 0

        order = np.argsort(np.where(running, T, np.inf), axis=1)  # running order before the lap
        lap_time = pace + deg * age + rng.normal(0, p.lap_noise, (sims, n))

        # Pit stops: planned stop lap, or opportunistic under a new SC inside the window.
        in_window = lap >= int(laps * p.stop_window[0]) * 0.8
        pit_now = ~stopped & running & ((stop_lap == lap) | (new_sc[:, None] & in_window))
        loss = np.where(under_sc[:, None], inputs.pit_loss_sc, inputs.pit_loss_green)
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
        if not under_sc.all():
            # Traffic, front to back: a car that would close within min_gap of the car ahead
            # passes with P(advantage), else it's held behind and pays dirty air.
            green = ~under_sc
            for k in range(1, n):
                car = order[:, k]
                ahead = order[:, k - 1]
                t_car = new_T[rows[:, 0], car]
                t_ahead = new_T[rows[:, 0], ahead]
                ok = running[rows[:, 0], car] & running[rows[:, 0], ahead] & green
                close = ok & (t_car < t_ahead + p.min_gap)
                advantage = lap_time[rows[:, 0], ahead] - lap_time[rows[:, 0], car]
                p_pass = 1 / (1 + np.exp(-(advantage - p.pass_threshold) / p.pass_scale))
                held = close & (rng.random(sims) >= p_pass)
                new_T[rows[:, 0], car] = np.where(held, t_ahead + p.min_gap + p.dirty_air, t_car)
        T = np.where(running, new_T, T)

    # Finishing order: classified cars by time; DNFs behind, those who lasted longer ahead.
    key = np.where(dnf_lap > laps, T, 1e9 - dnf_lap)
    positions = np.empty((sims, n), dtype=int)
    positions[rows, np.argsort(key, axis=1)] = np.arange(1, n + 1)[None, :]
    return Prediction([d.tla for d in inputs.drivers], positions)
