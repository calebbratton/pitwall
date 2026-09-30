# Race simulator — design

Strategy calls compare futures (pit now vs stay out, soft vs hard). The simulator produces those
futures; data makes it accurate; the LLM only explains its results.

Cost: pure Python/numpy on the local machine, no API calls. One decision ≈ 1,000 simulated races
× 22 cars × ~60 laps ≈ 1.3M lap evaluations, vectorised across simulations: ~1 s.

## 1. Objective

Maximise **expected finishing position / points**, not race time — track position matters.
Report distributions ("P4 or better: 70%"), not just point estimates.

## 2. Lap-time model

    lap_time(car, lap) = base_pace[car]
                       + tyre_deg[compound](tyre_age)      # linear, with a cliff past age_cliff
                       + fuel_effect * fuel_laps_remaining
                       + track_evolution(lap)
                       + dirty_air_penalty (if < 1 s behind another car)
                       + noise ~ N(0, consistency[car])

| Parameter | Source | Notes |
|---|---|---|
| `tyre_deg[compound]` (+ cliff) | Same weekend: practice long runs → updated lap by lap in the race | Pooled across drivers (compounds behave alike), per-driver offsets; kept **with uncertainty** |
| `base_pace[car]` | Race laps so far (tyre/fuel corrected); practice/qualifying before the start | |
| `fuel_effect` | Prior (labelled assumption), refined across stints once separable | Within one stint fuel and tyre age are collinear; needs pit-stop resets to separate |
| `consistency[car]` | Residual std of that car's clean laps | |
| `dirty_air_penalty` | 2026 laps with interval < 1 s vs clean air | |
| Pit loss green / SC / VSC | MultiViewer, measured per circuit-year | Already used by `strategy.py` |
| Overtaking model: P(pass \| pace delta) | 2026 races per circuit (position changes vs pace delta); fallback by circuit type | What makes Monaco ≠ Monza |
| SC / VSC hazard per lap, duration | Per-circuit neutralisation history, **2024–2026** | Incident rates depend on the circuit more than on the car rules, so older seasons help here |

Everything car-dependent comes from the current weekend (2026 regulations reset car behaviour);
circuit-dependent quantities may use older seasons.

## 3. One simulated lap (all simulations at once, numpy arrays shaped [sims, cars])

1. Neutralisation: draw SC/VSC start with the per-lap hazard; SC → everyone runs at SC pace and
   gaps compress to a queue over ~2 laps; VSC → +~35 % lap time, gaps preserved. Draw durations.
2. Pit decisions: each car's plan (candidate plan for the car being evaluated; likely plans for
   rivals) — pitting adds the green/SC/VSC loss and resets tyre age/compound.
3. Intended lap times from the model above.
4. Traffic: sort by cumulative time; a car closing on the one ahead passes with P(pass | delta),
   otherwise it's held behind (minimum gap) and pays the dirty-air penalty.
5. Rules: mandatory second dry compound (penalty if violated), retirements (small hazard).

Finish order = laps completed, then cumulative time.

## 4. Decisions

- Enumerate options (pit this lap / next laps / at lap X; compound choice; stay out).
- Simulate every option with **common random numbers** (same SC draws, same noise) so differences
  come from the decision, not luck.
- Rank by expected points / position; show risk (spread, worst plausible case) and the scenario
  that flips the call ("if a second SC comes, staying out wins").
- The same engine answers post-race questions: "best strategy" = actual result vs the best
  simulated alternative per driver; "SC luck" = result with vs without the neutralisation.

## 5. Validation (from the first version)

1. Lap-time model: predict the next N laps' times per car; report error by compound and age.
2. Simulator: start from lap L of a finished 2026 race with the real strategies; compare the
   predicted finish order and gaps with reality (rank correlation, mean position error).
3. Decisions: backtest pit calls at every 2026 SC/VSC using only data available then; score
   outcomes; check calibration (70 % claims should come true ~70 % of the time).
Leave-one-race-out throughout; never tune on the race being scored.

## 6. Build order

1. Correctness fixes: exclude SC/VSC laps from pace statistics; "last race" + race summary.
2. Lap extraction from the feed (same code for practice, race, archive, live).
3. Lap-time model fit with uncertainty (+ tests on synthetic data with known parameters).
4. Deterministic single-run simulator reproducing a finished race from its real strategies.
5. Traffic/overtaking and SC/VSC mechanics; per-circuit parameters.
6. Monte Carlo + option evaluation; wire into pit calls, chat tools and the UI (pit-window chart).
7. Backtest suite on 2026 (runs in CI on cached fixtures).
