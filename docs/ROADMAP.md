# Pit Wall AI — Roadmap: an all-encompassing race strategy platform

Goal: use every publicly available data source to inform strategy calls — weather, tyres and
degradation, driving style, race state — live and in post-race review.

## Guiding principles

1. **Same weekend > same season > older seasons.** 2026 changed power units, aero, chassis and
   tyres, so car/tyre behaviour is measured from the current race weekend (practice long runs,
   the race so far) — never learned from pre-2026 cars. Older seasons are used only to test
   regulation-independent logic.
2. **Deterministic core, LLM on top.** Numbers come from data and explicit models; the LLM routes
   questions, explains calls and writes reviews. Every call states its inputs and assumptions.
3. **Measure before trusting.** Every model is scored on the 2026 benchmark (held-out races)
   against outcomes, not just against what teams did.
4. **Personal, non-commercial use.** The F1 TV token is local-only (see CLAUDE.md). Public demos
   run on archive replays and post-race data with an "unofficial" notice.

## Data inventory (verified 2026-09-29 unless marked)

| Source | What | Access | Status |
|---|---|---|---|
| F1 live timing — archive (`livetiming.formula1.com/static`) | Timing, gaps, stints, tyres, race control, track status, lap count, pit-lane times, team radio (MP3), weather, speed traps, **car positions** (`Position.z`), **car telemetry** (`CarData.z`: RPM, speed, gear, throttle, brake) | Public, finished sessions, 2024+ | Used: timing, tyres, radio, positions. Unused: CarData, WeatherData, TimingStats |
| F1 live timing — live (`signalrcore`) | Same topics in real time | Timing topics public; `Position.z`/`CarData.z` need the user's own F1 TV login | Client not built yet |
| OpenF1 (free historical) | Clean REST: laps w/ sectors, stints, pits, intervals, positions, weather, car data | Free after ~30 min post-session | Used for post-race chat |
| MultiViewer circuit API | Outline, rotation, corners, marshal sectors, **measured pit loss** (green/SC/VSC) per circuit-year | Public, unofficial; cached locally | Used |
| FIA Sporting Regulations | Rules, per issue in force | Public PDFs | Used (hybrid RAG) |
| Open-Meteo | Hourly forecast: rain probability, precipitation, temperature, wind at the circuit | Free, no key | Not yet |
| Jolpica (Ergast successor) | Results, grids, standings, circuits (lat/lon) | Free | Not yet |
| Pirelli compound nominations (which C1–C6 are hard/medium/soft each event) | Pirelli / FIA event documents | Public (to verify format) | Not yet — matters: "SOFT" is a different compound per event |
| FIA event documents | Stewards' decisions, penalties, event notes | Public PDFs (to verify) | Not yet |

## Phases (simulate-first — see `docs/SIMULATION.md` for the engine design)

The core of the platform is a race simulator: strategy decisions compare simulated futures,
data makes the simulation accurate, and the LLM explains results rather than judging strategy.

### 1. Correctness fixes
- Exclude SC/VSC/red-flag laps from all pace statistics (track-status timeline).
- "Last race" resolution and the compact whole-field race summary (token-limit fix).

### 2. Tyre and lap-time model from the current weekend
- Lap extraction from the feed (practice, race; archive and live).
- Per-compound degradation (+ cliff) from practice long runs, updated live in the race; fuel
  correction; per-driver pace and consistency; all with uncertainty.
- Map SOFT/MEDIUM/HARD to Pirelli's C-compounds for the event.

### 3. Race simulator
- Deterministic single run that reproduces a finished race from its real strategies (validation).
- Traffic/overtaking model per circuit, dirty air, SC/VSC mechanics, rules.

### 4. Monte Carlo decisions
- Options × ~1,000 simulations with common random numbers; expected points/position + risk.
- Replaces the rule-of-thumb pit calls; powers "best strategy", counterfactual and "what if the
  SC comes now?" questions in chat, and a pit-window chart in the UI.

### 5. Backtesting & evals (from phase 2 onwards)
- Lap-time prediction error; simulator finish-order accuracy; 2026 SC/VSC decision backtests
  scored on outcomes with calibration; leave-one-race-out.
- The original 20 chat scenarios: faithfulness and retrieval precision (DeepEval).

### 6. Weather
- Live `WeatherData`; Open-Meteo forecast → rain probability as a simulator input
  (crossover/intermediate calls); track-temperature effect on degradation.

### 7. Driving style (CarData.z)
- Braking, throttle, lift-and-coast, tyre-management signatures; feeds per-driver tyre model.

### 8. Live
- SignalR client + recorder; token-gated positions/telemetry for local use only.

### 9. Context sources
- Pirelli nominations, FIA stewards' documents, Jolpica results.
