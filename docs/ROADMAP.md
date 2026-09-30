# Pit Wall AI — Roadmap: an all-encompassing race strategy platform

Goal: use every publicly available data source to inform strategy calls — weather, tyres and
degradation, driving style, race state — live and in post-race review.

## North star: predict rivals' strategy, then beat it

"If I'm managing Red Bull, when will McLaren stop — and when should we go to undercut them?"
Everything below builds toward this:

1. **Rival pit-timing model** — for every rival, every lap: P(pits this lap). A per-lap
   ("discrete-time hazard") model: each lap of each 2026 race is one pit-or-stay decision, so a
   season gives thousands of examples even though there are only ~15 races. Features:
   - tyre state: compound, age, measured degradation vs the stint's start, laps remaining,
     whether the second compound is still owed
   - threats: gap to the car behind/ahead vs pit loss (undercut exposure), traffic at pit exit
   - race state: SC/VSC, rain, rivals who just pitted (covering behaviour)
   - team tendencies: per-team effects learned from 2026 (e.g. reacts to undercuts within a lap,
     double-stacks under SC, stops early vs extends) — shrunk toward the field average when
     a team has little data
   - live signals: radio ("box", "tyres are gone"), pit-lane readiness where public
2. **Reaction model** — P(rival covers | we pit), from 2026 undercut attempts: the difference
   between "they'll stop on lap 22" and "they'll stop the lap after we do".
3. **Counter-strategy search** — simulate our candidate pit laps against the rival's predicted
   (and reactive) behaviour; pick the lap that maximises P(we come out ahead) / expected points.
   UI: "UNDERCUT ON NOR: box lap 21 → 64% ahead after their stop; they cover within 1 lap 40% of
   the time".
4. **Validation** — predict rivals' actual pit laps in held-out 2026 races (log-loss,
   calibration, error in laps) against a baseline of "median stint length for that compound";
   backtest counter-strategy calls against what happened.

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

### 2. Data warehouse (DuckDB) — numbers in SQL, text in Qdrant
- Local DuckDB/Parquet store of 2024–2026 races: laps (tagged: neutralised?, compound, tyre age,
  stint, pit in/out, position, gap ahead, weather), stints, pit stops, results, race control.
  Source: OpenF1 free historical (rate-limited; one-time, cached) — live laps from the feed use
  the same schema later.
- Chat gets curated warehouse query tools (small aggregate results = cheap tokens); optional
  read-only SQL with row limits. A vector DB is for text (radio, race control, stewards, race
  reports), not for numbers.

### 3. Tyre and lap-time model from the current weekend
- Per-compound degradation (+ cliff) from practice long runs, updated live in the race; fuel
  correction; per-driver pace and consistency; all with uncertainty. Fitting runs on the warehouse.
- Map SOFT/MEDIUM/HARD to Pirelli's C-compounds for the event.
- First consumers: **undercut/overcut check** (two-car: fresh-tyre gain vs rival's old tyres,
  out-lap warm-up, gap, until rival responds) shown per close battle in the LIVE tab and
  backtested on 2026 undercut attempts; **strategy-gain metric** (places gained beyond pace,
  excluding retirements ahead, vs a pace-expected finish rather than the grid).

### 3b. Pit-wall tools on the tyre model (early visible wins)
- **Rejoin predictor**: "box now → P7, 1.2 s behind OCO, in traffic for ~4 laps", shown as a
  ghost car on the track map. Building block for the undercut check.
- **Pit stop time distribution per team** (stationary time + slow-stop risk) from pit data.

### 4. Race simulator
- Deterministic single run that reproduces a finished race from its real strategies (validation).
- Traffic/overtaking model per circuit, dirty air, SC/VSC mechanics, rules.

### 4b. Opponents, incidents and 2026 specifics
- **Rival pit-timing and reaction models** (see North star): rivals simulated as likely,
  reactive behaviour, not fixed plans.
- **SC/VSC hazard per circuit and lap** (lap 1, restarts, street circuits) from 2024–2026.
- **2026 energy/overtaking**: battery-deployment signatures from speed traps / CarData feed the
  overtaking model.
- **Live state estimation**: Kalman-filtered gaps and pace for stable projections.
- **Radio → structured signals**: small LLM extraction of events ("tyres gone", damage, "box")
  from transcripts, used as model inputs.

### 5. Monte Carlo decisions
- Options × ~1,000 simulations with common random numbers; expected points/position + risk.
- Replaces the rule-of-thumb pit calls; continuous green-flag pit windows; VSC-specific timing
  (a VSC can end before the car reaches pit entry); powers "best strategy", counterfactual and "what if the
  SC comes now?" questions in chat, and a pit-window chart in the UI.

### 6. Backtesting & evals (from phase 3 onwards)
- Lap-time prediction error; simulator finish-order accuracy; 2026 SC/VSC decision backtests
  scored on outcomes with calibration; leave-one-race-out.
- The original 20 chat scenarios: faithfulness and retrieval precision (DeepEval).
- Expert-consensus checks: per race, a few claims from post-race coverage (e.g. Baku 2026: the
  medium→soft switch under the SC beat soft→medium; Lindblad's gains were half retirements)
  that the tool's analysis should agree with.

### 7. Weather
- Finding (2026 races, per-lap track effects from the panel fit): a per-lap trend (fuel +
  rubber, −0.02…−0.09 s/lap) explains 80–95 % of track evolution; within-race track temperature
  adds ~nothing and its apparent effects are implausible/confounded (temperature moves with the
  lap count). So: evolution comes from the race's own lap effects live; weather adjustments
  only for big swings (clouds, 8–10 °C) and rain (rubber reset); estimate a pooled s/°C across
  races and weekend sessions (FP/Q/race at different temperatures) rather than per race.
- Rain radar nowcasting + crossover-lap model for intermediates.
- Live `WeatherData`; Open-Meteo forecast → rain probability as a simulator input
  (crossover/intermediate calls); track-temperature effect on degradation.

### 8. Driving style (CarData.z)
- Braking, throttle, lift-and-coast, tyre-management signatures; feeds per-driver tyre model.

### 8b. Strategist controls
- "Assume the car behind 1-stops", "assume an SC on lap 40": edit assumptions and re-run
  (strategy trees). Benchmark UX: broadcast "F1 Insights" graphics, but with assumptions and
  probabilities shown.

### 9. Live
- SignalR client + recorder; token-gated positions/telemetry for local use only.

### 10. Context sources
- Pirelli nominations, FIA stewards' documents, Jolpica results.
