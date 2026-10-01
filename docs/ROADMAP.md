# Pit Wall AI — Roadmap: an all-encompassing race strategy platform

Goal: use every publicly available data source to inform strategy calls — weather, tyres and
degradation, driving style, race state — live and in post-race review.

## Product (user decision, 2026-09-30): a watch companion
A second screen next to F1 TV, not a replacement: F1 TV shows what is happening; Pit Wall AI says
what matters and whether calls were right. **Push, don't pull** (the user's favourite framing):
nobody wants to stare at another dashboard during a race; the value is a short stream of alerts at
the moments that matter — "SC: Norris should pit, Russell stay out." / "Leclerc's undercut on
Hamilton is on." / "Piastri's softs are past anything seen this year." Alerts are event-triggered
(not on an interval), fire only when material, persist 2 laps and update in place, are ranked
under a rate budget, and are resolved afterwards ("undercut worked, +1"); every alert type needs a measured hit rate before it's shown.
- **Live dashboard:** timing tower, SC/VSC alert stream (pit calls, who wins from here), radio
  transcripts (follow one driver), live strategy chat.
- **Post-race chat:** "was pitting the right call for <driver>?" answered with measured numbers,
  not LLM guesses.
- **Pit decision review engine** (shared by both chats): rebuild the race state before the
  decision (archive replay / live per-lap snapshots), measure tyre deltas from that race (fresh
  vs old from cars that stopped, the driver's own degradation, C-number life, measured pit
  loss), simulate pit now vs stay out 1-N laps vs no stop, report expected position + P(gain) +
  the deltas used. Validate against what happened after real stops (85 races) before trusting.
- **What's unique vs MultiViewer / f1-dash / F1 TV** (timing towers and maps are table stakes):
  calibrated SC calls + win odds, rejoin markers for any car, transcribed radio for the followed
  driver, grounded Q&A with cited regulations, luck-adjusted pace, "was it the right call".
- **Product backlog (agreed 2026-10-01), target the KL race Sun 4 Oct 07:00 UTC:**
  1. Alert stream v1: SC calls + undercut/overcut alerts from the pit-review engine, each with a
     follow-up ("it worked: +1"); event-triggered, materiality thresholds, hit rate backtested.
  2. "Your call" mode: at an SC the viewer picks PIT / STAY OUT for their driver before seeing
     the engine's call, then the outcome — viewer vs engine vs race.
  3. Post-race verdict cards per driver strategy (right / wrong / unlucky, with the numbers) and
     one-line "what just happened" after big swings.
- Pre-race order prediction is near its ceiling (qualifying explains most of it): frozen as a
  panel, not the focus.

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
- **Historical priors, with measured transfer**: fit 2024–25 races with the same model and
  measure how well a circuit's past degradation predicts 2026 at the same circuit. If it does,
  use it as a prior with a fitted scale factor and spread (most valuable before FP2 at a
  circuit, e.g. Kuala Lumpur/Singapore); the weekend's own data then updates it. Circuit
  severity and tyre-curve shape should transfer; absolute rates and compound gaps won't.
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

### 4c. Learn the model's structure from 2023–2025, fit 2026 (user decision, 2026-09-30)
2026 alone is 15 races: too few to tell which predictors matter. Ground-effect seasons
2023–2025 (~70 weekends, OpenF1; 2022 isn't on OpenF1 and F1's 2022 archive index is not public)
decide **which predictors and functional forms** to use; 2026 data then sets the magnitudes
(team pace), with history as shrinkage priors where 2026 is thin.
- Candidates: qualifying gap, grid, practice long runs (driver/team), sprint pace, rolling team
  form, driver race-vs-qualifying tendency, circuit type (overtaking, low drag), weather, tyre
  wear by C-number, lap-1 gains, grid penalties.
- Car-independent mechanics fitted on history: pit loss, SC/VSC hazard per circuit,
  overtaking difficulty per circuit, lap-1 position changes, tyre-curve shapes.
- Protocol: forward-chaining (predict each race from earlier races only), luck-adjusted
  targets, and a predictor must help **in each of 2023, 2024 and 2025 separately** to count.
  Then it must still improve 2026 out of sample, or it's dropped.
- Internal only: users still see the current and previous season.
- Then, in order: rain scenarios (forecast-driven), grid penalties, strategy engine (per-car
  optimal stops from tyre curves + pit loss instead of random windows), like-track overtaking.

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
- Public telemetry shows how a car is driven (speed, throttle, brake, gear), not tyre state
  (temperatures/pressures aren't public): telemetry→wear links are learned against the tyre
  model's measured degradation; 2024–25 as a prior, re-checked on 2026 (energy management
  changed throttle traces).
- Braking, throttle, lift-and-coast, tyre-management signatures; feeds per-driver tyre model.

### 8b. Strategist controls
- "Assume the car behind 1-stops", "assume an SC on lap 40": edit assumptions and re-run
  (strategy trees). Benchmark UX: broadcast "F1 Insights" graphics, but with assumptions and
  probabilities shown.

### 9. Live
- SignalR client + recorder; token-gated positions/telemetry for local use only.

### 9a. Pit-wall screens for watching (user reference, 2026-09-30)
The user shared photos of real team pit-wall screens (timing towers, run timeline, track map with
pit-rejoin markers, track line, weather/race control). Adopt, in order:
1. **Rejoin markers**: where the followed car would come out if it pitted now (green / SC /
   VSC loss) and who it would be behind — the rejoin predictor with a picture.
2. **Track line**: the field on one line by gap, pit-loss window marked. Built from timing
   gaps, so it works live without the F1 TV token (Position.z is gated live).
3. **Run timeline**: lap × driver grid of lap times coloured best / normal / slow / pit.
4. **Battle panel**: followed car's gaps ahead/behind with a few-lap trend, Overtake range (<1 s).
5. **Weather strip**: WeatherData (track/air temp) + the Open-Meteo race-window rain forecast.

### 9b. Season analysis in chat (after the items above)
- **"Season" chat mode**: route season-wide questions ("what type of track suits McLaren this
  season?", "which team is best on high-speed tracks?") to `src/analysis/track_fit.py`
  (measured circuit features vs luck-free team form; built, CLI only today) and have the
  analyst explain the numbers.
- **Like-track priors** (measured 2026-09-30: finishing order mostly follows qualifying pace,
  which already captures car–track fit; overtaking difficulty only matters at low-drag tracks —
  Monza 0.90 vs 0.79, Spa 0.98 vs 0.92): a circuit-type overtaking setting, first for the
  in-race predictor; circuit-type priors for tyre severity and SC likelihood; early-week
  predictions before practice. Richer circuit features from telemetry (full-throttle share,
  slow vs fast corners).
- Investigate Silverstone 2026 (pre-race rank correlation 0.35 — likely weather).

### 10. Context sources
- Pirelli nominations, FIA stewards' documents, Jolpica results.
