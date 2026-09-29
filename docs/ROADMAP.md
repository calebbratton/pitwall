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

## Phases

### 1. Tyre model from the current weekend
- Fit per-compound degradation (s/lap of tyre age) and fuel correction from green-flag laps:
  first from **practice long runs** (FP1–FP3 of the same weekend), then updated live with the
  race so far. Per-driver adjustments when enough laps exist.
- Map SOFT/MEDIUM/HARD to Pirelli's C-compounds for the event.
- Replace the fixed "fresh ≤ 5 / old ≥ 15 laps" rules in `strategy.py` with the expected time
  gain of fresh tyres over the remaining laps vs the measured pit loss.

### 2. Weather
- Live `WeatherData` (track/air temp, rain, wind) in snapshots and the UI.
- Open-Meteo forecast for the race window: rain probability → crossover/intermediate calls.
- Track-temperature effect on degradation (from the weekend's own laps).

### 3. Race simulator
- Lap-by-lap projection per driver: tyre model + fuel + pit loss + traffic (dirty-air/overtaking
  difficulty per circuit). Answers: optimal pit window, undercut/overcut value, one- vs two-stop,
  "what if the SC comes out now?".
- Monte Carlo over safety-car probability (from 2026 incident rates per circuit type).

### 4. Driving style (CarData.z, archive + own-login live)
- Per-driver: braking points, minimum corner speeds, throttle application, lift-and-coast,
  tyre-management signatures; compare teammates and stints. Feeds the per-driver tyre model.

### 5. Benchmark & evals (runs throughout)
- 2026 safety-car/VSC events (~30 so far, +1 weekend at a time via the recorder) scored on
  outcomes (positions ±5 laps and at the finish), leave-one-race-out.
- The original 20 chat scenarios: faithfulness and retrieval precision (DeepEval).

### 6. Live
- SignalR client + recorder (every live session becomes a replayable fixture).
- Token-gated positions/telemetry for local use only.

### 7. Context sources
- Pirelli nominations, FIA stewards' documents (penalties change strategy), Jolpica results.
