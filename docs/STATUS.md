# Status — 2026-09-30 (updated overnight)

Where the project stands, what's running, and what's next. See `CLAUDE.md` for architecture
and rules, `docs/ROADMAP.md` for phases, `docs/SIMULATION.md` for the simulator design.

## Built and working
- **Post-race strategy chat** (LangGraph, Groq): `python -m src.chat` / UI "STRATEGY CHAT".
- **FIA regulations RAG** (2024–2026 issues, hybrid search, jargon glossary).
- **Warehouse** (DuckDB): 2026 + 2025 races, 2026 practice/quali/sprints. `python -m src.warehouse.build`.
- **Tyre model** (panel fit + stint fallback), `python -m src.tyre_report`.
- **Pre-race prediction** (Monte Carlo, luck-adjusted backtest, LORO-tuned):
  `python -m src.sim.predict --year 2026 --place "Kuala Lumpur" --laps 56`
  (2026 out-of-sample: rank corr 0.86 ≈ qualifying order; better win probabilities than grid).
- **Luck principle**: SC/VSC/DNF luck excluded from predictions and from anything learned.
- **Live**: F1 SignalR client + recorder, server-run live sessions, SC/VSC pit calls,
  "who wins from here" in-race sim, radio transcription, track map, LIVE-tab chat.
- **UI** (`~/projects/pitwall-ui`): chat, LIVE tab (map, tower, pit calls, predictions, radio,
  strategy chat). **Follows the real live session** (`/api/live/current` + stream, polls every
  30 s, auto-switches on a new live_id, de-duplicates catch-ups, idle note between sessions);
  test replays are a secondary control with "Back to live".
- **In-race predictor calibrated** (`python -m src.sim.inrace_backtest`): 57 checkpoints (every
  2026 SC/VSC + green checkpoints), leave-one-race-out → pace uncertainty 0.3 s/lap; winner
  log-loss 1.95 → 1.33, Brier 0.0408 → 0.0329. "99%" calls used to come true 68% of the time;
  ~85% calls now come true 86%. Still mildly overconfident in the 50–80% band.

## This weekend: Kuala Lumpur (C2/C3/C4)
- FP1 Fri 2 Oct 04:30 UTC · FP2 08:00 · FP3 Sat 04:30 · **Quali Sat 08:00** · **Race Sun 4 Oct 07:00**.
- Run the server following the live feed: `PITWALL_LIVE_AUTOSTART=1 uvicorn src.api:app --port 8000`
  (or `POST /api/live/start`) — it's running like this now. Open http://localhost:5173 → LIVE. Terminal view: `python -m src.live --live`. Recordings land in
  `data/livetiming/recordings/` — each session becomes a test fixture.
- After quali: ingest + predict:
  `python -m src.warehouse.ingest --years 2026 --sessions "Practice 1" "Practice 2" "Practice 3" Qualifying --settle-minutes 45`
  then `python -m src.warehouse.build` and the predict command above. Score it after the race.

## Overnight 2026-09-30 → 10-01 (hand-off)
Done and committed: pre-race probability calibration (`src/sim/calibrate.py`); eval suite
(`python -m src.evals.run`; 15/15 rules scenarios scored, faithfulness ~1.0 except
unsafe-release 0.7; race scenarios still running/pending OpenF1 budget); grid penalties
(`predict --penalty VER=10 --pit-lane HAM`) and a pre-race prediction from the official grid on
the live feed; rejoin projections + reference lap time in snapshots; weather + Open-Meteo
forecast events (KL Sunday: 98% rain); lap events. UI: track line (rejoin markers, battle),
widget dashboard (move / min / max / hide, saved per browser, 1-column on phones), weather bar,
run timeline.
Eval run (20 scenarios): faithfulness 0.98 (19 judged), retrieval hit rate 1.0 but precision
0.33 (reranker candidate), routing 0.95. Fixed from it: short race names ("Spa"), race-vs-rules
routing. Judge caveat: Qwen (reasoning off, truths limit 15 to fit Groq's 1000 output-token/min
cap) returns non-standard "borderline" verdicts and misses supported facts beyond the 15 it
extracts — e.g. it marked a correct Overtake claim (B7.2.3 c.ii) unsupported. Treat scores as a
lower bound; a better judge (or TypeSafe Jev, see chat 2026-09-30) is worth testing.
Wet setting and forecast mixing are in (`predict --rain P / --forecast`); WET_PARAMS provisional.
Historical study DONE (2023-26 all weekends in the warehouse, 85 races; results in
data/study_2023_2026.txt): team race pace in earlier races is the only predictor beyond
qualifying that helps every season (finish rank corr +0.069/+0.038/+0.007/+0.024; pace
+0.100/+0.046/+0.024/+0.010). Long runs, team race-vs-quali bias, driver Sunday gain: no.
In the simulator (SimParams.team_pace_weight) it improves order every season but worsens
calibrated winner log-loss 2024-26, so it's off by default — open question: use it for the
order/points but not the win probabilities, or find why favourites get overrated.
Wet: 8 wet races 2023-25; a wet setting doesn't beat dry out of sample, so forecasts don't
change predictions automatically (`predict --rain` is a what-if).

## Race-weekend dry run (2026-09-30) and FP1 checklist
Baku 2026 FP1 / qualifying / race replayed through the live code path. Fixed: missing archive
topics (LapCount in practice) crashed replays; no session header outside races; blank gaps in
practice/qualifying (now best-lap timing, qualifying per segment, KO); pit calls on a practice
VSC test (pit calls / predictions / rejoin are race-only); session changeover on one connection
(new live session per SessionInfo Key); venue-name mismatches (Sepang / Kuala Lumpur) for the
pre-race lookup and Pirelli C-numbers; calibrated chances kept ordered.
FP1 (Fri 2 Oct 04:30 UTC = Thu 23:30 CT): keep `PITWALL_LIVE_AUTOSTART=1 uvicorn src.api:app
--port 8000` and the UI running; it records to data/livetiming/recordings/. Expect: session header
"Practice 1", best-lap tower, weather + forecast, radio, run timeline; no track map unless
MultiViewer publishes circuit 12 (Position.z needs the F1 TV token live). Not testable in
advance: the real SignalR feed for a new session and the changeover FP1 -> FP2.

## Alert stream status (2026-10-01)
SC-call alerts on. Undercut alerts ON with the learned model (src/sim/undercut_model.py): 570
real first-stop undercut attempts 2023-26 (`python -m src.sim.undercut_data`, cached in
data/warehouse/undercut_attempts.jsonl), logistic regression, leave-one-race-out Brier 0.228 vs
0.251 same-season base rate, better every season, calibrated. Fires at P >= 0.6 held 2 laps:
~10 alerts over the 15 2026 races. The simulator version had no skill (Brier 0.234 vs 0.217 on
2026) - its "car ahead covers next lap" assumption holds only 141/587 times.
Next for the alert stream: factual alerts (rejoin position, oldest tyres, fastest lap, rain),
LLM phrasing, UI panel.

## Next (in order)
1. **Watch FP1 (Fri 2 Oct 04:30 UTC) on the live path** and fix whatever the real feed reveals
   (first live session through the new code). Recordings become fixtures for replays/tests.
2. ~~Tyre life from measured wear by C-number~~ — **done** (`python -m src.models.tyre_curves`):
   2026 race data shows no drop-off (teams pit first), so life = proven lower bound (C2 43, C3/C4
   39, C1/C5 31 laps); the in-race predictor and tyre notes use it. Follow-up: re-run the
   in-race calibration with curves fitted per fold (the 0.3 s/lap calibration predates them).
3. ~~Practice long-run signal~~ — **tested, negative**: team-level long runs, sprint weighting
   ×2/×5 and sprint-only race pace all leave the tuner at qualifying weight 1.0 (LORO rank corr
   0.860 either way). In 2026, qualifying pace already carries the race-pace information;
   `long_run_level="team"` is kept as an option.
   Why (measured 2026-09-30, fresh-tyre laps, driver x compound effects, FP 2023-26): FP2 lap
   times get ~4.7 s/hour *slower* through the session (quali sims on low fuel first, race sims on
   high fuel later). Fuel/programme differences (seconds) swamp track evolution (tenths), and
   neither is observable from timing alone. Use practice for tyre degradation (the per-lap slope
   within a run is largely fuel-independent) — the pit decision review needs that — not pace.
4. **AI additions agreed with the user (2026-09-30):** ~~pre-race probability calibration~~ —
   **done** (`python -m src.sim.calibrate`): one temperature per horizon, leave-one-race-out on
   2026; win log-loss 1.57 → 1.53, podium 0.286 → 0.245, points 0.736 → 0.426 (raw sim was very
   overconfident in the midfield). `predict` applies it (`--raw` to see the simulator's own
   numbers). With recency-weighted team race pace (added 2026-09-30, user chose to keep it)
   Madrid predicts ANT 35% / NOR 27%. Final setting (4000 sims x 3 seeds, 85 races): the DRIVER's
   recent race pace (user's call to question the team average) at weight 0.25: 58/85 winners vs
   52 for pole, calibrated winner log-loss 1.085 (none 1.159, team 1.100). Winner counts from
   1000-sim runs swing ~±4 by seed: decide on log-loss or use 4000+ sims.: Mercedes had the faster race car in the 13 earlier races
   and NOR's pole margin was 0.011 s. NOR is still the luck-adjusted winner it's scored against. Next: LLM-judge evals (DeepEval/Ragas:
   faithfulness + retrieval precision, Qwen on Groq as judge); then a regulations reranker, Bayesian
   tyre priors (PyMC/NumPyro), radio → structured events, live anomaly detection, LightGBM
   rival pit model, tracing (Langfuse/LangSmith), voice, vision.
4b. **Historical predictor study (ROADMAP §4c):** ingest 2023–2025 all sessions (started
   2026-09-30: `python -m src.warehouse.ingest --years 2025 2024 2023 --sessions Race
   Qualifying "Practice 1" "Practice 2" "Practice 3" Sprint "Sprint Qualifying" "Sprint
   Shootout"`, resumable; then `python -m src.warehouse.build`), rank predictors with
   forward-chaining per season, settle the 2026 model; then rain scenarios, grid penalties,
   strategy engine, like-track overtaking.
5. Later: rejoin predictor / undercut check, rival pit-timing model (north star).
6. After those: **season chat mode + like-track priors** (roadmap §9b). Track-fit analysis is
   built as a CLI: `python -m src.analysis.track_fit --team McLaren`.

## Known caveats
- Kuala Lumpur: no circuit history (last F1 race 2017); race distance assumed 56 laps.
- MultiViewer has no data for some new circuits (Madrid) → default pit loss.
- Groq free tier: ~8K tokens/min per model; prompts are budgeted.
- Commit only when tests pass (`pytest -q` ≈ 3 s; tests never touch the network).
