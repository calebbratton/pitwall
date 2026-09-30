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

## Next (in order)
1. **Watch FP1 (Fri 2 Oct 04:30 UTC) on the live path** and fix whatever the real feed reveals
   (first live session through the new code). Recordings become fixtures for replays/tests.
2. **Tyre life from measured wear / drop-off by C-number** (stint lengths are strategy-driven,
   not physical) — `reference/pirelli_nominations_2026.json` has all 2026 C-numbers.
3. Improve the practice long-run signal (team-level, fuel-corrected, sprint-weighted) so
   pre-race predictions beat qualifying order.
4. **AI additions agreed with the user (2026-09-30):** LLM-judge evals (DeepEval/Ragas:
   faithfulness + retrieval precision, Qwen on Groq as judge) and probability calibration of the
   pre-race model (isotonic / temperature scaling) first; then a regulations reranker, Bayesian
   tyre priors (PyMC/NumPyro), radio → structured events, live anomaly detection, LightGBM
   rival pit model, tracing (Langfuse/LangSmith), voice, vision.
5. Later: rejoin predictor / undercut check, rival pit-timing model (north star).
6. After those: **season chat mode + like-track priors** (roadmap §9b). Track-fit analysis is
   built as a CLI: `python -m src.analysis.track_fit --team McLaren`.

## Known caveats
- Kuala Lumpur: no circuit history (last F1 race 2017); race distance assumed 56 laps.
- MultiViewer has no data for some new circuits (Madrid) → default pit loss.
- Groq free tier: ~8K tokens/min per model; prompts are budgeted.
- Commit only when tests pass (`pytest -q` ≈ 3 s; tests never touch the network).
