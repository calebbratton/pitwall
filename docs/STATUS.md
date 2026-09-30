# Status — 2026-09-30

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
  strategy chat). Currently only plays **test replays** — no "follow live" mode yet.

## This weekend: Kuala Lumpur (C2/C3/C4)
- FP1 Fri 2 Oct 04:30 UTC · FP2 08:00 · FP3 Sat 04:30 · **Quali Sat 08:00** · **Race Sun 4 Oct 07:00**.
- Run the server following the live feed: `PITWALL_LIVE_AUTOSTART=1 uvicorn src.api:app --port 8000`
  (or `POST /api/live/start`). Terminal view: `python -m src.live --live`. Recordings land in
  `data/livetiming/recordings/` — each session becomes a test fixture.
- After quali: ingest + predict:
  `python -m src.warehouse.ingest --years 2026 --sessions "Practice 1" "Practice 2" "Practice 3" Qualifying --settle-minutes 45`
  then `python -m src.warehouse.build` and the predict command above. Score it after the race.

## Next (in order)
1. **UI "follow live" mode** (small; for the UI repo): on load, `GET /api/live/current`; if a
   live session exists, subscribe to `GET /api/live/stream?live_id=` (same events as replays,
   first `live` event with `source: "live"`, catch-up then live); keep test replay as fallback.
2. **Calibrate the in-race predictor**: backtest "who wins from here" at every 2026 SC/VSC
   (luck-adjusted); it's overconfident (~99% with 20 laps left).
3. **Tyre life from measured wear / drop-off by C-number** (stint lengths are strategy-driven,
   not physical) — `reference/pirelli_nominations_2026.json` has all 2026 C-numbers.
4. Improve the practice long-run signal (team-level, fuel-corrected, sprint-weighted) so
   pre-race predictions beat qualifying order.
5. Later: rejoin predictor / undercut check, rival pit-timing model (north star), evals.

## Known caveats
- Kuala Lumpur: no circuit history (last F1 race 2017); race distance assumed 56 laps.
- MultiViewer has no data for some new circuits (Madrid) → default pit loss.
- Groq free tier: ~8K tokens/min per model; prompts are budgeted.
- Commit only when tests pass (`pytest -q` ≈ 3 s; tests never touch the network).
