# Locked-in race predictions

User standard (2026-10-01): every race weekend gets two predictions, written by the live server
the moment they're made and committed here so they can't be revised:

- `<location>-pre-Q3.json` — as Q3 starts: practice, Q1/Q2 times, season form; grid not final.
- `<location>-grid.json` — when the race feed publishes the official grid (~1 h before the start).

Each file holds the time, the model settings and every driver's expected position and win /
podium / points chances. Score them after the race with `python -m src.sim.scorecard` (against the
luck-adjusted result and the raw result). Files are written once and never overwritten.
