# Pit Wall

Post-race Formula 1 strategy and rules analysis, built on public data.

Ask things like *"What if Hamilton had pitted under the VSC on lap 39 in Singapore?"* or
*"Should Antonelli have got a track limits penalty there?"* and get an answer worked out from
the race's own timing, the FIA Sporting Regulations in force that weekend, and the stewards'
published decisions. Numbers come from data or models, never from an LLM's memory.

Pit Wall is mainly an **MCP server**: any MCP client (Claude Code, Claude Desktop, ...) gets
race data and strategy models as tools, and the client's LLM does the reasoning. There's also a
built-in chat (LangGraph on Groq's free tier) and an HTTP API for the separate
[`pitwall-ui`](https://github.com/calebbratton/pitwall-ui) front end.

> Unofficial hobby project. Not affiliated with Formula 1, the FIA or any team.

## What it can do

- **Pit-stop reviews and what-ifs.** `review_pit_stop` rebuilds the race at the decision lap and
  simulates the alternatives with traffic: pit now, stay out, or a stop the driver never made.
  Tyre wear, fresh-tyre gain and pit loss are measured from that race (each car's own laps), and
  the result leads with a time ledger (stop cost vs time gained to the flag), with the real
  finish alongside.
- **Who wins from here.** Win, podium and expected-position chances from the race state at any
  lap, calibrated against every 2026 safety car and VSC.
- **Rules and penalties.** Hybrid search (dense + BM25) over the FIA Sporting Regulations,
  2024 to 2026, always filtered to the issue in force on race day. Stewards' decisions are
  scraped from the FIA documents site and searchable per car and race. Paddock jargon ("red
  flag", "undercut") is mapped to regulation wording before searching.
- **Race data.** Results, laps, stints, pit stops, running order and gaps by lap, race control,
  incidents (track-limits deletions, investigations, penalties) and weather.
- **Predictions.** Pre-race Monte Carlo predictions from qualifying and recent form, plus
  luck-adjusted results.

### The luck principle

Safety car, VSC and red flag timing (and retirements) are luck. Predictions leave them out, and
anything learned or scored from past races uses luck-adjusted results (`src/sim/luck.py`), so no
driver or team gains credit in the models for a lucky neutralisation.

### Season scope

The current season and the previous one (a rolling window from `src/seasons.py`). 2026 is the
main season; older races are kept only as test fixtures.

## MCP tools

| Group       | Tools |
|-------------|-------|
| Calendar    | `list_races`, `find_driver` (numbers, acronyms, first names, nicknames) |
| Race data   | `race_results`, `race_summary`, `lap_times`, `pace_summary`, `tyre_stints`, `pit_stops`, `positions_by_lap`, `gap_between`, `race_control`, `race_incidents`, `weather` |
| Rules       | `search_regulations`, `stewards_decisions` |
| Models      | `review_pit_stop`, `who_wins_from_here`, `predict_race`, `luck_adjusted_result`, `tyre_life` |

Every result is compact JSON with a `source`. Anything unknown comes back as `null` or an
`ERROR: ...` string listing the valid options, never a guessed value. The tools make no LLM
calls. The server also ships a `pitwall_rules` prompt with its answering rules.

## Getting started

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env                    # add a free GROQ_API_KEY (only the chat needs it)

python scripts/ingest_regulations.py    # download, chunk and index the FIA regs (a few minutes)
python scripts/ingest_decisions.py      # stewards' decisions (incremental)
python -m src.warehouse.refresh         # ingest finished sessions into the local warehouse
```

All storage is local: Qdrant in embedded mode, `fastembed` for embeddings, DuckDB for the
warehouse, and cached OpenF1 responses under `data/` (gitignored).

### Use it from an MCP client

Claude Code, over stdio:

```bash
claude mcp add pitwall -- uv run --directory ~/projects/pitwall python -m src.mcp_server
```

Or over HTTP, served by the API process:

```bash
uvicorn src.api:app --port 8000         # MCP at http://localhost:8000/mcp
```

The HTTP endpoint is stateless with JSON responses. It only accepts localhost `Host` headers
unless you list public hostnames in `PITWALL_MCP_HOSTS` (comma-separated) when hosting it.

### Built-in chat

```bash
python -m src.chat --trace              # terminal chat (Groq)
uvicorn src.api:app --reload --port 8000  # POST /api/chat (SSE) for pitwall-ui
```

The chat graph is `route → resolve → fetch ⇄ tools → retrieve → analyze → synthesize`. Only
`route`, `fetch` and `analyze` call an LLM; resolving the race, fetching key race events and
rendering the answer are deterministic, and citations of articles that weren't retrieved are
dropped.

## Data sources

| Source | Used for |
|--------|----------|
| [OpenF1](https://openf1.org) (free historical API) | Sessions, laps, stints, pit stops, positions, race control, weather |
| F1 live-timing feed (SignalR) and its static archive | Race timing for the simulators; own recordings of live sessions |
| FIA Sporting Regulations (PDFs) | Regulations index (`src/rag/sources.py`) |
| FIA documents site | Stewards' decisions (`src/rag/decisions.py`) |
| Open-Meteo | Weather forecasts |

Race timing for the models comes from a full local recording if there is one, then F1's static
archive (local cache first), then a rebuild from OpenF1 (`src/livetiming/sources.py`). The static
archive has returned 401 since 2026-10-16, so new races come from recordings or OpenF1. Each
review says which source it used.

To keep data current, run `python -m src.warehouse.refresh` on a schedule, or set
`PITWALL_REFRESH_MINUTES=30` in the API process. It's idempotent.

## Live tools (local only)

`src/livetiming/` has a live-race toolkit: a SignalR client and recorder, SC/VSC pit calls,
in-race predictions, battle and undercut alerts, team radio transcription (Groq Whisper) and a
track map. It's for personal local use, not the public product. Replays of finished sessions run
the same pipeline:

```bash
python -m src.live --year 2026 --meeting Azerbaijan --speed 60 --from-lap 29
```

A personal F1 TV token unlocks extra live topics. It's off by default, needs both
`F1TV_SUBSCRIPTION_TOKEN` and `PITWALL_USE_F1TV_TOKEN=1`, and is never used by a hosted server.

## Configuration

Secrets go in `.env` (see `.env.example`).

| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | `groq` (default) or `anthropic` (explicit opt-in only) |
| `LLM_MODEL`, `LLM_MODEL_<ROLE>` | Model for every role, or one of ROUTER / FETCHER / ANALYST / JUDGE |
| `GROQ_API_KEY` | Free key from console.groq.com, for the chat |
| `ANTHROPIC_API_KEY` | Only with `LLM_PROVIDER=anthropic` |
| `PITWALL_MCP_HOSTS` | Public hostnames allowed for `/mcp` when hosted |
| `PITWALL_REFRESH_MINUTES` | Refresh the warehouse from the API process |
| `PITWALL_UI_ORIGINS` | CORS origins for pitwall-ui (default `http://localhost:5173`) |
| `PITWALL_LIVE_AUTOSTART` | Follow the live feed when the API starts (local use) |

## Tests and evals

```bash
pytest tests --ignore=tests/evals       # fast and offline: fixtures, mock OpenF1, fake LLMs
python -m src.evals.run                 # benchmark: routing, retrieval precision, faithfulness
PITWALL_EVALS=1 pytest tests/evals      # the same as regression floors (calls a real LLM)
```

## Repository layout

```
src/
  mcp_server.py  MCP server (stdio, and /mcp inside the API)
  api.py         FastAPI app: chat SSE, live sessions, /mcp
  agents/        LangGraph chat graph and prompts
  tools/         OpenF1 client and telemetry tools
  rag/           Regulation chunking and hybrid search, stewards' decisions, jargon glossary
  sim/           Pit reviews, in-race and pre-race simulators, luck adjustment, learned models
  warehouse/     DuckDB warehouse: ingest, build, refresh
  livetiming/    Live feed, archive replay, race state, strategy calls, alerts
  models/        Tyre compounds and wear curves
  llm/           Chat-model factory (the only place provider SDKs are imported) and transcription
scripts/         Regulation and decision ingest, fixture recording
predictions/     Locked-in race predictions, written once per weekend
docs/            STATUS.md (current state), ROADMAP.md, SIMULATION.md
tests/           Offline unit tests; tests/evals/ hits a real LLM
```

See `CLAUDE.md` for architecture rules and conventions, and `docs/STATUS.md` for where things
stand.
