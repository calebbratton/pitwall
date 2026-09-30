# Pit Wall AI

Post-race strategy review engine for Formula 1. It takes a natural-language strategy question
(e.g. "Could McLaren have undercut on lap 30 at Monaco 2024?"), pulls race telemetry from the
OpenF1 API, retrieves the relevant FIA Sporting Regulation clauses via hybrid RAG, and returns a
grounded strategic verdict.

This is a hobby/portfolio project run by one person. **Optimize for $0 running cost**. LLM inference is
hosted (the dev machine is too weak for local models); everything else runs locally. Don't add paid services, hosted infra, or cloud dependencies unless asked.

**Resuming work?** Read `docs/STATUS.md` (current state, run commands, next steps).

**Season scope (user decision):** users get the **current season and the previous one** only —
a rolling window from `src/seasons.py` (`supported_seasons()`), enforced in the chat graph, the
API (`GET /api/seasons`) and warehouse ingest defaults. Older races (Monaco/Miami 2024) remain
as test fixtures only.

**Luck principle (user decision):** safety car / VSC / red flag timing and retirements are
unpredictable, so (1) race predictions leave them out, and (2) **anything learned or scored from
past results uses luck-adjusted outcomes** (`src/sim/luck.py`) — a driver or team must never gain
favour in the model from neutralisation luck (e.g. Antonelli's Madrid 2026 win is Norris's in
the adjusted result). The live pit-wall tools still model SC/VSC, because there they're the
situation being decided on.

**Direction:** an all-encompassing race strategy platform built on public data. See
`docs/ROADMAP.md` (data inventory, principles, phases) before starting new features.

## Stack

| Concern         | Choice                                                                 |
|-----------------|------------------------------------------------------------------------|
| Language        | Python 3.11+ (managed with `uv`; system Python is 3.9, don't use it)   |
| Orchestration   | LangGraph (stateful graph; Route → Fetch → Retrieve → Analyze → Synthesize) |
| LLM             | Provider-agnostic via `src/llm/factory.py`. Default: **Groq** free tier, one model per role (see below). Optional: Anthropic. No local LLMs — dev machine can't run them |
| Vector store    | Qdrant in **embedded/local mode** (`QdrantClient(path=...)`), no server |
| Embeddings      | Local via `fastembed` (dense + BM25 sparse for hybrid search), no API cost |
| Structured data | OpenF1 REST API (free, no key) — `/stints`, `/laps`, `/race_control`   |
| Evals           | DeepEval (Faithfulness, contextual precision) + plain pytest assertions |

## Directory layout

```
src/
  agents/   LangGraph state, nodes, and graph.py (the state machine)
  tools/    OpenF1 client + tool functions exposed to the LLM (plus mock client)
  rag/      Regulation ingestion, chunking, hybrid retrieval
  llm/      Chat-model factory; the ONLY place provider SDKs are imported
scripts/    One-off setup scripts (download/chunk FIA regs, build index)
data/       Local artifacts: raw regs, Qdrant storage, cached OpenF1 JSON (gitignored)
tests/      Unit tests (no network, no LLM)
tests/evals/ Benchmark scenarios + DeepEval suites (hit a real LLM; run explicitly)
```

## Graph state

The graph state is a `TypedDict` with at least:
`current_query`, `race_context` (year, meeting_key, session_key), `fetched_telemetry_json`,
`retrieved_rules_text`, `evaluation_steps`. Nodes return partial state updates; never mutate
state in place.

## Graph (`src/agents/graph.py`)

`route → resolve → fetch ⇄ tools → retrieve → analyze → synthesize`, with a `rules` path that
skips telemetry and an error path (ambiguous/future race) straight to `synthesize`.

- LLM nodes: `route` (router), `fetch` (fetcher, tool calling), `analyze` (analyst). Everything
  else is deterministic on purpose: `resolve` maps year/place → OpenF1 session + regs issue,
  key race events (red flags, SC, VSC) are always fetched in code, and `synthesize` renders the
  answer from the structured `Analysis` and **drops citations of articles that weren't retrieved**.
- Groq models per role (`GROQ_DEFAULTS`): router `gpt-oss-20b`, fetcher `qwen3.8-27b` (the only
  one that makes parallel tool calls), analyst `gpt-oss-120b`, judge `qwen3.8-27b`. Free-tier
  limits are per model (8K tokens/min each), so splitting roles multiplies the budget.
- Use `with_schema()` for structured output, not `with_structured_output()` directly: Groq's
  tool-call route 400s when the model answers in prose.
- Fetch is capped at `MAX_FETCH_ROUNDS`. Telemetry tools validate every argument and return
  `ERROR: ...` strings (with valid options) instead of raising, so the model can retry.
- Chat memory: `memory_checkpointer()` (in-process). `messages` and `race_context` persist across
  turns; per-question fields are reset by `route`.
- Tests use scripted fake models (`tests/test_graph.py`); never call a real LLM in `tests/`.

## HTTP API (`src/api.py`) and UI

- `uvicorn src.api:app --reload --port 8000`. `POST /api/chat` streams SSE events `thread`,
  `step` (one per graph node), `answer`, `error`; `GET /api/health`. The contract is documented
  in the module docstring and consumed by the sibling repo `~/projects/pitwall-ui` — change both
  together. CORS origins via `PITWALL_UI_ORIGINS` (default `http://localhost:5173`).
- `create_app(make_graph)` takes a graph factory so `tests/test_api.py` runs with fake models.

## Live timing (`src/livetiming/`) — the live race tool

- Data source for live and replay is **F1's own live-timing feed**, not OpenF1 (OpenF1 live is
  paid; OpenF1's free historical API stays the source for post-race analysis in the graph).
  - Live: SignalR Core at `wss://livetiming.formula1.com/signalrcore`, hub `Streaming`. Timing
    topics work **without** an F1 TV token; only `CarData.z`/`Position.z` (and `PitStop`) are
    gated. The live client is not built yet — first real test is a live session.
  - Archive: `livetiming.formula1.com/static/<path>/<Topic>.jsonStream` has the same messages
    with offsets, for finished sessions (2024+, incl. 2026). Everything is built/tested on it.
- Pipeline: `archive.py` (messages) → `state.py` (merge partial updates: dict merge, list by
  index, `_deleted`) → `snapshot.py` (`RaceSnapshot`, source-independent) → `strategy.py`
  (deterministic SC/VSC PIT / STAY OUT calls; rules of thumb are named constants, echoed as
  assumptions) → `monitor.py` (events; `replay()` at any speed with fast-forward).
- Team radio: F1 publishes a curated subset of clips (MP3s, public). `src/llm/transcribe.py`
  transcribes them with Groq Whisper (free tier), cached in `data/livetiming/radio_transcripts.v<N>.json` (bump the version in
  `transcribe.py` when settings change). Prompt = one natural sentence with the speaking driver's
  full name (measured best; vocabulary lists got recited back). Segments failing Whisper's
  confidence gates are dropped.
  `radio` events go out immediately (`text` null unless cached); `radio_transcript` follows.
- **Live feed** (`client.py`): `LiveTimingClient` speaks SignalR Core (negotiate + cookies,
  handshake, `Subscribe`, pings, jittered reconnect that re-subscribes) and yields the same
  `Message`s as the archive; `Recorder` saves every session under `data/livetiming/recordings/`
  in archive format (a recording replays via `ArchiveSession(<folder>, cache_dir=<root>)`).
  Token topics (`Position.z`/`CarData.z`) only with `F1TV_SUBSCRIPTION_TOKEN` AND
  `PITWALL_USE_F1TV_TOKEN=1`.
- **One pipeline for live and replay:** `monitor.pump()` turns (message, quiet) into events;
  `replay()` and the live runner both use it. Live sessions (`session.py`) are server-owned:
  `POST /api/live/start` (or `PITWALL_LIVE_AUTOSTART=1`) runs the feed as a background task that
  broadcasts to `GET /api/live/stream?live_id=` subscribers (late joiners get a catch-up);
  `GET /api/live/current`. Replays via `/api/live/replay` create a session too (test source).
- **In-race** (`src/sim/inrace.py`): "who wins from here" from the current state (recent pace,
  effective in-race wear, tyre life, owed stops taken under a current SC/VSC, SC pit stops from
  real gaps then the queue). Emitted as `prediction` events at each SC/VSC; the LIVE tab chat
  (`POST /api/live/ask`, `src/agents/live_chat.py`) answers from it with one LLM call.
- Try it: `python -m src.live --live` (live feed) or
  `python -m src.live --year 2026 --meeting Azerbaijan --speed 60 --from-lap 29`
  (two safety cars) or `--year 2024 --meeting Miami --from-lap 26`. API: `GET /api/live/sessions`,
  `GET /api/live/replay?path=&speed=&from_lap=&transcribe=` (SSE; UI's LIVE tab).
- Track map: `track.py` traces the circuit outline from the leader's `Position.z` path over lap 3
  and the replay emits `track` (once) and `positions` (~4/s) events. `Position.z` is public in
  the archive but gated behind an F1 TV login on the live feed.
- **F1 TV token policy (user decision):** the user's personal F1 TV token (`F1TV_SUBSCRIPTION_TOKEN`
  in `.env`) is for the user's own local use only. Never use it to serve data to other people,
  never send it to a browser, never store it server-side in a deployed app. Token-gated features
  are off by default and enabled only by a local setting. A public deployment must use archive
  replays / post-race data, or have each viewer sign in with their own F1 TV account.
- Tests use `tests/fixtures/livetiming/miami_2024_sc_lap28.json` (state at the lap-28 SC) and
  synthetic messages; never hit the archive or Groq in `tests/`.

## Regulations (RAG)

- **2026 is the primary season** (the user analyses races as this season happens). 2024-2025
  are indexed too for historical benchmarks.
- Sources live in `src/rag/sources.py`. Each season has several issues; `source_for_race()` picks
  the issue in force on race day, and every search is filtered to that season + issue. When the FIA
  publishes a new issue, add it there and re-run `scripts/ingest_regulations.py`.
- Two document formats, two parsers in `src/rag/chunking.py`: `classic` (2023-2025, `30.5`) and
  `section_b` (2026+, `B6.3.6`). The 2026 PDF renders the "ff" ligature as `‘`/`W`; it's fixed
  in `_fix_ligatures`. Appendices are not indexed.
- One chunk per clause (long ones split into parts); `RegChunk.citation` is what answers must cite.
- `src/rag/glossary.py` rewrites paddock jargon into regulation vocabulary ("red flag" →
  "suspension", "undercut" → "pit stop tyre change"). Substitute, don't append: appended words
  dilute the embedding. `retrieve` merges all query variants with reciprocal rank fusion.
- Ingest takes a few minutes on this machine (CPU embeddings); searches are fast.

## OpenF1 notes

- `get_session(year, place)` matches country, location or circuit, accent-insensitive. Several
  countries host multiple races (2026: Spain = Barcelona + Madrid; USA = Miami, Austin, Las
  Vegas), so an ambiguous country raises with the options rather than guessing.
- OpenF1 lists the full season calendar, including races that haven't happened yet (no data).

## Conventions

- **LLM provider isolation:** nodes get a model from `src/llm/` (selected by `LLM_PROVIDER` /
  `LLM_MODEL` env vars). Never import `groq`, `anthropic`, or `langchain_*` provider
  packages outside `src/llm/`.
- **Tool calling:** tools are plain typed Python functions with docstrings; bind them with
  LangChain's `bind_tools`. Keep tool outputs compact (trim OpenF1 payloads to needed fields) —
  small models and free-tier token limits punish large contexts.
- **No hallucinated numbers:** every lap time, compound, and lap number in a final answer must
  come from `fetched_telemetry_json`. Every regulation claim must cite an article from
  `retrieved_rules_text`. This is what the Faithfulness eval checks.
- **Caching:** cache OpenF1 responses to `data/cache/` keyed by URL. Historical race data never
  changes, and caching keeps tests/evals fast and polite to the free API.
- **Mocks:** unit tests use the mock OpenF1 client with fixtures; they must not hit the network.
- Type hints everywhere; Pydantic models for OpenF1 records and tool argument schemas.
- Secrets live in `.env` (gitignored); `.env.example` documents every variable.

## Commands

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
python scripts/ingest_regulations.py   # download + chunk + index FIA regs (~minutes)
python scripts/record_fixtures.py --year 2024 --place Monaco --drivers 4 81 16
python -m src.chat --trace            # chat in the terminal (uses Groq)
uvicorn src.api:app --reload --port 8000  # API for pitwall-ui
pytest tests --ignore=tests/evals      # fast, offline
python -m src.evals.run                # benchmark: route / retrieval precision / faithfulness (Qwen judge)
PITWALL_EVALS=1 pytest tests/evals     # same, as regression floors (calls the LLM)
```

## Environment variables

- `LLM_PROVIDER` — `groq` (default) | `anthropic`
- `LLM_MODEL` — model id for all roles; `LLM_MODEL_<ROLE>` (ROUTER/FETCHER/ANALYST/JUDGE) overrides one role
- `GROQ_API_KEY` — required for the default provider
- `ANTHROPIC_API_KEY` — **not configured yet, on purpose** (avoiding accidental charges). Never
  default to, fall back to, or silently switch to Anthropic; it must be an explicit opt-in via
  `LLM_PROVIDER=anthropic`, and the factory should raise a clear error if the key is missing.
