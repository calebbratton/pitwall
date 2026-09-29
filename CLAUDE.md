# Pit Wall AI

Post-race strategy review engine for Formula 1. It takes a natural-language strategy question
(e.g. "Could McLaren have undercut on lap 30 at Monaco 2024?"), pulls race telemetry from the
OpenF1 API, retrieves the relevant FIA Sporting Regulation clauses via hybrid RAG, and returns a
grounded strategic verdict.

This is a hobby/portfolio project run by one person. **Optimize for $0 running cost**. LLM inference is
hosted (the dev machine is too weak for local models); everything else runs locally. Don't add paid services, hosted infra, or cloud dependencies unless asked.

## Stack

| Concern         | Choice                                                                 |
|-----------------|------------------------------------------------------------------------|
| Language        | Python 3.11+ (managed with `uv`; system Python is 3.9, don't use it)   |
| Orchestration   | LangGraph (stateful graph; Route → Fetch → Retrieve → Analyze → Synthesize) |
| LLM             | Provider-agnostic via `src/llm/` factory. Default: **Groq** (free tier) with `qwen/qwen3.8-27b` (a Groq *preview* model — may be pulled at short notice; override via `LLM_MODEL`). Optional: Anthropic. No local LLMs — dev machine can't run them |
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
pytest tests --ignore=tests/evals      # fast, offline
pytest tests/evals                     # benchmark suite, calls the LLM
```

## Environment variables

- `LLM_PROVIDER` — `groq` (default) | `anthropic`
- `LLM_MODEL` — model id for the chosen provider
- `GROQ_API_KEY` — required for the default provider
- `ANTHROPIC_API_KEY` — **not configured yet, on purpose** (avoiding accidental charges). Never
  default to, fall back to, or silently switch to Anthropic; it must be an explicit opt-in via
  `LLM_PROVIDER=anthropic`, and the factory should raise a clear error if the key is missing.
