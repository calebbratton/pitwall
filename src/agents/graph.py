"""Pit Wall AI state machine: Route -> Resolve -> Fetch <-> Tools -> Retrieve -> Analyze -> Synthesize.

    START -> route -> resolve --(race)--> fetch <-> tools
                        |                   |
                        |--(rules)--------> retrieve -> analyze -> synthesize -> END
                        '--(error)--------------------------------> synthesize

Dependencies (models, OpenF1 client, regulation index) are injected into `build_graph` so tests
can swap in fakes.
"""

import json
from datetime import UTC, date, datetime
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from src.agents import prompts
from src.agents.state import (
    Analysis,
    PitWallState,
    RetrievedRule,
    RouteDecision,
    ToolResult,
)
from src.llm.factory import Role, get_chat_model, with_schema
from src.rag.glossary import expand_query
from src.rag.index import RegulationIndex
from src.rag.sources import source_for_race
from src.tools.openf1 import OpenF1Client, OpenF1Error
from src.tools.telemetry import build_telemetry_tools, key_race_events, race_summary

MAX_FETCH_ROUNDS = 4
RULES_PER_QUERY = 4
MAX_RULES = 5
RRF_K = 60  # standard reciprocal-rank-fusion constant
# Groq free tier caps input at ~7-8K tokens/minute per model (~4 chars/token), so every prompt
# that grows with data has a character budget.
MAX_TOOL_RESULT_CHARS = 2500
OLD_ROUND_RESULT_CHARS = 300  # earlier rounds' results, when resent to the fetcher
MAX_ANALYST_TELEMETRY_CHARS = 12000
HISTORY_MESSAGES = 6
LATEST_PLACES = {"latest", "last", "last race", "latest race", "most recent", "most recent race"}


def memory_checkpointer() -> InMemorySaver:
    """In-process chat memory that is allowed to round-trip our Pydantic state types."""
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("src.agents.state", name) for name in ("RouteDecision", "Analysis", "Finding")
        ]
    )
    return InMemorySaver(serde=serde)


def _clip(text: str, limit: int) -> str:
    return (
        text if len(text) <= limit else f"{text[:limit]}... [truncated {len(text) - limit} chars]"
    )


def _compact_old_rounds(messages: list) -> list:
    """The fetcher already saw earlier rounds' tool results; resend them shortened so the
    transcript doesn't outgrow the per-minute token limit. The latest round stays whole."""
    last_ai = max((i for i, m in enumerate(messages) if isinstance(m, AIMessage)), default=-1)
    return [
        ToolMessage(_clip(m.content, OLD_ROUND_RESULT_CHARS), tool_call_id=m.tool_call_id)
        if isinstance(m, ToolMessage) and i < last_ai
        else m
        for i, m in enumerate(messages)
    ]


def _history(state: PitWallState) -> str:
    msgs = state.get("messages", [])[-HISTORY_MESSAGES:]
    return "\n".join(f"{m.type}: {m.content}" for m in msgs) or "(none)"


def _race_label(state: PitWallState) -> str:
    ctx = state.get("race_context")
    return ctx["place"] if ctx else "none"


def build_graph(
    openf1: OpenF1Client,
    index: RegulationIndex,
    models: dict[Role, BaseChatModel] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    today: date | None = None,
):
    models = models or {}

    def model(role: Role) -> BaseChatModel:
        if role not in models:
            models[role] = get_chat_model(role)
        return models[role]

    tools_cache: dict[int, list] = {}

    def tools_for(session_key: int) -> list:
        if session_key not in tools_cache:
            session = openf1.get_session_by_key(session_key)
            tools_cache[session_key] = build_telemetry_tools(openf1, session)
        return tools_cache[session_key]

    def _now() -> datetime:
        return (
            datetime.combine(today, datetime.max.time(), tzinfo=UTC) if today else datetime.now(UTC)
        )

    def _today() -> date:
        return _now().date()

    # --- nodes ---------------------------------------------------------------------------

    def route(state: PitWallState) -> dict[str, Any]:
        query = state["messages"][-1].content
        prompt = prompts.ROUTER.format(
            today=_today().isoformat(),
            current_year=_today().year,
            race_context=_race_label(state),
            history=_history(state),
        )
        decision: RouteDecision = with_schema(model("router"), RouteDecision).invoke(
            [SystemMessage(prompt), HumanMessage(query)]
        )
        return {
            "current_query": query,
            "route": decision,
            "reg_context": None,
            "fetch_messages": [],
            "fetch_rounds": 0,
            "fetched_telemetry_json": [],
            "retrieved_rules_text": [],
            "error": None,
            "evaluation_steps": [
                (
                    f"route: mode={decision.mode} year={decision.year} place={decision.place} "
                    f"focus={decision.focus!r} queries={decision.regulation_queries}"
                )
            ],
        }

    def resolve(state: PitWallState) -> dict[str, Any]:
        """Deterministic: turn the router's year/place into an OpenF1 session and pick the
        regulations issue in force. No LLM, so no hallucinated session keys."""
        decision = state["route"]
        steps = list(state["evaluation_steps"])
        race_context = state.get("race_context")

        if decision.mode == "race" and decision.place:
            year = decision.year or _today().year
            try:
                if decision.place.strip().casefold() in LATEST_PLACES:
                    session = openf1.get_latest_session(_now())
                    year = session.year
                else:
                    session = openf1.get_session(year, decision.place)
            except OpenF1Error as e:
                return {"error": str(e), "evaluation_steps": [*steps, f"resolve: {e}"]}
            if datetime.fromisoformat(session.date_start) > _now():
                msg = f"{session.label} hasn't happened yet, so there is no race data."
                return {"error": msg, "evaluation_steps": [*steps, f"resolve: {msg}"]}
            race_context = {
                "year": year,
                "place": session.label,
                "session_key": session.session_key,
                "race_date": session.date_start[:10],
            }

        if decision.mode == "race" and not race_context:
            msg = "Which race? Name the season and location, e.g. 'Monaco 2024'."
            return {"error": msg, "evaluation_steps": [*steps, "resolve: no race"]}

        if decision.mode == "race":
            ref_date = date.fromisoformat(race_context["race_date"])
        elif decision.year and decision.year != _today().year:
            ref_date = date(decision.year, 12, 31)
        else:
            ref_date = _today()
        season = ref_date.year if decision.mode == "race" else (decision.year or _today().year)
        try:
            source = source_for_race(season, ref_date)
        except ValueError as e:
            return {"error": str(e), "evaluation_steps": [*steps, f"resolve: {e}"]}

        telemetry = []
        if decision.mode == "race":
            session_key = race_context["session_key"]
            events = key_race_events(openf1, session_key)
            summary = race_summary(openf1, openf1.get_session_by_key(session_key))
            telemetry = [
                ToolResult(tool="key_race_events", args={}, result=events),
                ToolResult(tool="race_summary", args={}, result=summary),
            ]
        return {
            "race_context": race_context,
            "reg_context": {"season": source.season, "issue": source.issue},
            "fetched_telemetry_json": telemetry,
            "evaluation_steps": [
                *steps,
                (
                    f"resolve: race={race_context['place'] if decision.mode == 'race' else '-'} "
                    f"regs={source.season} issue {source.issue}"
                ),
            ],
        }

    def fetch(state: PitWallState) -> dict[str, Any]:
        tools = tools_for(state["race_context"]["session_key"])
        messages = state["fetch_messages"] or [
            SystemMessage(
                prompts.FETCH.format(
                    focus=state["route"].focus,
                    race=_race_label(state),
                    max_rounds=MAX_FETCH_ROUNDS,
                    key_events=state["fetched_telemetry_json"][0]["result"],
                    race_summary=state["fetched_telemetry_json"][1]["result"],
                )
            ),
            HumanMessage(state["current_query"]),
        ]
        reply = model("fetcher").bind_tools(tools).invoke(_compact_old_rounds(messages))
        calls = [f"{c['name']}({c['args']})" for c in reply.tool_calls]
        return {
            "fetch_messages": [*messages, reply],
            "fetch_rounds": state["fetch_rounds"] + 1,
            "evaluation_steps": [
                *state["evaluation_steps"],
                f"fetch round {state['fetch_rounds'] + 1}: {calls or 'done'}",
            ],
        }

    def run_tools(state: PitWallState) -> dict[str, Any]:
        tools = {t.name: t for t in tools_for(state["race_context"]["session_key"])}
        reply: AIMessage = state["fetch_messages"][-1]
        tool_messages, results = [], []
        for call in reply.tool_calls:
            if call["name"] in tools:
                try:
                    output = tools[call["name"]].invoke(call["args"])
                except Exception as e:  # noqa: BLE001 — any tool failure goes back to the model to retry
                    output = f"ERROR: {type(e).__name__}: {e}"
            else:
                output = f"ERROR: unknown tool {call['name']!r}"
            output = _clip(output, MAX_TOOL_RESULT_CHARS)
            tool_messages.append(ToolMessage(output, tool_call_id=call["id"]))
            results.append(ToolResult(tool=call["name"], args=call["args"], result=output))
        return {
            "fetch_messages": [*state["fetch_messages"], *tool_messages],
            "fetched_telemetry_json": [*state["fetched_telemetry_json"], *results],
        }

    def retrieve(state: PitWallState) -> dict[str, Any]:
        """Hybrid search per query (router queries, the question, and glossary expansions of
        both), merged with reciprocal rank fusion so clauses that several phrasings agree on
        rank first."""
        reg = state["reg_context"]
        queries = [*state["route"].regulation_queries, state["current_query"]]
        queries += [e for q in list(queries) if (e := expand_query(q))]
        scores: dict[str, float] = {}
        rules_by_id: dict[str, RetrievedRule] = {}
        for query in queries:
            hits = index.search(query, reg["season"], reg["issue"], k=RULES_PER_QUERY)
            for rank, hit in enumerate(hits):
                c = hit.chunk
                scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1 / (RRF_K + rank)
                rules_by_id[c.chunk_id] = RetrievedRule(
                    article=c.article, citation=c.citation, text=c.text
                )
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:MAX_RULES]
        rules = [rules_by_id[chunk_id] for chunk_id in ranked]
        return {
            "retrieved_rules_text": rules,
            "evaluation_steps": [
                *state["evaluation_steps"],
                f"retrieve ({len(queries)} queries): {[r['article'] for r in rules]}",
            ],
        }

    def analyze(state: PitWallState) -> dict[str, Any]:
        reg = state["reg_context"]
        results = state["fetched_telemetry_json"]
        per_result = MAX_ANALYST_TELEMETRY_CHARS // max(len(results), 1)
        telemetry = [
            {"tool": r["tool"], "args": r["args"], "result": _clip(r["result"], per_result)}
            for r in results
        ]
        rules = "\n\n".join(f"[{r['article']}] {r['text']}" for r in state["retrieved_rules_text"])
        prompt = prompts.ANALYST.format(
            question=state["current_query"],
            race=_race_label(state) if state["route"].mode == "race" else "n/a (rules question)",
            telemetry=json.dumps(telemetry, separators=(",", ":")) if telemetry else "(none)",
            reg_source=f"{reg['season']} Sporting Regulations, Issue {reg['issue']}",
            rules=rules or "(none retrieved)",
        )
        analysis: Analysis = with_schema(model("analyst"), Analysis).invoke(
            [SystemMessage(prompt), HumanMessage(state["current_query"])]
        )
        return {
            "analysis": analysis,
            "evaluation_steps": [
                *state["evaluation_steps"],
                f"analyze: {len(analysis.findings)} findings",
            ],
        }

    def synthesize(state: PitWallState) -> dict[str, Any]:
        """Deterministic rendering (no LLM). Drops citations of articles that weren't retrieved,
        so the answer can only cite regulation text the model was actually shown."""
        if state.get("error"):
            return {"messages": [AIMessage(state["error"])]}

        analysis = state["analysis"]
        rules = {r["article"]: r for r in state["retrieved_rules_text"]}
        steps = list(state["evaluation_steps"])
        lines = [analysis.verdict, ""]
        cited: dict[str, str] = {}
        for f in analysis.findings:
            valid = [a for a in f.articles if a in rules]
            dropped = sorted(set(f.articles) - set(valid))
            if dropped:
                steps.append(f"synthesize: dropped unretrieved citations {dropped}")
            for a in valid:
                cited[a] = rules[a]["citation"]
            refs = f" [{', '.join(valid)}]" if valid else ""
            evidence = f" ({'; '.join(f.evidence)})" if f.evidence else ""
            lines.append(f"- {f.claim}{evidence}{refs}")
        if analysis.caveats:
            lines += ["", "Caveats:", *(f"- {c}" for c in analysis.caveats)]
        if cited:
            lines += ["", "Regulations:", *(f"- {c}" for c in cited.values())]
        return {"messages": [AIMessage("\n".join(lines))], "evaluation_steps": steps}

    # --- edges ---------------------------------------------------------------------------

    def after_resolve(state: PitWallState) -> str:
        if state.get("error"):
            return "synthesize"
        return "fetch" if state["route"].mode == "race" else "retrieve"

    def after_fetch(state: PitWallState) -> str:
        last = state["fetch_messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "retrieve"

    def after_tools(state: PitWallState) -> str:
        return "fetch" if state["fetch_rounds"] < MAX_FETCH_ROUNDS else "retrieve"

    g = StateGraph(PitWallState)
    for name, fn in [
        ("route", route),
        ("resolve", resolve),
        ("fetch", fetch),
        ("tools", run_tools),
        ("retrieve", retrieve),
        ("analyze", analyze),
        ("synthesize", synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_edge("route", "resolve")
    g.add_conditional_edges("resolve", after_resolve, ["fetch", "retrieve", "synthesize"])
    g.add_conditional_edges("fetch", after_fetch, ["tools", "retrieve"])
    g.add_conditional_edges("tools", after_tools, ["fetch", "retrieve"])
    g.add_edge("retrieve", "analyze")
    g.add_edge("analyze", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile(checkpointer=checkpointer)
