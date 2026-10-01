"""HTTP API for the chat UI (pitwall-ui). Streams graph progress as Server-Sent Events.

Usage: uvicorn src.api:app --reload --port 8000

POST /api/chat {"thread_id": str | null, "message": str} -> text/event-stream:
  event: thread  {"thread_id"}                      first; send it back for follow-ups
  event: step    {"node", "detail"}                 one per graph node as it completes
  event: answer  {"answer", "mode", "race", "regs", "citations", "tool_calls", "steps"}
  event: error   {"message"}
"""

import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from src.agents.graph import build_graph, memory_checkpointer
from src.agents.live_chat import answer, prediction_event
from src.livetiming.archive import ArchiveSession, list_sessions
from src.livetiming.monitor import replay
from src.livetiming.session import SessionRegistry
from src.llm.transcribe import RadioTranscriber
from src.rag.index import RegulationIndex
from src.seasons import out_of_scope_message, supported_seasons
from src.sim.inrace import NotEnoughData
from src.sim.prerace import grid_prediction_event
from src.tools.openf1 import HttpOpenF1Client

log = logging.getLogger(__name__)
UI_ORIGINS = os.getenv("PITWALL_UI_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(
    ","
)


class ChatRequest(BaseModel):
    thread_id: str | None = None
    message: str = Field(min_length=1, max_length=2000)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _step_detail(node: str, update: dict[str, Any], seen_steps: int) -> str:
    steps = update.get("evaluation_steps") or []
    if len(steps) > seen_steps:
        return steps[-1]
    if node == "tools":
        # The tools node appends ToolMessages to the fetch transcript; count this batch.
        batch = 0
        for message in reversed(update.get("fetch_messages", [])):
            if not isinstance(message, ToolMessage):
                break
            batch += 1
        return f"ran {batch} tool call{'s' if batch != 1 else ''}"
    return node


def _answer(values: dict[str, Any]) -> dict[str, Any]:
    route = values.get("route")
    mode = route.mode if route else None
    return {
        "answer": values["messages"][-1].content,
        "mode": mode,
        "race": values.get("race_context") if mode == "race" and not values.get("error") else None,
        "regs": values.get("reg_context"),
        "citations": values.get("retrieved_rules_text") or [],
        "tool_calls": [
            {"tool": r["tool"], "args": r["args"]}
            for r in values.get("fetched_telemetry_json") or []
        ],
        "steps": values.get("evaluation_steps") or [],
    }


def _default_graph():
    # Embedded Qdrant allows one client per storage folder, so the index is opened once.
    index = RegulationIndex()
    graph = build_graph(
        HttpOpenF1Client(),
        index,
        checkpointer=memory_checkpointer(),
        seasons=supported_seasons(),
    )
    return graph, index.close


def _default_transcriber() -> RadioTranscriber | None:
    return RadioTranscriber() if RadioTranscriber.available() else None


def create_app(
    make_graph: Callable[[], tuple[Any, Callable[[], None]]] = _default_graph,
    make_transcriber: Callable[[], Any] = _default_transcriber,
    chat_model: Any = None,
) -> FastAPI:
    """make_graph returns (compiled graph, cleanup). Tests inject a graph with fake models and
    no transcriber."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.graph, cleanup = make_graph()
        app.state.transcriber = make_transcriber()
        app.state.sessions = SessionRegistry()
        app.state.chat_model = chat_model  # None -> the analyst model from the factory
        if os.getenv("PITWALL_LIVE_AUTOSTART", "").strip() in ("1", "true", "yes"):
            _start_live(app)  # race weekends: follow the live feed from server start
        yield
        await app.state.sessions.stop_live()
        cleanup()
        if app.state.transcriber:
            await app.state.transcriber.aclose()

    app = FastAPI(title="Pit Wall AI", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=UI_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"]
    )
    app.get("/api/health")(health)
    app.post("/api/chat")(chat)
    app.get("/api/seasons")(seasons)
    app.get("/api/live/sessions")(live_sessions)
    app.get("/api/live/replay")(live_replay)
    app.post("/api/live/ask")(live_ask)
    app.post("/api/live/start")(live_start)
    app.post("/api/live/stop")(live_stop)
    app.get("/api/live/current")(live_current)
    app.get("/api/live/stream")(live_stream)
    return app


def health() -> dict[str, str]:
    return {"status": "ok"}


def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    graph = request.app.state.graph
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    def events() -> Iterator[str]:
        yield _sse("thread", {"thread_id": thread_id})
        seen_steps = 0
        try:
            for chunk in graph.stream(
                {"messages": [HumanMessage(req.message)]}, config, stream_mode="updates"
            ):
                for node, update in chunk.items():
                    update = update or {}
                    detail = _step_detail(node, update, seen_steps)
                    if "evaluation_steps" in update:
                        seen_steps = len(update["evaluation_steps"])
                    yield _sse("step", {"node": node, "detail": detail})
            yield _sse("answer", _answer(graph.get_state(config).values))
        except Exception as e:
            log.exception("chat failed")
            yield _sse("error", {"message": f"{type(e).__name__}: {e}"})

    # Sync generator: Starlette runs it in a threadpool, so graph.stream doesn't block the loop.
    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


def seasons() -> list[int]:
    """Seasons available to users (current and previous), newest first."""
    return supported_seasons()


def live_sessions(year: int) -> list[dict[str, str]]:
    """Races in F1's live-timing archive for `year` (for the replay picker)."""
    if year not in supported_seasons():
        raise HTTPException(404, out_of_scope_message(year))
    return list_sessions(year)


def live_replay(
    request: Request,
    path: str,
    speed: float = Query(20.0, gt=0, le=1000),
    from_lap: int | None = Query(None, ge=1),
    transcribe: bool = True,
) -> StreamingResponse:
    """Replay an archived race as if live (SSE, GET so EventSource works). Event names are the
    monitor's event types: session, snapshot, track_status, race_control, radio,
    radio_transcript, pit_calls, end."""
    transcriber = request.app.state.transcriber if transcribe else None
    session = request.app.state.sessions.create("replay", path)

    async def events() -> AsyncIterator[str]:
        yield _sse("live", {"type": "live", "live_id": session.live_id, "source": "replay"})
        try:
            async for event in replay(
                ArchiveSession(path),
                speed=speed,
                from_lap=from_lap,
                transcriber=transcriber,
                monitor=session.monitor,
                on_neutralisation=prediction_event,
                on_grid=grid_prediction_event,
                alerts=True,
            ):
                yield _sse(event["type"], event)
        except Exception as e:
            log.exception("replay failed")
            yield _sse("error", {"message": f"{type(e).__name__}: {e}"})
        finally:
            # Replays feed the session only while their stream is open (they're a test source);
            # the race state stays queryable afterwards. The live feed will run as a server task.
            session.finished = time.time()

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


def _start_live(app: FastAPI):
    return app.state.sessions.start_live(
        transcriber=app.state.transcriber,
        on_neutralisation=prediction_event,
        on_grid=grid_prediction_event,
        alerts=True,
    )


def _describe(session) -> dict[str, Any]:
    snap = session.monitor.snapshot()
    return {
        "live_id": session.live_id,
        "source": session.source,
        "connected": session.connected,
        "recording": session.recording,
        "meeting": snap.meeting,
        "session": snap.session,
        "lap": snap.current_lap,
        "total_laps": snap.total_laps,
        "status": snap.track_status,
    }


async def live_start(request: Request) -> dict[str, Any]:
    """Follow the live F1 timing feed (a server task; idempotent). Local tool: no auth."""
    return _describe(_start_live(request.app))


async def live_stop(request: Request) -> dict[str, str]:
    await request.app.state.sessions.stop_live()
    return {"status": "stopped"}


def live_current(request: Request) -> dict[str, Any]:
    """The live-feed session, if one is running (or finished recently)."""
    session = request.app.state.sessions.current_live
    if session is None:
        raise HTTPException(404, "No live session. POST /api/live/start to follow the live feed.")
    return _describe(session)


def live_stream(request: Request, live_id: str) -> StreamingResponse:
    """Subscribe to a live session (SSE): `live`, a catch-up of the current state, then events."""
    session = request.app.state.sessions.get(live_id)
    if session is None:
        raise HTTPException(404, "That live session isn't running any more.")

    async def events() -> AsyncIterator[str]:
        async for event in session.subscribe():
            yield _sse(event["type"], event)

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


class AskRequest(BaseModel):
    live_id: str
    message: str = Field(min_length=1, max_length=1000)


def live_ask(req: AskRequest, request: Request) -> dict[str, Any]:
    """Answer a question about a live session's race as it stands right now."""
    session = request.app.state.sessions.get(req.live_id)
    if session is None:
        raise HTTPException(404, "That live session isn't running any more.")
    try:
        return answer(session.monitor, req.message, model=request.app.state.chat_model)
    except NotEnoughData as e:
        raise HTTPException(409, f"Not enough race data yet: {e}") from e


app = create_app()
