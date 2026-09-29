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
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from src.agents.graph import build_graph, memory_checkpointer
from src.livetiming.archive import ArchiveSession, list_sessions
from src.livetiming.monitor import replay
from src.llm.transcribe import RadioTranscriber
from src.rag.index import RegulationIndex
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
    graph = build_graph(HttpOpenF1Client(), index, checkpointer=memory_checkpointer())
    return graph, index.close


def _default_transcriber() -> RadioTranscriber | None:
    return RadioTranscriber() if RadioTranscriber.available() else None


def create_app(
    make_graph: Callable[[], tuple[Any, Callable[[], None]]] = _default_graph,
    make_transcriber: Callable[[], Any] = _default_transcriber,
) -> FastAPI:
    """make_graph returns (compiled graph, cleanup). Tests inject a graph with fake models and
    no transcriber."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.graph, cleanup = make_graph()
        app.state.transcriber = make_transcriber()
        yield
        cleanup()
        if app.state.transcriber:
            await app.state.transcriber.aclose()

    app = FastAPI(title="Pit Wall AI", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=UI_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"]
    )
    app.get("/api/health")(health)
    app.post("/api/chat")(chat)
    app.get("/api/live/sessions")(live_sessions)
    app.get("/api/live/replay")(live_replay)
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


def live_sessions(year: int) -> list[dict[str, str]]:
    """Races in F1's live-timing archive for `year` (for the replay picker)."""
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

    async def events() -> AsyncIterator[str]:
        try:
            async for event in replay(
                ArchiveSession(path), speed=speed, from_lap=from_lap, transcriber=transcriber
            ):
                yield _sse(event["type"], event)
        except Exception as e:
            log.exception("replay failed")
            yield _sse("error", {"message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


app = create_app()
