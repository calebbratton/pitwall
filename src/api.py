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
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from src.agents.graph import build_graph, memory_checkpointer
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


def create_app(
    make_graph: Callable[[], tuple[Any, Callable[[], None]]] = _default_graph,
) -> FastAPI:
    """make_graph returns (compiled graph, cleanup). Tests inject a graph with fake models."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.graph, cleanup = make_graph()
        yield
        cleanup()

    app = FastAPI(title="Pit Wall AI", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=UI_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"]
    )
    app.get("/api/health")(health)
    app.post("/api/chat")(chat)
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


app = create_app()
