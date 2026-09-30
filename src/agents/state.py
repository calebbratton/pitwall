"""Graph state and the structured outputs the LLM nodes produce."""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class RaceContext(TypedDict):
    year: int
    place: str  # as resolved by OpenF1: "Monaco (Monaco, 2024-05-26)"
    session_key: int
    race_date: str  # ISO date


class RegContext(TypedDict):
    season: int
    issue: int


class ToolResult(TypedDict):
    tool: str
    args: dict[str, Any]
    result: str


class RetrievedRule(TypedDict):
    article: str
    citation: str
    text: str


class PitWallState(TypedDict, total=False):
    # Persist across chat turns (checkpointed per thread).
    messages: Annotated[list[AnyMessage], add_messages]
    race_context: RaceContext | None

    # Per-question fields: overwritten (no reducer) by the router at the start of each turn.
    current_query: str
    route: "RouteDecision"
    reg_context: RegContext | None
    fetch_messages: list[AnyMessage]  # the fetch agent's private tool-calling transcript
    fetch_rounds: int
    fetched_telemetry_json: list[ToolResult]
    retrieved_rules_text: list[RetrievedRule]
    analysis: "Analysis"
    evaluation_steps: list[str]
    error: str | None


class RouteDecision(BaseModel):
    """How to handle the user's question."""

    mode: Literal["rules", "race"] = Field(
        description='"rules" for regulation-only questions; "race" when the answer needs data '
        "from a specific race (lap times, tyres, flags, strategy)."
    )
    year: int | None = Field(
        None, description="Season, if stated or implied by the conversation. Null if unknown."
    )
    place: str | None = Field(
        None,
        description="Race location as a country, city or circuit (e.g. Monaco, Madrid, Silverstone)."
        ' Use "latest" for the last / most recent race. Null for rules questions that don\'t name'
        " a race, or to keep the previous race.",
    )
    focus: str = Field(
        description="One sentence: what to analyse, naming drivers/teams/laps if mentioned."
    )
    regulation_queries: list[str] = Field(
        description="2-3 search queries phrased in FIA Sporting Regulations vocabulary, e.g. "
        '"race suspension work permitted on cars", "use of two dry-weather tyre specifications".',
        min_length=1,
        max_length=3,
    )


class Finding(BaseModel):
    claim: str = Field(description="One factual or analytical statement.")
    evidence: list[str] = Field(
        default_factory=list,
        description="Exact numbers copied from telemetry tool results that support the claim.",
    )
    articles: list[str] = Field(
        default_factory=list,
        description='Article numbers from the provided regulations only, e.g. "B6.3.6", "30.5".',
    )


class Analysis(BaseModel):
    verdict: str = Field(description="Direct answer to the question in 1-3 sentences.")
    findings: list[Finding] = Field(description="Supporting findings, most important first.")
    caveats: list[str] = Field(
        default_factory=list, description="Missing data or assumptions that limit the verdict."
    )
