"""MCP server tools on recorded OpenF1 fixtures (Monaco 2024): no network, no LLM."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.mcp_server import TOOLS, PitwallTools, build_server
from src.rag.chunking import RegChunk
from src.rag.index import RetrievedClause
from src.tools.openf1 import MockOpenF1Client

MONACO_2024 = Path(__file__).parent / "fixtures/openf1/2024_monaco"


@pytest.fixture
def tools():
    return PitwallTools(
        MockOpenF1Client(MONACO_2024),
        seasons=[2024],
        now=lambda: datetime(2024, 12, 31, tzinfo=UTC),
    )


def test_find_driver_by_first_name_number_and_acronym(tools):
    for name in ("Charles", "16", "lec", "Leclerc"):
        out = json.loads(tools.find_driver(2024, name, "Monaco"))
        assert [m["acronym"] for m in out["matches"]] == ["LEC"], name
    assert tools.find_driver(2024, "Kimi", "Monaco").startswith("ERROR: unknown driver")
    assert "LEC (Charles LECLERC)" in tools.find_driver(2024, "Kimi", "Monaco")


def test_race_window_future_and_ambiguity_are_errors_with_options(tools):
    assert "isn't available" in tools.race_results(2023, "Monaco")
    early = PitwallTools(
        MockOpenF1Client(MONACO_2024), seasons=[2024], now=lambda: datetime(2024, 5, 1, tzinfo=UTC)
    )
    assert "hasn't happened yet" in early.race_results(2024, "Monaco")
    assert tools.race_results(2024, "Atlantis").startswith("ERROR: No 2024 Race found")


def test_race_results_report_status_grid_and_unknown_gaps(tools):
    out = json.loads(tools.race_results(2024, "Monaco"))
    first, last = out["results"][0], out["results"][-1]
    assert (first["driver"], first["position"], first["status"]) == ("LEC", 1, "finished")
    assert first["grid"] == 1
    assert last["status"] == "DNF" and last["laps"] == 0
    # Not in this recording: reported as unknown, not as zero.
    assert first["gap_to_winner"] is None
    assert out["source"].startswith("OpenF1")


def test_pit_stops_flag_the_red_flag_lap(tools):
    out = json.loads(tools.pit_stops(2024, "Monaco", "LEC"))
    first = out["stops"][0]
    assert first["lap"] == 1 and first["pit_lane_s"] == 2358.0
    assert first["stationary_s"] is None


def test_positions_and_gaps_from_line_crossings(tools):
    pos = json.loads(tools.positions_by_lap(2024, "Monaco", 70, 72, ["LEC", "PIA", "NOR"]))
    lap70 = pos["positions"]["70"]
    assert lap70["LEC"] < lap70["PIA"] < lap70["NOR"]
    gaps = json.loads(tools.gap_between(2024, "Monaco", "PIA", "LEC", 70, 72))["gap_s"]
    assert all(g is not None and g > 0 for g in gaps.values())  # PIA behind LEC


def test_tyre_stints_carry_each_stints_own_wear(tools):
    out = json.loads(tools.tyre_stints(2024, "Monaco", "NOR"))
    assert out["stints"] and all("wear_s_per_lap" in s for s in out["stints"])


def test_regulations_and_stewards_search_cite_their_sources(tools):
    class Index:
        def search(self, query, season, issue, k=5):
            chunk = RegChunk(
                "id", "48.12", "PIT LANE", 1, "No car may be released...", season, issue
            )
            return [RetrievedClause(chunk, 1.0)]

    tools._make_regulations = Index
    out = json.loads(tools.search_regulations("unsafe release", 2024, "Monaco"))
    assert out["clauses"][0]["citation"].endswith("Article 48.12")
    assert out["source"].startswith("FIA 2024 Sporting Regulations")
    assert "aren't configured" in tools.stewards_decisions("impeding")


def test_server_lists_every_tool_and_answers_a_call(tools):
    server = build_server(tools)
    listed = {t.name for t in asyncio.run(server.list_tools())}
    assert listed == set(TOOLS)
    result = asyncio.run(
        server.call_tool("find_driver", {"year": 2024, "name": "Charles", "race": "Monaco"})
    )
    text = result.content[0].text
    assert '"LEC"' in text


def test_api_serves_mcp_at_slash_mcp(monkeypatch, tools):
    from fastapi.testclient import TestClient

    from src.api import create_app

    monkeypatch.setenv("PITWALL_MCP_HOSTS", "testserver")
    app = create_app(
        make_graph=lambda: (None, lambda: None),
        make_transcriber=lambda: None,
        make_mcp_tools=lambda: tools,
    )
    headers = {"Accept": "application/json, text/event-stream"}
    with TestClient(app) as http:
        init = http.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            },
        )
        assert init.status_code == 200, init.text
        listed = http.post(
            "/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        )
        assert listed.status_code == 200, listed.text
        assert "review_pit_stop" in listed.text
        assert http.get("/api/health").status_code == 200
