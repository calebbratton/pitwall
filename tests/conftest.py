"""Test-wide guardrails."""

from urllib.parse import urlparse

import pytest
import websockets


@pytest.fixture(autouse=True)
def no_external_websockets(monkeypatch):
    """Tests must never reach the real F1 feed (a live feed never ends, so a leak shows up as a
    hang — it happened once). Only local fake servers are allowed."""
    real_connect = websockets.connect

    def guarded(uri, *args, **kwargs):
        host = urlparse(uri).hostname
        if host not in ("127.0.0.1", "localhost"):
            raise RuntimeError(f"test tried to open a websocket to {host}")
        return real_connect(uri, *args, **kwargs)

    monkeypatch.setattr("src.livetiming.client.websockets.connect", guarded)
