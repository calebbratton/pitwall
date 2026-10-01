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


@pytest.fixture(autouse=True)
def no_weather_forecasts(monkeypatch):
    """The live pipeline asks Open-Meteo for a rain forecast; tests never do."""

    def blocked(*args, **kwargs):
        raise RuntimeError("weather forecast blocked in tests")

    monkeypatch.setattr("src.livetiming.monitor.race_rain", blocked)


@pytest.fixture(autouse=True)
def no_real_prediction_log(monkeypatch, tmp_path):
    """The locked-in prediction log (predictions/) is a record; tests write to a temp folder."""
    monkeypatch.setattr("src.sim.prediction_log.PREDICTIONS", tmp_path / "predictions")


@pytest.fixture(autouse=True)
def no_alert_llm(monkeypatch):
    """Alert phrasing calls a real LLM; tests use the templates."""
    monkeypatch.setenv("PITWALL_ALERT_LLM", "0")


@pytest.fixture(autouse=True)
def no_live_models(monkeypatch):
    """The live chat's learned-model extras train on the local warehouse; tests skip them."""
    monkeypatch.setenv("PITWALL_LIVE_MODELS", "0")
