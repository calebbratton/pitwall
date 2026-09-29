from pathlib import Path

import httpx
import pytest

from src.tools.openf1 import HttpOpenF1Client, MockOpenF1Client, OpenF1Error

MONACO_2024 = Path(__file__).parent / "fixtures/openf1/2024_monaco"
SESSION_KEY = 9523


@pytest.fixture
def client() -> MockOpenF1Client:
    return MockOpenF1Client(MONACO_2024)


def test_get_session(client):
    session = client.get_session(2024, "Monaco")
    assert session.session_key == SESSION_KEY
    assert session.circuit_short_name == "Monte Carlo"


def test_get_session_missing_raises(client):
    with pytest.raises(OpenF1Error):
        client.get_session(2024, "Atlantis")


def test_get_drivers_by_team(client):
    drivers = client.get_drivers(SESSION_KEY, team_name="McLaren")
    assert {d.name_acronym for d in drivers} == {"NOR", "PIA"}


def test_get_stints_sorted_for_driver(client):
    stints = client.get_stints(SESSION_KEY, driver_number=4)
    assert [s.compound for s in stints] == ["MEDIUM", "HARD"]
    assert stints[-1].lap_end == 78


def test_get_laps_range(client):
    laps = client.get_laps(SESSION_KEY, driver_number=4, lap_start=28, lap_end=32)
    assert [lap.lap_number for lap in laps] == [28, 29, 30, 31, 32]
    assert next(lap for lap in laps if lap.lap_number == 30).lap_duration == 78.403


def test_race_control_red_flag_on_lap_1(client):
    msgs = client.get_race_control(SESSION_KEY, category="Flag", lap_end=1)
    assert any(m.flag == "RED" for m in msgs)


def _http_client(handler, tmp_path: Path | None = None) -> HttpOpenF1Client:
    c = HttpOpenF1Client(cache_dir=tmp_path)
    c._http = httpx.Client(base_url="https://test", transport=httpx.MockTransport(handler))
    return c


def test_http_404_is_empty_and_cached(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(404, json={"detail": "No results found."})

    c = _http_client(handler, tmp_path)
    assert c.get_stints(SESSION_KEY, driver_number=999) == []
    assert c.get_stints(SESSION_KEY, driver_number=999) == []
    assert len(calls) == 1


def test_http_retries_on_429(monkeypatch):
    monkeypatch.setattr("src.tools.openf1.time.sleep", lambda _: None)
    responses = iter([httpx.Response(429), httpx.Response(200, json=[])])
    c = _http_client(lambda request: next(responses))
    assert c.get_drivers(SESSION_KEY) == []


def test_http_server_error_raises():
    c = _http_client(lambda request: httpx.Response(500))
    with pytest.raises(OpenF1Error):
        c.get_drivers(SESSION_KEY)
