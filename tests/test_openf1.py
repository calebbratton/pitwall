from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from src.tools.openf1 import HttpOpenF1Client, MockOpenF1Client, OpenF1Error

MONACO_2024 = Path(__file__).parent / "fixtures/openf1/2024_monaco"
SESSION_KEY = 9523


@pytest.fixture
def client() -> MockOpenF1Client:
    return MockOpenF1Client(MONACO_2024)


@pytest.mark.parametrize("place", ["Monaco", "monte carlo", "MONACO"])
def test_get_session_by_country_location_or_circuit(client, place):
    session = client.get_session(2024, place)
    assert session.session_key == SESSION_KEY
    assert session.circuit_short_name == "Monte Carlo"


def test_get_session_ignores_accents(client):
    assert client.get_session(2024, "Montreal").location == "Montréal"


def test_get_session_ambiguous_country_raises_with_options(client):
    with pytest.raises(OpenF1Error, match="Miami.*Austin.*Las Vegas"):
        client.get_session(2024, "United States")
    assert client.get_session(2024, "Las Vegas").country_name == "United States"


def test_get_session_missing_raises(client):
    with pytest.raises(OpenF1Error, match="Known locations"):
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


NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)
STINTS = [{"driver_number": 4, "stint_number": 1, "compound": "HARD", "session_key": SESSION_KEY}]


def _http_client(handler, tmp_path: Path | None = None, now: datetime = NOW) -> HttpOpenF1Client:
    c = HttpOpenF1Client(cache_dir=tmp_path, now=lambda: now)
    c._http = httpx.Client(base_url="https://test", transport=httpx.MockTransport(handler))
    return c


def _api(session_end: datetime, calls: list):
    """Fake OpenF1: a session ending at `session_end`, and some stints for it."""

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/sessions":
            return httpx.Response(
                200, json=[{"session_key": SESSION_KEY, "date_end": session_end.isoformat()}]
            )
        return httpx.Response(200, json=STINTS)

    return handler


def test_http_settled_session_is_cached_forever(tmp_path):
    calls = []
    c = _http_client(_api(NOW - timedelta(days=3), calls), tmp_path)
    c.get_stints(SESSION_KEY)
    later = _http_client(_api(NOW, calls), tmp_path, now=NOW + timedelta(days=365))
    assert later.get_stints(SESSION_KEY)[0].compound == "HARD"
    assert calls.count("/stints") == 1


def test_http_recent_session_is_refetched(tmp_path):
    calls = []
    c = _http_client(_api(NOW - timedelta(hours=1), calls), tmp_path)
    c.get_stints(SESSION_KEY)
    c.get_stints(SESSION_KEY)
    assert calls.count("/stints") == 2


def test_http_calendar_expires(tmp_path):
    calls = []
    handler = lambda request: calls.append(1) or httpx.Response(200, json=[{"year": 2026}])
    _http_client(handler, tmp_path)._fetch("sessions", {"year": 2026})
    _http_client(handler, tmp_path, now=NOW + timedelta(hours=1))._fetch("sessions", {"year": 2026})
    assert len(calls) == 1
    _http_client(handler, tmp_path, now=NOW + timedelta(days=1))._fetch("sessions", {"year": 2026})
    assert len(calls) == 2


def test_http_404_is_empty_and_not_cached(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(404, json={"detail": "No results found."})

    c = _http_client(handler, tmp_path)
    assert c.get_stints(SESSION_KEY, driver_number=999) == []
    assert c.get_stints(SESSION_KEY, driver_number=999) == []
    assert len(calls) == 2


def test_http_live_data_refusal_explains_paid_window():
    c = _http_client(lambda request: httpx.Response(401))
    with pytest.raises(OpenF1Error, match="30 minutes after"):
        c.get_drivers(SESSION_KEY)


def test_http_retries_on_429(monkeypatch):
    monkeypatch.setattr("src.tools.openf1.time.sleep", lambda _: None)
    responses = iter([httpx.Response(429), httpx.Response(200, json=[])])
    c = _http_client(lambda request: next(responses))
    assert c.get_drivers(SESSION_KEY) == []


def test_http_server_error_raises():
    c = _http_client(lambda request: httpx.Response(500))
    with pytest.raises(OpenF1Error):
        c.get_drivers(SESSION_KEY)


def test_get_session_accepts_short_names(client):
    assert client.get_session(2024, "Spa").location == "Spa-Francorchamps"
    assert client.get_session(2024, "Yas").location == "Yas Island"
    with pytest.raises(OpenF1Error, match="several"):
        client.get_session(2024, "Mon")  # Monaco, Montréal, Monza
