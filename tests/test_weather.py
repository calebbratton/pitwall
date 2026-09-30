from datetime import UTC, datetime

import httpx

from src.weather.forecast import coordinates, race_rain


def _client(requests: list) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if "geocoding" in request.url.host:
            return httpx.Response(200, json={"results": [{"latitude": 3.1, "longitude": 101.7}]})
        hours = [f"2026-10-04T{h:02d}:00" for h in range(24)]
        prob = [10] * 24
        prob[8] = 70
        return httpx.Response(
            200,
            json={
                "hourly": {
                    "time": hours,
                    "precipitation_probability": prob,
                    "precipitation": [0.2] * 24,
                }
            },
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_known_circuit_uses_reference_coords_accent_insensitive():
    requests: list = []
    assert coordinates("São Paulo", _client(requests))[2] == "circuit"
    assert coordinates("Montréal", _client(requests))[2] == "circuit"
    assert requests == []


def test_race_window_takes_peak_probability():
    requests: list = []
    o = race_rain("Kuala Lumpur", datetime(2026, 10, 4, 7, 0, tzinfo=UTC), client=_client(requests))
    assert o.coords_source == "geocoded"
    assert [t for t, _, _ in o.hourly] == [
        "2026-10-04T07:00",
        "2026-10-04T08:00",
        "2026-10-04T09:00",
    ]
    assert o.p_rain == 0.7 and o.mm == 0.6
