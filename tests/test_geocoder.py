"""Testes para src/search/geocoder.py — mocka requests e time."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch, call

import pytest

import src.search.geocoder as geocoder_module
from src.search.geocoder import geocode, geocode_com_fallback, geocode_detalhado


@pytest.fixture(autouse=True)
def reset_last_call():
    """Zera o contador de rate limit antes de cada teste."""
    geocoder_module._last_call = 0.0
    yield
    geocoder_module._last_call = 0.0


def _make_nominatim_response(lat="-23.46", lon="-46.53", address=None):
    resp = MagicMock()
    result = [{"lat": lat, "lon": lon}]
    if address is not None:
        result[0]["address"] = address
    resp.json.return_value = result
    return resp


def _make_empty_nominatim_response():
    resp = MagicMock()
    resp.json.return_value = []
    return resp


# ─── geocode ──────────────────────────────────────────────────────────────────

class TestGeocode:
    def test_returns_lat_lon_on_success(self):
        resp = _make_nominatim_response("-23.46", "-46.53")
        with patch("src.search.geocoder.requests.get", return_value=resp), \
             patch("src.search.geocoder.time.sleep"), \
             patch("src.search.geocoder.time.time", return_value=999.0):
            result = geocode("Guarulhos, São Paulo")
        assert result == (-23.46, -46.53)

    def test_returns_none_on_empty_response(self):
        resp = _make_empty_nominatim_response()
        with patch("src.search.geocoder.requests.get", return_value=resp), \
             patch("src.search.geocoder.time.sleep"), \
             patch("src.search.geocoder.time.time", return_value=999.0):
            result = geocode("Lugar Inexistente XYZ")
        assert result is None

    def test_returns_none_on_exception(self):
        with patch("src.search.geocoder.requests.get", side_effect=ConnectionError()), \
             patch("src.search.geocoder.time.sleep"), \
             patch("src.search.geocoder.time.time", return_value=999.0):
            result = geocode("qualquer coisa")
        assert result is None

    def test_respects_rate_limit_sleep(self):
        """Se último call foi há < 1.1s, deve dormir."""
        geocoder_module._last_call = 1000.0  # simula chamada recente
        resp = _make_nominatim_response()
        with patch("src.search.geocoder.requests.get", return_value=resp) as mock_get, \
             patch("src.search.geocoder.time.sleep") as mock_sleep, \
             patch("src.search.geocoder.time.time", side_effect=[1000.5, 1001.0]):
            geocode("Guarulhos")
        mock_sleep.assert_called_once()
        sleep_arg = mock_sleep.call_args[0][0]
        assert sleep_arg > 0

    def test_no_sleep_when_enough_time_elapsed(self):
        geocoder_module._last_call = 0.0
        resp = _make_nominatim_response()
        with patch("src.search.geocoder.requests.get", return_value=resp), \
             patch("src.search.geocoder.time.sleep") as mock_sleep, \
             patch("src.search.geocoder.time.time", return_value=1100.0):
            geocode("Guarulhos")
        mock_sleep.assert_not_called()

    def test_updates_last_call_after_request(self):
        resp = _make_nominatim_response()
        with patch("src.search.geocoder.requests.get", return_value=resp), \
             patch("src.search.geocoder.time.sleep"), \
             patch("src.search.geocoder.time.time", return_value=5000.0):
            geocode("Guarulhos")
        assert geocoder_module._last_call == 5000.0

    def test_float_conversion(self):
        resp = _make_nominatim_response("-23.550520", "-46.633308")
        with patch("src.search.geocoder.requests.get", return_value=resp), \
             patch("src.search.geocoder.time.sleep"), \
             patch("src.search.geocoder.time.time", return_value=999.0):
            result = geocode("São Paulo")
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


# ─── geocode_com_fallback ─────────────────────────────────────────────────────

class TestGeocodeComFallback:
    def test_returns_geocoded_when_successful(self):
        with patch("src.search.geocoder.geocode", return_value=(-23.46, -46.53)) as mock_geo:
            result = geocode_com_fallback("Guarulhos, São Paulo")
        assert result == (-23.46, -46.53)

    def test_falls_back_to_cidade_fallback(self):
        # First call fails (texto, Brasil), second succeeds (cidade_fallback, Brasil)
        with patch("src.search.geocoder.geocode", side_effect=[None, (-23.55, -46.63)]):
            result = geocode_com_fallback("Lugar Inexistente", "São Paulo, São Paulo")
        assert result == (-23.55, -46.63)

    def test_falls_back_to_hardcoded_coords(self):
        with patch("src.search.geocoder.geocode", return_value=None):
            result = geocode_com_fallback("Nada", "Nada Também")
        assert result == (-23.4628, -46.5333)

    def test_appends_brasil_to_query(self):
        calls = []
        def fake_geocode(texto):
            calls.append(texto)
            return (-23.46, -46.53)
        with patch("src.search.geocoder.geocode", side_effect=fake_geocode):
            geocode_com_fallback("Guarulhos")
        assert "Brasil" in calls[0]

    def test_default_fallback_city(self):
        called = []
        def fake_geocode(texto):
            called.append(texto)
            return None
        with patch("src.search.geocoder.geocode", side_effect=fake_geocode):
            geocode_com_fallback("Nada")
        assert any("Guarulhos" in c for c in called)


# ─── geocode_detalhado ────────────────────────────────────────────────────────

class TestGeocodeDetalhado:
    def _make_detailed_response(self, suburb="Vila Prudente", city="São Paulo",
                                 state="São Paulo", lat="-23.55", lon="-46.63"):
        resp = MagicMock()
        resp.json.return_value = [{
            "lat": lat,
            "lon": lon,
            "address": {
                "suburb": suburb,
                "city": city,
                "state": state,
            }
        }]
        return resp

    def test_returns_full_dict(self):
        resp = self._make_detailed_response()
        with patch("src.search.geocoder.requests.get", return_value=resp), \
             patch("src.search.geocoder.time.sleep"), \
             patch("src.search.geocoder.time.time", return_value=999.0):
            result = geocode_detalhado("Vila Prudente, São Paulo")
        assert result["lat"] == -23.55
        assert result["lon"] == -46.63
        assert result["cidade"] == "São Paulo"
        assert result["estado"] == "São Paulo"
        assert result["bairro"] == "Vila Prudente"

    def test_returns_empty_on_empty_response(self):
        resp = MagicMock()
        resp.json.return_value = []
        with patch("src.search.geocoder.requests.get", return_value=resp), \
             patch("src.search.geocoder.time.sleep"), \
             patch("src.search.geocoder.time.time", return_value=999.0):
            result = geocode_detalhado("Nenhum Lugar")
        assert result == {}

    def test_returns_empty_on_exception(self):
        with patch("src.search.geocoder.requests.get", side_effect=ConnectionError()), \
             patch("src.search.geocoder.time.sleep"), \
             patch("src.search.geocoder.time.time", return_value=999.0):
            result = geocode_detalhado("qualquer")
        assert result == {}

    def test_bairro_city_district_fallback(self):
        resp = MagicMock()
        resp.json.return_value = [{
            "lat": "-23.55", "lon": "-46.63",
            "address": {"city_district": "Mooca", "city": "São Paulo", "state": "SP"}
        }]
        with patch("src.search.geocoder.requests.get", return_value=resp), \
             patch("src.search.geocoder.time.sleep"), \
             patch("src.search.geocoder.time.time", return_value=999.0):
            result = geocode_detalhado("Mooca, São Paulo")
        assert result["bairro"] == "Mooca"

    def test_bairro_neighbourhood_fallback(self):
        resp = MagicMock()
        resp.json.return_value = [{
            "lat": "-23.55", "lon": "-46.63",
            "address": {"neighbourhood": "Bela Vista", "city": "São Paulo", "state": "SP"}
        }]
        with patch("src.search.geocoder.requests.get", return_value=resp), \
             patch("src.search.geocoder.time.sleep"), \
             patch("src.search.geocoder.time.time", return_value=999.0):
            result = geocode_detalhado("Bela Vista")
        assert result["bairro"] == "Bela Vista"

    def test_cidade_town_fallback(self):
        resp = MagicMock()
        resp.json.return_value = [{
            "lat": "-23.0", "lon": "-47.0",
            "address": {"town": "Campinas", "state": "São Paulo"}
        }]
        with patch("src.search.geocoder.requests.get", return_value=resp), \
             patch("src.search.geocoder.time.sleep"), \
             patch("src.search.geocoder.time.time", return_value=999.0):
            result = geocode_detalhado("Campinas")
        assert result["cidade"] == "Campinas"

    def test_cidade_municipality_fallback(self):
        resp = MagicMock()
        resp.json.return_value = [{
            "lat": "-23.0", "lon": "-47.0",
            "address": {"municipality": "Interior SP", "state": "SP"}
        }]
        with patch("src.search.geocoder.requests.get", return_value=resp), \
             patch("src.search.geocoder.time.sleep"), \
             patch("src.search.geocoder.time.time", return_value=999.0):
            result = geocode_detalhado("Interior")
        assert result["cidade"] == "Interior SP"

    def test_respects_rate_limit(self):
        geocoder_module._last_call = 1000.0
        resp = self._make_detailed_response()
        with patch("src.search.geocoder.requests.get", return_value=resp), \
             patch("src.search.geocoder.time.sleep") as mock_sleep, \
             patch("src.search.geocoder.time.time", side_effect=[1000.5, 1001.0]):
            geocode_detalhado("Guarulhos")
        mock_sleep.assert_called_once()

    def test_bairro_quarter_fallback(self):
        resp = MagicMock()
        resp.json.return_value = [{
            "lat": "-23.55", "lon": "-46.63",
            "address": {"quarter": "Jardim Paulistano", "city": "São Paulo", "state": "SP"}
        }]
        with patch("src.search.geocoder.requests.get", return_value=resp), \
             patch("src.search.geocoder.time.sleep"), \
             patch("src.search.geocoder.time.time", return_value=999.0):
            result = geocode_detalhado("Jardim Paulistano")
        assert result["bairro"] == "Jardim Paulistano"
