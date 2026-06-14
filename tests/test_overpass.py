"""Testes para src/search/overpass.py — mocka requests.post."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.search.overpass import buscar_pois_localizacoes, AMENITY_MAP, RAILWAY_MAP


def _mock_overpass_response(elements):
    resp = MagicMock()
    resp.json.return_value = {"elements": elements}
    return resp


LAT, LON = -23.46, -46.53
RAIO = 1000


class TestBuscarPoisLocalizacoes:
    def test_returns_dict_with_category_keys(self):
        resp = _mock_overpass_response([{"lat": -23.46, "lon": -46.53}])
        with patch("src.search.overpass.requests.post", return_value=resp):
            result = buscar_pois_localizacoes(LAT, LON, RAIO, ["supermercado"])
        assert "supermercado" in result

    def test_extracts_lat_lon_tuples(self):
        elements = [
            {"lat": -23.46, "lon": -46.53},
            {"lat": -23.47, "lon": -46.54},
        ]
        resp = _mock_overpass_response(elements)
        with patch("src.search.overpass.requests.post", return_value=resp):
            result = buscar_pois_localizacoes(LAT, LON, RAIO, ["escola"])
        assert len(result["escola"]) == 2
        assert result["escola"][0] == (-23.46, -46.53)

    def test_skips_elements_without_lat_lon(self):
        elements = [
            {"lat": -23.46, "lon": -46.53},
            {"id": 999},  # sem lat/lon
        ]
        resp = _mock_overpass_response(elements)
        with patch("src.search.overpass.requests.post", return_value=resp):
            result = buscar_pois_localizacoes(LAT, LON, RAIO, ["restaurante"])
        assert len(result["restaurante"]) == 1

    def test_empty_elements_returns_empty_list(self):
        resp = _mock_overpass_response([])
        with patch("src.search.overpass.requests.post", return_value=resp):
            result = buscar_pois_localizacoes(LAT, LON, RAIO, ["hospital"])
        assert result["hospital"] == []

    def test_exception_returns_empty_list(self):
        with patch("src.search.overpass.requests.post", side_effect=ConnectionError()):
            result = buscar_pois_localizacoes(LAT, LON, RAIO, ["metro"])
        assert result["metro"] == []

    def test_empty_categorias_returns_empty_dict(self):
        result = buscar_pois_localizacoes(LAT, LON, RAIO, [])
        assert result == {}

    def test_multiple_categories(self):
        resp = _mock_overpass_response([{"lat": -23.46, "lon": -46.53}])
        with patch("src.search.overpass.requests.post", return_value=resp):
            result = buscar_pois_localizacoes(LAT, LON, RAIO, ["escola", "supermercado"])
        assert "escola" in result
        assert "supermercado" in result

    def test_parque_uses_leisure_tag(self):
        resp = _mock_overpass_response([{"lat": -23.46, "lon": -46.53}])
        with patch("src.search.overpass.requests.post", return_value=resp) as mock_post:
            buscar_pois_localizacoes(LAT, LON, RAIO, ["parque"])
        query = mock_post.call_args[1]["data"]["data"]
        assert "leisure" in query
        assert "park" in query

    def test_metro_uses_railway_tag(self):
        resp = _mock_overpass_response([{"lat": -23.46, "lon": -46.53}])
        with patch("src.search.overpass.requests.post", return_value=resp) as mock_post:
            buscar_pois_localizacoes(LAT, LON, RAIO, ["metro"])
        query = mock_post.call_args[1]["data"]["data"]
        assert "railway" in query

    def test_metro_also_uses_amenity_tag(self):
        resp = _mock_overpass_response([])
        with patch("src.search.overpass.requests.post", return_value=resp) as mock_post:
            buscar_pois_localizacoes(LAT, LON, RAIO, ["metro"])
        query = mock_post.call_args[1]["data"]["data"]
        assert "amenity" in query

    def test_supermercado_amenity_filter(self):
        resp = _mock_overpass_response([])
        with patch("src.search.overpass.requests.post", return_value=resp) as mock_post:
            buscar_pois_localizacoes(LAT, LON, RAIO, ["supermercado"])
        query = mock_post.call_args[1]["data"]["data"]
        assert "supermarket" in query

    def test_unknown_category_returns_empty(self):
        # Category not in AMENITY_MAP with no amenities → no partes → empty result
        result = buscar_pois_localizacoes(LAT, LON, RAIO, ["categoria_inexistente"])
        assert result.get("categoria_inexistente") == []

    def test_escola_amenity_filter_includes_school(self):
        resp = _mock_overpass_response([])
        with patch("src.search.overpass.requests.post", return_value=resp) as mock_post:
            buscar_pois_localizacoes(LAT, LON, RAIO, ["escola"])
        query = mock_post.call_args[1]["data"]["data"]
        assert "school" in query

    def test_hospital_amenity_filter(self):
        resp = _mock_overpass_response([])
        with patch("src.search.overpass.requests.post", return_value=resp) as mock_post:
            buscar_pois_localizacoes(LAT, LON, RAIO, ["hospital"])
        query = mock_post.call_args[1]["data"]["data"]
        assert "hospital" in query

    def test_restaurante_amenity_filter(self):
        resp = _mock_overpass_response([])
        with patch("src.search.overpass.requests.post", return_value=resp) as mock_post:
            buscar_pois_localizacoes(LAT, LON, RAIO, ["restaurante"])
        query = mock_post.call_args[1]["data"]["data"]
        assert "restaurant" in query

    def test_json_parse_exception_returns_empty(self):
        resp = MagicMock()
        resp.json.side_effect = ValueError("invalid json")
        with patch("src.search.overpass.requests.post", return_value=resp):
            result = buscar_pois_localizacoes(LAT, LON, RAIO, ["escola"])
        assert result["escola"] == []
