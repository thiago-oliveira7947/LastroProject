"""Testes para src/external/mercadolivre.py."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

import src.external.mercadolivre as ml_module
from src.external.mercadolivre import (
    get_credentials,
    auth_url,
    trocar_codigo,
    esta_autenticado,
    _inferir_tipo,
    _get_attr,
    _normalizar,
    buscar_publico,
    buscar,
)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("ML_CLIENT_ID", raising=False)
    monkeypatch.delenv("ML_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("ML_REDIRECT_URI", raising=False)


@pytest.fixture
def with_credentials(monkeypatch):
    monkeypatch.setenv("ML_CLIENT_ID", "test_id")
    monkeypatch.setenv("ML_CLIENT_SECRET", "test_secret")


@pytest.fixture
def valid_token_data():
    return {
        "access_token": "ACCESS123",
        "refresh_token": "REFRESH456",
        "expires_at": time.time() + 3600,
        "expires_in": 3600,
    }


# ─── get_credentials ──────────────────────────────────────────────────────────

class TestGetCredentials:
    def test_returns_none_without_env(self):
        assert get_credentials() is None

    def test_returns_tuple_with_both_env_vars(self, monkeypatch):
        monkeypatch.setenv("ML_CLIENT_ID", "myid")
        monkeypatch.setenv("ML_CLIENT_SECRET", "mysecret")
        result = get_credentials()
        assert result == ("myid", "mysecret")

    def test_returns_none_with_only_client_id(self, monkeypatch):
        monkeypatch.setenv("ML_CLIENT_ID", "myid")
        assert get_credentials() is None

    def test_returns_none_with_only_client_secret(self, monkeypatch):
        monkeypatch.setenv("ML_CLIENT_SECRET", "mysecret")
        assert get_credentials() is None

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("ML_CLIENT_ID", "  myid  ")
        monkeypatch.setenv("ML_CLIENT_SECRET", "  mysecret  ")
        result = get_credentials()
        assert result == ("myid", "mysecret")

    def test_empty_strings_returns_none(self, monkeypatch):
        monkeypatch.setenv("ML_CLIENT_ID", "")
        monkeypatch.setenv("ML_CLIENT_SECRET", "")
        assert get_credentials() is None


# ─── auth_url ─────────────────────────────────────────────────────────────────

class TestAuthUrl:
    def test_returns_empty_without_credentials(self):
        assert auth_url() == ""

    def test_returns_url_with_credentials(self, with_credentials):
        url = auth_url()
        assert "auth.mercadolibre.com.br" in url
        assert "test_id" in url
        assert "response_type=code" in url

    def test_url_contains_redirect_uri(self, with_credentials, monkeypatch):
        monkeypatch.setenv("ML_REDIRECT_URI", "http://myapp.com/callback")
        url = auth_url()
        assert "myapp.com" in url

    def test_url_contains_scope_read(self, with_credentials):
        url = auth_url()
        assert "scope=read" in url

    def test_default_redirect_is_localhost(self, with_credentials):
        url = auth_url()
        assert "localhost" in url


# ─── trocar_codigo ────────────────────────────────────────────────────────────

class TestTrocarCodigo:
    def test_returns_false_without_credentials(self):
        assert trocar_codigo("mycode") is False

    def test_returns_true_on_success(self, with_credentials, tmp_path, monkeypatch):
        monkeypatch.setattr(ml_module, "_TOKEN_FILE", tmp_path / ".ml_token.json")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "access_token": "AT123",
            "refresh_token": "RT456",
            "expires_in": 3600,
        }
        with patch("src.external.mercadolivre.requests.post", return_value=mock_resp), \
             patch("src.external.mercadolivre.time.time", return_value=1000.0):
            result = trocar_codigo("mycode")
        assert result is True

    def test_saves_token_file(self, with_credentials, tmp_path, monkeypatch):
        token_file = tmp_path / ".ml_token.json"
        monkeypatch.setattr(ml_module, "_TOKEN_FILE", token_file)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "access_token": "AT123",
            "refresh_token": "RT456",
            "expires_in": 3600,
        }
        with patch("src.external.mercadolivre.requests.post", return_value=mock_resp), \
             patch("src.external.mercadolivre.time.time", return_value=1000.0):
            trocar_codigo("mycode")
        assert token_file.exists()
        data = json.loads(token_file.read_text())
        assert data["access_token"] == "AT123"
        assert "expires_at" in data

    def test_returns_false_when_no_access_token_in_response(self, with_credentials, tmp_path, monkeypatch):
        monkeypatch.setattr(ml_module, "_TOKEN_FILE", tmp_path / ".ml_token.json")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "invalid_code"}
        with patch("src.external.mercadolivre.requests.post", return_value=mock_resp):
            result = trocar_codigo("badcode")
        assert result is False

    def test_returns_false_on_exception(self, with_credentials):
        with patch("src.external.mercadolivre.requests.post", side_effect=ConnectionError()):
            result = trocar_codigo("mycode")
        assert result is False


# ─── _renovar_token (indirect via _access_token) ──────────────────────────────

class TestRenovarToken:
    def test_renovar_token_no_credentials_returns_none(self):
        from src.external.mercadolivre import _renovar_token
        result = _renovar_token("refresh_token")
        assert result is None

    def test_renovar_token_success(self, with_credentials, tmp_path, monkeypatch):
        from src.external.mercadolivre import _renovar_token
        token_file = tmp_path / ".ml_token.json"
        monkeypatch.setattr(ml_module, "_TOKEN_FILE", token_file)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "access_token": "NEW_AT",
            "refresh_token": "NEW_RT",
            "expires_in": 3600,
        }
        with patch("src.external.mercadolivre.requests.post", return_value=mock_resp), \
             patch("src.external.mercadolivre.time.time", return_value=1000.0):
            result = _renovar_token("old_refresh")
        assert result["access_token"] == "NEW_AT"

    def test_renovar_token_exception_returns_none(self, with_credentials):
        from src.external.mercadolivre import _renovar_token
        with patch("src.external.mercadolivre.requests.post", side_effect=ConnectionError()):
            result = _renovar_token("refresh")
        assert result is None


# ─── _access_token ────────────────────────────────────────────────────────────

class TestAccessToken:
    def test_returns_none_when_file_missing(self, tmp_path, monkeypatch):
        from src.external.mercadolivre import _access_token
        monkeypatch.setattr(ml_module, "_TOKEN_FILE", tmp_path / "no_file.json")
        assert _access_token() is None

    def test_returns_access_token_when_valid(self, tmp_path, monkeypatch, valid_token_data):
        from src.external.mercadolivre import _access_token
        token_file = tmp_path / ".ml_token.json"
        token_file.write_text(json.dumps(valid_token_data))
        monkeypatch.setattr(ml_module, "_TOKEN_FILE", token_file)
        result = _access_token()
        assert result == "ACCESS123"

    def test_triggers_refresh_when_expired(self, tmp_path, monkeypatch, with_credentials):
        from src.external.mercadolivre import _access_token
        expired_data = {
            "access_token": "OLD_AT",
            "refresh_token": "RF123",
            "expires_at": 0,  # expired
        }
        token_file = tmp_path / ".ml_token.json"
        token_file.write_text(json.dumps(expired_data))
        monkeypatch.setattr(ml_module, "_TOKEN_FILE", token_file)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "access_token": "NEW_AT",
            "refresh_token": "NEW_RF",
            "expires_in": 3600,
        }
        with patch("src.external.mercadolivre.requests.post", return_value=mock_resp), \
             patch("src.external.mercadolivre.time.time", return_value=1000.0):
            result = _access_token()
        assert result == "NEW_AT"

    def test_returns_none_on_corrupt_file(self, tmp_path, monkeypatch):
        from src.external.mercadolivre import _access_token
        token_file = tmp_path / ".ml_token.json"
        token_file.write_text("INVALID JSON")
        monkeypatch.setattr(ml_module, "_TOKEN_FILE", token_file)
        result = _access_token()
        assert result is None


# ─── esta_autenticado ─────────────────────────────────────────────────────────

class TestEstaAutenticado:
    def test_returns_false_without_token(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ml_module, "_TOKEN_FILE", tmp_path / "no_file.json")
        assert esta_autenticado() is False

    def test_returns_true_with_valid_token(self, tmp_path, monkeypatch, valid_token_data):
        token_file = tmp_path / ".ml_token.json"
        token_file.write_text(json.dumps(valid_token_data))
        monkeypatch.setattr(ml_module, "_TOKEN_FILE", token_file)
        assert esta_autenticado() is True


# ─── _inferir_tipo ────────────────────────────────────────────────────────────

class TestInferirTipo:
    def test_apartamento_detected(self):
        assert _inferir_tipo("Apartamento 3 quartos centro") == "apartment"

    def test_apto_shorthand(self):
        assert _inferir_tipo("Apto com 2 vagas") == "apartment"

    def test_flat_detected(self):
        assert _inferir_tipo("Flat moderno Jardins") == "apartment"

    def test_cobertura_detected(self):
        assert _inferir_tipo("Cobertura duplex") == "apartment"

    def test_studio_detected(self):
        assert _inferir_tipo("Studio no centro") == "apartment"

    def test_casa_detected(self):
        assert _inferir_tipo("Casa 4 quartos") == "house"

    def test_sobrado_detected(self):
        assert _inferir_tipo("Sobrado com piscina") == "house"

    def test_chacara_detected(self):
        assert _inferir_tipo("Chacara sitio natureza") == "house"

    def test_sitio_detected(self):
        assert _inferir_tipo("Sitio com lago") == "house"

    def test_terreno_detected(self):
        assert _inferir_tipo("Terreno 500m2") == "land"

    def test_lote_detected(self):
        # "residencial" contém "residencia" → house ganha; usar input sem ambiguidade
        assert _inferir_tipo("Lote baldio") == "land"

    def test_sala_comercial_detected(self):
        assert _inferir_tipo("Sala comercial 80m2") == "commercial"

    def test_galpao_detected(self):
        assert _inferir_tipo("Galpao industrial") == "commercial"

    def test_loja_detected(self):
        assert _inferir_tipo("Loja ponto comercial") == "commercial"

    def test_escritorio_detected(self):
        assert _inferir_tipo("Escritorio moderno") == "commercial"

    def test_unknown_defaults_apartment(self):
        assert _inferir_tipo("Imóvel sem categoria") == "apartment"

    def test_case_insensitive(self):
        assert _inferir_tipo("APARTAMENTO") == "apartment"


# ─── _get_attr ────────────────────────────────────────────────────────────────

class TestGetAttr:
    def test_finds_attribute(self):
        attrs = [{"id": "BEDROOMS", "value_name": "3"}]
        assert _get_attr(attrs, ["BEDROOMS"]) == 3.0

    def test_returns_zero_when_not_found(self):
        attrs = [{"id": "OTHER", "value_name": "5"}]
        assert _get_attr(attrs, ["BEDROOMS"]) == 0.0

    def test_tries_multiple_ids(self):
        attrs = [{"id": "COVERED_AREA", "value_name": "80"}]
        assert _get_attr(attrs, ["TOTAL_AREA", "COVERED_AREA", "USEFUL_AREA"]) == 80.0

    def test_handles_non_numeric_value(self):
        attrs = [{"id": "BEDROOMS", "value_name": "consultar"}]
        assert _get_attr(attrs, ["BEDROOMS"]) == 0.0

    def test_handles_empty_attrs(self):
        assert _get_attr([], ["BEDROOMS"]) == 0.0

    def test_handles_none_value_name(self):
        attrs = [{"id": "BEDROOMS", "value_name": None}]
        assert _get_attr(attrs, ["BEDROOMS"]) == 0.0

    def test_strips_non_numeric_chars(self):
        attrs = [{"id": "PARKING_LOTS", "value_name": "2 vagas"}]
        assert _get_attr(attrs, ["PARKING_LOTS"]) == 2.0


# ─── _normalizar ──────────────────────────────────────────────────────────────

class TestNormalizar:
    def _raw_item(self, **overrides):
        base = {
            "id": "ML123",
            "title": "Apartamento 3 quartos em Guarulhos",
            "price": 350000,
            "thumbnail": "https://img.example.com-I.jpg",
            "permalink": "https://ml.com/MLB123",
            "location": {
                "city": {"name": "Guarulhos"},
                "state": {"name": "São Paulo"},
                "neighborhood": {"name": "Centro"},
                "latitude": -23.46,
                "longitude": -46.53,
            },
            "address": {},
            "attributes": [
                {"id": "BEDROOMS", "value_name": "3"},
                {"id": "BATHROOMS", "value_name": "2"},
                {"id": "TOTAL_AREA", "value_name": "90"},
                {"id": "PARKING_LOTS", "value_name": "1"},
            ],
        }
        base.update(overrides)
        return base

    def test_basic_fields(self):
        result = _normalizar(self._raw_item())
        assert result["id"] == "ML123"
        assert result["preco"] == 350000.0
        assert result["cidade"] == "guarulhos"
        assert result["estado"] == "São Paulo"
        assert result["bairro"] == "Centro"
        assert result["quartos"] == 3
        assert result["banheiros"] == 2
        assert result["area"] == 90.0
        assert result["vagas_garagem"] == 1

    def test_thumbnail_normalized(self):
        result = _normalizar(self._raw_item())
        assert "-O.jpg" in result["thumbnail"]
        assert "-I.jpg" not in result["thumbnail"]

    def test_tipo_inferred_from_title(self):
        result = _normalizar(self._raw_item(title="Casa 4 quartos"))
        assert result["tipo"] == "house"

    def test_latitude_longitude_extracted(self):
        result = _normalizar(self._raw_item())
        assert result["latitude"] == -23.46
        assert result["longitude"] == -46.53

    def test_no_location_returns_none_coords(self):
        raw = self._raw_item()
        raw["location"] = {}
        result = _normalizar(raw)
        assert result["latitude"] is None
        assert result["longitude"] is None

    def test_fonte_is_mercadolivre(self):
        result = _normalizar(self._raw_item())
        assert result["fonte"] == "mercadolivre"

    def test_condominio_iptu_zero(self):
        result = _normalizar(self._raw_item())
        assert result["condominio"] == 0.0
        assert result["iptu"] == 0.0

    def test_city_from_search_location_fallback(self):
        raw = self._raw_item()
        raw["location"] = {}
        raw["address"] = {"search_location": {"city": {"name": "Osasco"}, "state": {"name": "SP"}}}
        result = _normalizar(raw)
        assert "osasco" in result["cidade"]

    def test_city_from_address_fallback(self):
        raw = self._raw_item()
        raw["location"] = {}
        raw["address"] = {"city_name": "Mogi"}
        result = _normalizar(raw)
        assert "mogi" in result["cidade"]


# ─── buscar_publico ───────────────────────────────────────────────────────────

class TestBuscarPublico:
    def _make_response(self, status=200, results=None):
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = {"results": results or []}
        if status >= 400:
            resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
        else:
            resp.raise_for_status.return_value = None
        return resp

    def test_returns_results_on_success(self):
        raw = {
            "id": "ML1", "title": "Apartamento", "price": 300000,
            "thumbnail": "img.jpg", "permalink": "url",
            "location": {"city": {"name": "SP"}, "state": {"name": "SP"},
                         "neighborhood": {"name": "Centro"}},
            "address": {}, "attributes": [],
        }
        resp = self._make_response(200, [raw])
        with patch("src.external.mercadolivre.requests.get", return_value=resp):
            results = buscar_publico("apartamento guarulhos")
        assert len(results) == 1

    def test_returns_empty_on_401(self):
        resp = self._make_response(401)
        with patch("src.external.mercadolivre.requests.get", return_value=resp):
            assert buscar_publico("query") == []

    def test_returns_empty_on_403(self):
        resp = self._make_response(403)
        with patch("src.external.mercadolivre.requests.get", return_value=resp):
            assert buscar_publico("query") == []

    def test_returns_empty_on_429(self):
        resp = self._make_response(429)
        with patch("src.external.mercadolivre.requests.get", return_value=resp):
            assert buscar_publico("query") == []

    def test_returns_empty_on_exception(self):
        with patch("src.external.mercadolivre.requests.get", side_effect=ConnectionError()):
            assert buscar_publico("query") == []

    def test_limit_capped_at_48(self):
        resp = self._make_response(200, [])
        with patch("src.external.mercadolivre.requests.get", return_value=resp) as mock_get:
            buscar_publico("query", limit=100)
        params = mock_get.call_args[1]["params"]
        assert params["limit"] <= 48


# ─── buscar ───────────────────────────────────────────────────────────────────

class TestBuscar:
    def test_returns_empty_without_token(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ml_module, "_TOKEN_FILE", tmp_path / "no_token.json")
        assert buscar("query") == []

    def test_returns_results_with_valid_token(self, tmp_path, monkeypatch, valid_token_data):
        token_file = tmp_path / ".ml_token.json"
        token_file.write_text(json.dumps(valid_token_data))
        monkeypatch.setattr(ml_module, "_TOKEN_FILE", token_file)
        raw = {
            "id": "ML1", "title": "Apartamento", "price": 300000,
            "thumbnail": "img.jpg", "permalink": "url",
            "location": {"city": {"name": "SP"}, "state": {"name": "SP"},
                         "neighborhood": {"name": "Centro"}},
            "address": {}, "attributes": [],
        }
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"results": [raw]}
        resp.raise_for_status.return_value = None
        with patch("src.external.mercadolivre.requests.get", return_value=resp):
            results = buscar("apartamento")
        assert len(results) == 1

    def test_removes_token_on_401(self, tmp_path, monkeypatch, valid_token_data):
        token_file = tmp_path / ".ml_token.json"
        token_file.write_text(json.dumps(valid_token_data))
        monkeypatch.setattr(ml_module, "_TOKEN_FILE", token_file)
        resp = MagicMock()
        resp.status_code = 401
        with patch("src.external.mercadolivre.requests.get", return_value=resp):
            result = buscar("query")
        assert result == []
        assert not token_file.exists()

    def test_returns_empty_on_exception(self, tmp_path, monkeypatch, valid_token_data):
        token_file = tmp_path / ".ml_token.json"
        token_file.write_text(json.dumps(valid_token_data))
        monkeypatch.setattr(ml_module, "_TOKEN_FILE", token_file)
        with patch("src.external.mercadolivre.requests.get", side_effect=ConnectionError()):
            result = buscar("query")
        assert result == []

    def test_uses_bearer_token_in_header(self, tmp_path, monkeypatch, valid_token_data):
        token_file = tmp_path / ".ml_token.json"
        token_file.write_text(json.dumps(valid_token_data))
        monkeypatch.setattr(ml_module, "_TOKEN_FILE", token_file)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"results": []}
        resp.raise_for_status.return_value = None
        with patch("src.external.mercadolivre.requests.get", return_value=resp) as mock_get:
            buscar("query")
        headers = mock_get.call_args[1]["headers"]
        assert "Authorization" in headers
        assert "Bearer ACCESS123" in headers["Authorization"]
