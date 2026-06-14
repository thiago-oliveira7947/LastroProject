"""Testes para src/models/predict.py."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
import pytest

import src.config as cfg
import src.models.predict as predict_module
from src.models.predict import carregar_modelo, carregar_metadata, prever


@pytest.fixture(autouse=True)
def clear_lru_caches():
    """Limpa os caches lru_cache antes de cada teste."""
    carregar_modelo.cache_clear()
    carregar_metadata.cache_clear()
    yield
    carregar_modelo.cache_clear()
    carregar_metadata.cache_clear()


@pytest.fixture
def imovel_exemplo():
    return {
        "tipo": "apartment",
        "estado": "São Paulo",
        "cidade": "sao paulo",
        "quartos": 3,
        "banheiros": 2,
        "area": 90.0,
        "vagas_garagem": 2,
        "condominio": 700.0,
        "iptu": 4000.0,
        "latitude": -23.55,
        "longitude": -46.63,
    }


# ─── carregar_modelo ──────────────────────────────────────────────────────────

class TestCarregarModelo:
    def test_raises_file_not_found_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "MODEL_PATH", tmp_path / "no_model.joblib")
        with pytest.raises(FileNotFoundError, match="Modelo"):
            carregar_modelo()

    def test_loads_model_from_path(self, tmp_path, monkeypatch):
        model_path = tmp_path / "model.joblib"
        monkeypatch.setattr(cfg, "MODEL_PATH", model_path)
        mock_model = MagicMock()
        with patch("src.models.predict.joblib.load", return_value=mock_model) as mock_load:
            # Precisa de um arquivo para a verificação de existência
            model_path.touch()
            result = carregar_modelo()
        mock_load.assert_called_once_with(model_path)
        assert result is mock_model

    def test_caches_model(self, tmp_path, monkeypatch):
        model_path = tmp_path / "model.joblib"
        model_path.touch()
        monkeypatch.setattr(cfg, "MODEL_PATH", model_path)
        mock_model = MagicMock()
        with patch("src.models.predict.joblib.load", return_value=mock_model) as mock_load:
            r1 = carregar_modelo()
            r2 = carregar_modelo()
        mock_load.assert_called_once()
        assert r1 is r2

    def test_error_message_mentions_train(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "MODEL_PATH", tmp_path / "no_model.joblib")
        with pytest.raises(FileNotFoundError, match="python -m src.models.train"):
            carregar_modelo()


# ─── carregar_metadata ────────────────────────────────────────────────────────

class TestCarregarMetadata:
    def test_returns_empty_dict_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "METADATA_PATH", tmp_path / "no_meta.json")
        result = carregar_metadata()
        assert result == {}

    def test_loads_metadata_from_json(self, tmp_path, monkeypatch):
        meta_path = tmp_path / "metadata.json"
        expected = {"modelo": "HistGradientBoosting", "r2": 0.95}
        meta_path.write_text(json.dumps(expected))
        monkeypatch.setattr(cfg, "METADATA_PATH", meta_path)
        result = carregar_metadata()
        assert result == expected

    def test_caches_metadata(self, tmp_path, monkeypatch):
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps({"key": "value"}))
        monkeypatch.setattr(cfg, "METADATA_PATH", meta_path)
        r1 = carregar_metadata()
        r2 = carregar_metadata()
        assert r1 is r2

    def test_metadata_fields_accessible(self, tmp_path, monkeypatch):
        meta_path = tmp_path / "metadata.json"
        meta = {
            "modelo": "HistGradientBoostingRegressor",
            "features": ["tipo", "area"],
            "metricas": {"mae": 50000, "r2": 0.9},
        }
        meta_path.write_text(json.dumps(meta))
        monkeypatch.setattr(cfg, "METADATA_PATH", meta_path)
        result = carregar_metadata()
        assert result["metricas"]["r2"] == 0.9


# ─── prever ───────────────────────────────────────────────────────────────────

class TestPrever:
    def test_returns_float(self, tmp_path, monkeypatch, imovel_exemplo):
        model_path = tmp_path / "model.joblib"
        model_path.touch()
        monkeypatch.setattr(cfg, "MODEL_PATH", model_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([450000.0])
        with patch("src.models.predict.joblib.load", return_value=mock_model):
            result = prever(imovel_exemplo)
        assert isinstance(result, float)

    def test_returns_correct_value(self, tmp_path, monkeypatch, imovel_exemplo):
        model_path = tmp_path / "model.joblib"
        model_path.touch()
        monkeypatch.setattr(cfg, "MODEL_PATH", model_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([350000.75])
        with patch("src.models.predict.joblib.load", return_value=mock_model):
            result = prever(imovel_exemplo)
        assert result == pytest.approx(350000.75)

    def test_rounds_to_2_decimals(self, tmp_path, monkeypatch, imovel_exemplo):
        model_path = tmp_path / "model.joblib"
        model_path.touch()
        monkeypatch.setattr(cfg, "MODEL_PATH", model_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([123456.789])
        with patch("src.models.predict.joblib.load", return_value=mock_model):
            result = prever(imovel_exemplo)
        assert result == pytest.approx(123456.79)

    def test_passes_correct_features_to_model(self, tmp_path, monkeypatch, imovel_exemplo):
        model_path = tmp_path / "model.joblib"
        model_path.touch()
        monkeypatch.setattr(cfg, "MODEL_PATH", model_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([500000.0])
        with patch("src.models.predict.joblib.load", return_value=mock_model):
            prever(imovel_exemplo)
        call_df = mock_model.predict.call_args[0][0]
        assert isinstance(call_df, pd.DataFrame)
        for feat in cfg.FEATURES:
            assert feat in call_df.columns

    def test_passes_single_row_df(self, tmp_path, monkeypatch, imovel_exemplo):
        model_path = tmp_path / "model.joblib"
        model_path.touch()
        monkeypatch.setattr(cfg, "MODEL_PATH", model_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([500000.0])
        with patch("src.models.predict.joblib.load", return_value=mock_model):
            prever(imovel_exemplo)
        call_df = mock_model.predict.call_args[0][0]
        assert len(call_df) == 1

    def test_raises_on_missing_model(self, tmp_path, monkeypatch, imovel_exemplo):
        monkeypatch.setattr(cfg, "MODEL_PATH", tmp_path / "no_model.joblib")
        with pytest.raises(FileNotFoundError):
            prever(imovel_exemplo)

    def test_different_inputs_different_predictions(self, tmp_path, monkeypatch):
        model_path = tmp_path / "model.joblib"
        model_path.touch()
        monkeypatch.setattr(cfg, "MODEL_PATH", model_path)
        mock_model = MagicMock()
        mock_model.predict.side_effect = [np.array([300000.0]), np.array([600000.0])]
        base = {
            "tipo": "apartment", "estado": "São Paulo", "cidade": "sp",
            "quartos": 2, "banheiros": 1, "area": 60.0,
            "vagas_garagem": 0, "condominio": 0.0, "iptu": 0.0,
            "latitude": -23.55, "longitude": -46.63,
        }
        large = {**base, "quartos": 4, "area": 180.0}
        with patch("src.models.predict.joblib.load", return_value=mock_model):
            r1 = prever(base)
            r2 = prever(large)
        assert r1 != r2
