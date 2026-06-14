"""Testes para src/models/train.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import src.config as cfg
from src.models.train import (
    carregar_treino_teste,
    construir_pipeline,
    avaliar,
    treinar,
)


def _make_treino_teste_csvs(tmp_path: Path, n_treino=200, n_teste=50):
    """Cria treino.csv e teste.csv mínimos para testes."""
    rng = np.random.default_rng(42)
    n = n_treino + n_teste

    def _df(size):
        return pd.DataFrame({
            "tipo":         (["apartment", "house"] * (size // 2 + 1))[:size],
            "estado":       ["São Paulo"] * size,
            "cidade":       ["guarulhos"] * size,
            "quartos":      rng.integers(1, 5, size).tolist(),
            "banheiros":    rng.integers(1, 3, size).tolist(),
            "area":         rng.uniform(40, 200, size).round(2).tolist(),
            "vagas_garagem":rng.integers(0, 3, size).tolist(),
            "condominio":   rng.uniform(0, 800, size).round(0).tolist(),
            "iptu":         rng.uniform(0, 3000, size).round(0).tolist(),
            "latitude":     rng.uniform(-23.6, -23.4, size).round(6).tolist(),
            "longitude":    rng.uniform(-46.7, -46.5, size).round(6).tolist(),
            "preco":        rng.integers(150_000, 1_500_000, size).tolist(),
            "bairro":       ["Centro"] * size,
        })

    treino = _df(n_treino)
    teste = _df(n_teste)
    treino.to_csv(tmp_path / "treino.csv", index=False)
    teste.to_csv(tmp_path / "teste.csv", index=False)
    return treino, teste


# ─── carregar_treino_teste ────────────────────────────────────────────────────

class TestCarregarTreinoTeste:
    def test_raises_when_files_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "TRAIN_CSV", tmp_path / "no_treino.csv")
        monkeypatch.setattr(cfg, "TEST_CSV", tmp_path / "no_teste.csv")
        with pytest.raises(FileNotFoundError, match="treino.csv"):
            carregar_treino_teste()

    def test_loads_both_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "TRAIN_CSV", tmp_path / "treino.csv")
        monkeypatch.setattr(cfg, "TEST_CSV", tmp_path / "teste.csv")
        _make_treino_teste_csvs(tmp_path, 100, 25)
        treino, teste = carregar_treino_teste()
        assert len(treino) == 100
        assert len(teste) == 25

    def test_returns_dataframes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "TRAIN_CSV", tmp_path / "treino.csv")
        monkeypatch.setattr(cfg, "TEST_CSV", tmp_path / "teste.csv")
        _make_treino_teste_csvs(tmp_path)
        treino, teste = carregar_treino_teste()
        assert isinstance(treino, pd.DataFrame)
        assert isinstance(teste, pd.DataFrame)

    def test_raises_when_only_train_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "TRAIN_CSV", tmp_path / "no_treino.csv")
        monkeypatch.setattr(cfg, "TEST_CSV", tmp_path / "teste.csv")
        (tmp_path / "teste.csv").touch()
        with pytest.raises(FileNotFoundError):
            carregar_treino_teste()

    def test_raises_when_only_test_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "TRAIN_CSV", tmp_path / "treino.csv")
        monkeypatch.setattr(cfg, "TEST_CSV", tmp_path / "no_teste.csv")
        (tmp_path / "treino.csv").touch()
        with pytest.raises(FileNotFoundError):
            carregar_treino_teste()


# ─── construir_pipeline ───────────────────────────────────────────────────────

class TestConstruirPipeline:
    def test_returns_pipeline(self):
        pipeline = construir_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_has_pre_step(self):
        pipeline = construir_pipeline()
        assert "pre" in pipeline.named_steps

    def test_has_modelo_step(self):
        pipeline = construir_pipeline()
        assert "modelo" in pipeline.named_steps

    def test_modelo_is_hist_gradient_boosting(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        pipeline = construir_pipeline()
        assert isinstance(pipeline.named_steps["modelo"], HistGradientBoostingRegressor)

    def test_pre_is_column_transformer(self):
        from sklearn.compose import ColumnTransformer
        pipeline = construir_pipeline()
        assert isinstance(pipeline.named_steps["pre"], ColumnTransformer)

    def test_pipeline_fits_and_predicts(self, tmp_path):
        """Smoke test: pipeline treina e prediz sem erro."""
        rng = np.random.default_rng(42)
        n = 100
        X = pd.DataFrame({
            "tipo":         (["apartment", "house"] * (n // 2)),
            "estado":       ["São Paulo"] * n,
            "cidade":       ["guarulhos"] * n,
            "quartos":      rng.integers(1, 5, n).tolist(),
            "banheiros":    rng.integers(1, 3, n).tolist(),
            "area":         rng.uniform(40, 200, n).round(2).tolist(),
            "vagas_garagem":rng.integers(0, 3, n).tolist(),
            "condominio":   rng.uniform(0, 800, n).round(0).tolist(),
            "iptu":         rng.uniform(0, 3000, n).round(0).tolist(),
            "latitude":     rng.uniform(-23.6, -23.4, n).round(6).tolist(),
            "longitude":    rng.uniform(-46.7, -46.5, n).round(6).tolist(),
        })
        y = pd.Series(rng.integers(150_000, 1_500_000, n).tolist())
        pipeline = construir_pipeline()
        pipeline.fit(X, y)
        pred = pipeline.predict(X)
        assert len(pred) == n
        assert all(p > 0 for p in pred)


# ─── avaliar ──────────────────────────────────────────────────────────────────

class TestAvaliar:
    @pytest.fixture
    def fitted_pipeline(self):
        rng = np.random.default_rng(42)
        n = 100
        X = pd.DataFrame({
            "tipo":         (["apartment", "house"] * (n // 2)),
            "estado":       ["São Paulo"] * n,
            "cidade":       ["guarulhos"] * n,
            "quartos":      rng.integers(1, 5, n).tolist(),
            "banheiros":    rng.integers(1, 3, n).tolist(),
            "area":         rng.uniform(40, 200, n).round(2).tolist(),
            "vagas_garagem":rng.integers(0, 3, n).tolist(),
            "condominio":   rng.uniform(0, 800, n).round(0).tolist(),
            "iptu":         rng.uniform(0, 3000, n).round(0).tolist(),
            "latitude":     rng.uniform(-23.6, -23.4, n).round(6).tolist(),
            "longitude":    rng.uniform(-46.7, -46.5, n).round(6).tolist(),
        })
        y = pd.Series(rng.integers(150_000, 1_500_000, n).tolist())
        pipeline = construir_pipeline()
        pipeline.fit(X, y)
        return pipeline, X, y

    def test_returns_dict(self, fitted_pipeline):
        pipeline, X, y = fitted_pipeline
        result = avaliar(pipeline, X, y)
        assert isinstance(result, dict)

    def test_has_all_metric_keys(self, fitted_pipeline):
        pipeline, X, y = fitted_pipeline
        result = avaliar(pipeline, X, y)
        assert "mae" in result
        assert "rmse" in result
        assert "r2" in result
        assert "mape_pct" in result

    def test_mae_positive(self, fitted_pipeline):
        pipeline, X, y = fitted_pipeline
        result = avaliar(pipeline, X, y)
        assert result["mae"] >= 0

    def test_rmse_positive(self, fitted_pipeline):
        pipeline, X, y = fitted_pipeline
        result = avaliar(pipeline, X, y)
        assert result["rmse"] >= 0

    def test_r2_range(self, fitted_pipeline):
        pipeline, X, y = fitted_pipeline
        result = avaliar(pipeline, X, y)
        # R2 can be negative for bad models; on training set usually > 0
        assert isinstance(result["r2"], float)

    def test_mape_nonnegative(self, fitted_pipeline):
        pipeline, X, y = fitted_pipeline
        result = avaliar(pipeline, X, y)
        assert result["mape_pct"] >= 0

    def test_all_values_are_floats(self, fitted_pipeline):
        pipeline, X, y = fitted_pipeline
        result = avaliar(pipeline, X, y)
        for v in result.values():
            assert isinstance(v, float)

    def test_perfect_predictions_mae_zero(self):
        from unittest.mock import MagicMock
        pipeline = MagicMock()
        y = pd.Series([100.0, 200.0, 300.0])
        pipeline.predict.return_value = np.array([100.0, 200.0, 300.0])
        result = avaliar(pipeline, pd.DataFrame(), y)
        assert result["mae"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["r2"] == pytest.approx(1.0)


# ─── treinar (smoke test) ─────────────────────────────────────────────────────

class TestTreinar:
    def test_treinar_creates_model_and_metadata(self, tmp_path, monkeypatch):
        model_path = tmp_path / "model.joblib"
        meta_path = tmp_path / "metadata.json"
        monkeypatch.setattr(cfg, "TRAIN_CSV", tmp_path / "treino.csv")
        monkeypatch.setattr(cfg, "TEST_CSV", tmp_path / "teste.csv")
        monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(cfg, "MODEL_PATH", model_path)
        monkeypatch.setattr(cfg, "METADATA_PATH", meta_path)
        _make_treino_teste_csvs(tmp_path, 200, 50)

        result = treinar()

        assert model_path.exists()
        assert meta_path.exists()
        assert "metricas" in result
        assert "mae" in result["metricas"]
        assert result["metricas"]["n_treino"] == 200
        assert result["metricas"]["n_teste"] == 50

    def test_treinar_metadata_has_features(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "TRAIN_CSV", tmp_path / "treino.csv")
        monkeypatch.setattr(cfg, "TEST_CSV", tmp_path / "teste.csv")
        monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(cfg, "MODEL_PATH", tmp_path / "model.joblib")
        monkeypatch.setattr(cfg, "METADATA_PATH", tmp_path / "metadata.json")
        _make_treino_teste_csvs(tmp_path, 200, 50)

        result = treinar()

        assert result["features"] == cfg.FEATURES
        assert result["target"] == cfg.TARGET
        assert result["modelo"] == "HistGradientBoostingRegressor"

    def test_treinar_metadata_file_valid_json(self, tmp_path, monkeypatch):
        meta_path = tmp_path / "metadata.json"
        monkeypatch.setattr(cfg, "TRAIN_CSV", tmp_path / "treino.csv")
        monkeypatch.setattr(cfg, "TEST_CSV", tmp_path / "teste.csv")
        monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(cfg, "MODEL_PATH", tmp_path / "model.joblib")
        monkeypatch.setattr(cfg, "METADATA_PATH", meta_path)
        _make_treino_teste_csvs(tmp_path, 200, 50)

        treinar()

        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "treinado_em" in data
        assert "tipos_conhecidos" in data
        assert "estados_conhecidos" in data
