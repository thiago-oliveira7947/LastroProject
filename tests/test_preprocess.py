"""Testes para src/data/preprocess.py."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

import src.config as cfg
from src.data.preprocess import (
    _num_preco_br,
    _num_area_br,
    carregar_bruto,
    processar,
    main,
)


# ─── _num_preco_br ────────────────────────────────────────────────────────────

class TestNumPrecoBr:
    def test_removes_thousands_separator(self):
        s = pd.Series(["1.500.000"])
        result = _num_preco_br(s)
        assert result[0] == 1500000

    def test_plain_number(self):
        s = pd.Series(["185000"])
        result = _num_preco_br(s)
        assert result[0] == 185000

    def test_large_number(self):
        s = pd.Series(["12.000.000"])
        result = _num_preco_br(s)
        assert result[0] == 12000000

    def test_invalid_returns_nan(self):
        s = pd.Series(["consulte"])
        result = _num_preco_br(s)
        assert pd.isna(result[0])

    def test_empty_string_returns_nan(self):
        s = pd.Series([""])
        result = _num_preco_br(s)
        assert pd.isna(result[0])

    def test_strips_whitespace(self):
        s = pd.Series(["  500000  "])
        result = _num_preco_br(s)
        assert result[0] == 500000

    def test_multiple_values(self):
        s = pd.Series(["100.000", "200.000", "300.000"])
        result = _num_preco_br(s)
        assert list(result) == [100000, 200000, 300000]

    def test_no_separator(self):
        s = pd.Series(["50000"])
        result = _num_preco_br(s)
        assert result[0] == 50000

    def test_returns_series(self):
        s = pd.Series(["100.000"])
        result = _num_preco_br(s)
        assert isinstance(result, pd.Series)


# ─── _num_area_br ─────────────────────────────────────────────────────────────

class TestNumAreaBr:
    def test_comma_decimal(self):
        s = pd.Series(["29,32"])
        result = _num_area_br(s)
        assert result[0] == pytest.approx(29.32)

    def test_integer_area(self):
        s = pd.Series(["68"])
        result = _num_area_br(s)
        assert result[0] == 68.0

    def test_invalid_returns_nan(self):
        s = pd.Series(["consulte"])
        result = _num_area_br(s)
        assert pd.isna(result[0])

    def test_strips_whitespace(self):
        s = pd.Series(["  90,5  "])
        result = _num_area_br(s)
        assert result[0] == pytest.approx(90.5)

    def test_multiple_values(self):
        s = pd.Series(["60,0", "75,5", "120"])
        result = _num_area_br(s)
        assert result[0] == pytest.approx(60.0)
        assert result[1] == pytest.approx(75.5)
        assert result[2] == pytest.approx(120.0)

    def test_returns_series(self):
        s = pd.Series(["50"])
        result = _num_area_br(s)
        assert isinstance(result, pd.Series)


# ─── Fixture: raw DataFrame ────────────────────────────────────────────────────

def _make_raw_df(n: int = 20) -> pd.DataFrame:
    """DataFrame bruto no formato do CSV gerado por generate.py."""
    rng = np.random.default_rng(0)
    tipos = (["apartment", "house", "commercial", "land"] * (n // 4 + 1))[:n]
    return pd.DataFrame({
        "type":           tipos,
        "buy":            [True] * n,
        "rent":           [False] * n,
        "price_rent":     [0] * n,
        "price_buy":      rng.integers(200_000, 1_500_000, n).tolist(),
        "condo_fees":     rng.uniform(0, 800, n).round(0).tolist(),
        "iptu":           rng.uniform(0, 5000, n).round(0).tolist(),
        "bedrooms":       rng.integers(1, 5, n).tolist(),
        "bathrooms":      rng.integers(1, 3, n).tolist(),
        "area":           rng.uniform(40, 200, n).round(2).tolist(),
        "parking_spaces": rng.integers(0, 3, n).tolist(),
        "lat":            rng.uniform(-23.6, -23.4, n).round(6).tolist(),
        "long":           rng.uniform(-46.7, -46.4, n).round(6).tolist(),
        "neighborhood":   ["Centro"] * n,
        "city":           ["guarulhos"] * n,
        "state":          ["São Paulo"] * n,
        "country":        ["Brasil"] * n,
    })


# ─── carregar_bruto ───────────────────────────────────────────────────────────

class TestCarregarBruto:
    def test_loads_csv(self, tmp_path, monkeypatch):
        df = _make_raw_df(10)
        csv_path = tmp_path / "imoveis_simulado.csv"
        df.to_csv(csv_path, index=False)
        result = carregar_bruto(caminho=csv_path)
        assert len(result) == 10

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="CSV bruto"):
            carregar_bruto(caminho=tmp_path / "nonexistent.csv")

    def test_uses_default_sim_csv_path(self, tmp_path, monkeypatch):
        df = _make_raw_df(5)
        csv_path = tmp_path / "sim.csv"
        df.to_csv(csv_path, index=False)
        monkeypatch.setattr(cfg, "SIM_CSV", csv_path)
        result = carregar_bruto()
        assert len(result) == 5


# ─── processar ────────────────────────────────────────────────────────────────

class TestProcessar:
    def test_returns_dataframe(self):
        df = processar(_make_raw_df(20))
        assert isinstance(df, pd.DataFrame)

    def test_renames_columns(self):
        df = processar(_make_raw_df(20))
        assert "tipo" in df.columns
        assert "preco" in df.columns
        assert "bairro" in df.columns

    def test_removes_original_column_names(self):
        df = processar(_make_raw_df(20))
        assert "type" not in df.columns
        assert "price_buy" not in df.columns

    def test_drops_rows_with_zero_price(self):
        raw = _make_raw_df(10)
        raw.loc[0, "price_buy"] = 0
        df = processar(raw)
        assert 0 not in df["preco"].values

    def test_drops_rows_with_null_price(self):
        raw = _make_raw_df(10)
        raw.loc[0, "price_buy"] = None
        df = processar(raw)
        assert df["preco"].notna().all()

    def test_drops_rows_with_null_tipo(self):
        raw = _make_raw_df(10)
        raw.loc[0, "type"] = None
        df = processar(raw)
        assert df["tipo"].notna().all()

    def test_drops_rows_with_null_area(self):
        raw = _make_raw_df(10)
        raw.loc[0, "area"] = None
        df = processar(raw)
        assert df["area"].notna().all()

    def test_fills_condominio_na_with_zero(self):
        raw = _make_raw_df(10)
        raw.loc[0, "condo_fees"] = None
        df = processar(raw)
        assert df["condominio"].notna().all()
        assert df.loc[0, "condominio"] == 0.0

    def test_fills_iptu_na_with_zero(self):
        raw = _make_raw_df(10)
        raw.loc[0, "iptu"] = None
        df = processar(raw)
        assert df["iptu"].notna().all()

    def test_fills_vagas_na_with_zero(self):
        raw = _make_raw_df(10)
        raw.loc[0, "parking_spaces"] = None
        df = processar(raw)
        assert df["vagas_garagem"].notna().all()

    def test_raises_key_error_on_missing_columns(self):
        raw = _make_raw_df(5)
        raw = raw.drop(columns=["type"])
        with pytest.raises(KeyError, match="ausentes"):
            processar(raw)

    def test_output_has_only_expected_columns(self):
        df = processar(_make_raw_df(20))
        assert set(df.columns) == set(cfg.COLUNAS_PROCESSADAS)

    def test_numeric_types(self):
        df = processar(_make_raw_df(20))
        assert pd.api.types.is_float_dtype(df["area"]) or pd.api.types.is_integer_dtype(df["area"])
        assert pd.api.types.is_numeric_dtype(df["preco"])

    def test_price_in_br_format_parsed(self):
        raw = _make_raw_df(5)
        raw["price_buy"] = raw["price_buy"].apply(lambda x: f"{x:,.0f}".replace(",", "."))
        df = processar(raw)
        assert df["preco"].notna().all()
        assert (df["preco"] > 0).all()

    def test_area_in_br_format_parsed(self):
        raw = _make_raw_df(5)
        raw["area"] = raw["area"].apply(lambda x: str(x).replace(".", ","))
        df = processar(raw)
        assert df["area"].notna().all()

    def test_reset_index(self):
        df = processar(_make_raw_df(20))
        assert df.index.tolist() == list(range(len(df)))


# ─── main ─────────────────────────────────────────────────────────────────────

class TestMain:
    def test_main_creates_output_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(cfg, "PROCESSED_CSV", tmp_path / "imoveis.csv")
        monkeypatch.setattr(cfg, "TRAIN_CSV", tmp_path / "treino.csv")
        monkeypatch.setattr(cfg, "TEST_CSV", tmp_path / "teste.csv")

        raw = _make_raw_df(50)
        raw_csv = tmp_path / "raw.csv"
        raw.to_csv(raw_csv, index=False)
        monkeypatch.setattr(cfg, "SIM_CSV", raw_csv)

        main()

        assert (tmp_path / "imoveis.csv").exists()
        assert (tmp_path / "treino.csv").exists()
        assert (tmp_path / "teste.csv").exists()

    def test_main_splits_80_20(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(cfg, "PROCESSED_CSV", tmp_path / "imoveis.csv")
        monkeypatch.setattr(cfg, "TRAIN_CSV", tmp_path / "treino.csv")
        monkeypatch.setattr(cfg, "TEST_CSV", tmp_path / "teste.csv")

        raw = _make_raw_df(50)
        raw_csv = tmp_path / "raw.csv"
        raw.to_csv(raw_csv, index=False)
        monkeypatch.setattr(cfg, "SIM_CSV", raw_csv)

        main()

        treino = pd.read_csv(tmp_path / "treino.csv")
        teste = pd.read_csv(tmp_path / "teste.csv")
        total = len(treino) + len(teste)
        assert abs(len(treino) / total - 0.8) < 0.05
