"""Testes para src/data/generate.py."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from src.data.generate import gerar, CIDADES, BAIRROS_POR_CIDADE, TIPOS, TIPO_PESOS


class TestGerar:
    def test_returns_dataframe(self):
        df = gerar(n=100, seed=42)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        df = gerar(n=200, seed=0)
        assert len(df) == 200

    def test_correct_columns(self):
        df = gerar(n=10, seed=1)
        expected = {
            "type", "buy", "rent", "price_rent", "price_buy",
            "condo_fees", "iptu", "bedrooms", "bathrooms", "area",
            "parking_spaces", "lat", "long", "neighborhood", "city",
            "state", "country",
        }
        assert set(df.columns) == expected

    def test_exactly_17_columns(self):
        df = gerar(n=10, seed=1)
        assert df.shape[1] == 17

    def test_reproducible_with_same_seed(self):
        df1 = gerar(n=50, seed=99)
        df2 = gerar(n=50, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_results(self):
        df1 = gerar(n=100, seed=1)
        df2 = gerar(n=100, seed=2)
        assert not df1["price_buy"].equals(df2["price_buy"])

    def test_all_prices_positive(self):
        df = gerar(n=200, seed=42)
        assert (df["price_buy"] > 0).all()

    def test_minimum_price_is_50000(self):
        df = gerar(n=500, seed=42)
        assert (df["price_buy"] >= 50_000).all()

    def test_area_min_18(self):
        df = gerar(n=500, seed=42)
        assert (df["area"] >= 18).all()

    def test_area_max_2500(self):
        df = gerar(n=500, seed=42)
        assert (df["area"] <= 2500).all()

    def test_valid_tipos(self):
        df = gerar(n=200, seed=42)
        assert df["type"].isin(TIPOS).all()

    def test_tipo_distribution_roughly_correct(self):
        df = gerar(n=2000, seed=42)
        counts = df["type"].value_counts(normalize=True)
        # apartment ~50%
        assert 0.40 < counts.get("apartment", 0) < 0.60

    def test_buy_column_all_true(self):
        df = gerar(n=10, seed=1)
        assert df["buy"].all()

    def test_rent_column_all_false(self):
        df = gerar(n=10, seed=1)
        assert (~df["rent"]).all()

    def test_price_rent_all_zero(self):
        df = gerar(n=10, seed=1)
        assert (df["price_rent"] == 0).all()

    def test_country_all_brasil(self):
        df = gerar(n=10, seed=1)
        assert (df["country"] == "Brasil").all()

    def test_bedrooms_zero_for_land(self):
        df = gerar(n=1000, seed=42)
        land = df[df["type"] == "land"]
        assert (land["bedrooms"] == 0).all()

    def test_bathrooms_zero_for_land(self):
        df = gerar(n=1000, seed=42)
        land = df[df["type"] == "land"]
        assert (land["bathrooms"] == 0).all()

    def test_bedrooms_nonnegative(self):
        df = gerar(n=500, seed=42)
        assert (df["bedrooms"] >= 0).all()

    def test_bathrooms_nonnegative(self):
        df = gerar(n=500, seed=42)
        assert (df["bathrooms"] >= 0).all()

    def test_parking_spaces_nonnegative(self):
        df = gerar(n=500, seed=42)
        assert (df["parking_spaces"] >= 0).all()

    def test_condo_fees_nonnegative(self):
        df = gerar(n=500, seed=42)
        assert (df["condo_fees"] >= 0).all()

    def test_iptu_nonnegative(self):
        df = gerar(n=500, seed=42)
        assert (df["iptu"] >= 0).all()

    def test_lat_lon_in_brazil_range(self):
        df = gerar(n=500, seed=42)
        assert (df["lat"].between(-35, 6)).all()
        assert (df["long"].between(-75, -28)).all()

    def test_valid_cities_from_catalog(self):
        df = gerar(n=200, seed=42)
        catalog_cities = {c[0] for c in CIDADES}
        assert df["city"].isin(catalog_cities).all()

    def test_valid_states_from_catalog(self):
        df = gerar(n=200, seed=42)
        catalog_states = {c[1] for c in CIDADES}
        assert df["state"].isin(catalog_states).all()

    def test_neighborhoods_valid(self):
        df = gerar(n=200, seed=42)
        all_bairros = {b for bairros in BAIRROS_POR_CIDADE.values() for b in bairros}
        assert df["neighborhood"].isin(all_bairros).all()

    def test_area_rounded_to_2_decimals(self):
        df = gerar(n=50, seed=42)
        areas = df["area"].tolist()
        for a in areas:
            assert round(a, 2) == a

    def test_house_min_2_bedrooms(self):
        df = gerar(n=2000, seed=42)
        houses = df[df["type"] == "house"]
        assert (houses["bedrooms"] >= 2).all()

    def test_apartment_min_1_bedroom(self):
        df = gerar(n=2000, seed=42)
        apts = df[df["type"] == "apartment"]
        assert (apts["bedrooms"] >= 1).all()

    def test_small_n(self):
        df = gerar(n=1, seed=0)
        assert len(df) == 1

    def test_condo_fees_zero_for_land(self):
        df = gerar(n=1000, seed=42)
        land = df[df["type"] == "land"]
        # Land properties shouldn't have condo fees (initialized as zeros)
        assert (land["condo_fees"] == 0).all()


class TestMain:
    def test_main_generates_and_saves_csv(self, tmp_path, monkeypatch):
        import src.config as cfg
        import src.data.generate as gen_module
        monkeypatch.setattr(cfg, "RAW_DIR", tmp_path)
        monkeypatch.setattr(cfg, "SIM_CSV", tmp_path / "imoveis_simulado.csv")
        # Patcha gerar para usar n=50 pois o default N_LINHAS é avaliado em import
        small_df = gerar(n=50, seed=42)
        monkeypatch.setattr(gen_module, "gerar", lambda: small_df)
        gen_module.main()
        assert (tmp_path / "imoveis_simulado.csv").exists()
        df = pd.read_csv(tmp_path / "imoveis_simulado.csv")
        assert len(df) == 50
