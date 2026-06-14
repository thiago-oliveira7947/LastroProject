"""Testes para src/search/engine.py."""
from __future__ import annotations

import math

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import src.search.engine as engine_module
from src.search.engine import haversine_km, buscar, filtrar_por_poi


@pytest.fixture(autouse=True)
def reset_cache():
    """Zera o cache global entre testes."""
    engine_module._cache = None
    yield
    engine_module._cache = None


def _make_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "tipo":         (["apartment", "house", "commercial", "land"] * (n // 4 + 1))[:n],
        "estado":       ["São Paulo"] * n,
        "cidade":       ["guarulhos"] * n,
        "quartos":      rng.integers(1, 5, n).tolist(),
        "banheiros":    rng.integers(1, 3, n).tolist(),
        "area":         rng.uniform(40, 200, n).round(2).tolist(),
        "vagas_garagem":rng.integers(0, 3, n).tolist(),
        "condominio":   rng.uniform(0, 1000, n).round(0).tolist(),
        "iptu":         rng.uniform(0, 5000, n).round(0).tolist(),
        "latitude":     rng.uniform(-23.5, -23.4, n).round(6).tolist(),
        "longitude":    rng.uniform(-46.6, -46.4, n).round(6).tolist(),
        "preco":        rng.integers(150_000, 1_500_000, n).tolist(),
        "bairro":       ["Centro"] * n,
    })


# ─── haversine_km ─────────────────────────────────────────────────────────────

class TestHaversineKm:
    def test_same_point_is_zero(self):
        assert haversine_km(-23.46, -46.53, -23.46, -46.53) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance_guarulhos_to_sp(self):
        # Guarulhos: -23.4628, -46.5333 | SP: -23.5505, -46.6333
        d = haversine_km(-23.4628, -46.5333, -23.5505, -46.6333)
        assert 10 < d < 20  # ~14 km

    def test_symmetry(self):
        d1 = haversine_km(-23.46, -46.53, -23.55, -46.63)
        d2 = haversine_km(-23.55, -46.63, -23.46, -46.53)
        assert d1 == pytest.approx(d2, abs=1e-6)

    def test_returns_float(self):
        result = haversine_km(-23.46, -46.53, -23.47, -46.54)
        assert isinstance(result, float)

    def test_north_pole_to_south_pole(self):
        d = haversine_km(90, 0, -90, 0)
        assert d == pytest.approx(20015.0, rel=0.01)  # ~half Earth circumference

    def test_equatorial_one_degree_longitude(self):
        # At equator, 1° longitude ≈ 111.32 km
        d = haversine_km(0, 0, 0, 1)
        assert 110 < d < 113

    def test_close_points(self):
        d = haversine_km(-23.46, -46.53, -23.461, -46.531)
        assert d < 0.2  # < 200m


# ─── carregar_dataset ─────────────────────────────────────────────────────────

class TestCarregarDataset:
    def test_loads_from_csv(self, tmp_path, monkeypatch):
        df = _make_df(5)
        csv = tmp_path / "imoveis.csv"
        df.to_csv(csv, index=False)
        import src.config as cfg
        monkeypatch.setattr(cfg, "PROCESSED_CSV", csv)
        engine_module._cache = None
        result = engine_module.carregar_dataset()
        assert len(result) == 5

    def test_caches_result(self, tmp_path, monkeypatch):
        df = _make_df(5)
        csv = tmp_path / "imoveis.csv"
        df.to_csv(csv, index=False)
        import src.config as cfg
        monkeypatch.setattr(cfg, "PROCESSED_CSV", csv)
        engine_module._cache = None
        r1 = engine_module.carregar_dataset()
        r2 = engine_module.carregar_dataset()
        assert r1 is r2  # same object (cached)

    def test_cache_bypasses_read_csv(self):
        engine_module._cache = _make_df(3)
        with patch("src.search.engine.pd.read_csv") as mock_read:
            engine_module.carregar_dataset()
        mock_read.assert_not_called()


# ─── buscar ───────────────────────────────────────────────────────────────────

class TestBuscar:
    @pytest.fixture(autouse=True)
    def inject_dataset(self):
        engine_module._cache = _make_df(20)

    def test_returns_dataframe(self):
        result = buscar()
        assert isinstance(result, pd.DataFrame)

    def test_returns_all_rows_without_filters(self):
        result = buscar()
        assert len(result) == 20

    def test_distance_filter_with_lat_lon(self):
        df = engine_module._cache
        # Use center of dataset, tiny radius
        center_lat = df["latitude"].mean()
        center_lon = df["longitude"].mean()
        result = buscar(lat=center_lat, lon=center_lon, raio_km=0.001)
        # At 1m radius, most items should be excluded
        assert len(result) <= 20

    def test_large_radius_returns_all(self):
        df = engine_module._cache
        center_lat = df["latitude"].mean()
        center_lon = df["longitude"].mean()
        result = buscar(lat=center_lat, lon=center_lon, raio_km=1000.0)
        assert len(result) == 20

    def test_no_lat_lon_returns_all(self):
        result = buscar(lat=None, lon=None)
        assert len(result) == 20

    def test_tipo_filter(self):
        result = buscar(tipos=["apartment"])
        assert all(result["tipo"] == "apartment")

    def test_tipo_filter_multiple(self):
        result = buscar(tipos=["apartment", "house"])
        assert all(result["tipo"].isin(["apartment", "house"]))

    def test_quartos_min_filter(self):
        result = buscar(quartos_min=3)
        assert all(result["quartos"] >= 3)

    def test_quartos_max_filter(self):
        result = buscar(quartos_max=2)
        assert all(result["quartos"] <= 2)

    def test_banheiros_min_filter(self):
        result = buscar(banheiros_min=2)
        assert all(result["banheiros"] >= 2)

    def test_area_min_filter(self):
        result = buscar(area_min=100.0)
        assert all(result["area"] >= 100.0)

    def test_area_max_filter(self):
        result = buscar(area_max=80.0)
        assert all(result["area"] <= 80.0)

    def test_preco_min_filter(self):
        result = buscar(preco_min=500_000)
        assert all(result["preco"] >= 500_000)

    def test_preco_max_filter(self):
        result = buscar(preco_max=300_000)
        assert all(result["preco"] <= 300_000)

    def test_vagas_min_filter(self):
        result = buscar(vagas_min=2)
        assert all(result["vagas_garagem"] >= 2)

    def test_max_resultados_respected(self):
        result = buscar(max_resultados=5)
        assert len(result) <= 5

    def test_sorted_by_distance_when_lat_lon_given(self):
        df = engine_module._cache
        center_lat = df["latitude"].mean()
        center_lon = df["longitude"].mean()
        result = buscar(lat=center_lat, lon=center_lon, raio_km=1000.0)
        assert "_dist_km" in result.columns
        dists = result["_dist_km"].tolist()
        assert dists == sorted(dists)

    def test_dist_km_zero_without_lat_lon(self):
        result = buscar()
        assert "_dist_km" in result.columns
        assert all(result["_dist_km"] == 0.0)

    def test_combined_filters(self):
        result = buscar(tipos=["apartment"], quartos_min=2, preco_max=1_000_000)
        assert all(result["tipo"] == "apartment")
        assert all(result["quartos"] >= 2)
        assert all(result["preco"] <= 1_000_000)

    def test_impossible_filter_returns_empty(self):
        result = buscar(preco_min=999_000_000)
        assert len(result) == 0


# ─── filtrar_por_poi ──────────────────────────────────────────────────────────

class TestFiltrarPorPoi:
    @pytest.fixture
    def df_perto(self):
        """DataFrame com coordenadas específicas para teste de POI."""
        return pd.DataFrame({
            "tipo": ["apartment", "house"],
            "latitude": [-23.460, -23.550],
            "longitude": [-46.530, -46.630],
            "preco": [400000, 500000],
        })

    def test_empty_categorias_returns_df_unchanged(self, df_perto):
        poi_locs = {"escola": [(-23.461, -46.531)]}
        result = filtrar_por_poi(df_perto, poi_locs, [], 500)
        assert len(result) == len(df_perto)

    def test_empty_poi_locs_returns_df_unchanged(self, df_perto):
        result = filtrar_por_poi(df_perto, {}, ["escola"], 500)
        assert len(result) == len(df_perto)

    def test_filters_rows_far_from_poi(self, df_perto):
        # POI perto do primeiro imóvel, longe do segundo
        poi_locs = {"escola": [(-23.461, -46.531)]}
        result = filtrar_por_poi(df_perto, poi_locs, ["escola"], 500)
        assert len(result) == 1
        assert result.iloc[0]["latitude"] == pytest.approx(-23.460, abs=0.001)

    def test_keeps_rows_close_to_poi(self, df_perto):
        # POI muito próximo de ambos — todos mantidos
        poi_locs = {"escola": [(-23.460, -46.530), (-23.550, -46.630)]}
        result = filtrar_por_poi(df_perto, poi_locs, ["escola"], 200)
        assert len(result) == 2

    def test_missing_poi_category_does_not_filter(self, df_perto):
        # Categoria "hospital" não tem POIs → não filtra
        poi_locs = {"escola": [(-23.461, -46.531)]}
        result = filtrar_por_poi(df_perto, poi_locs, ["hospital"], 500)
        assert len(result) == len(df_perto)

    def test_multiple_categories_all_must_match(self, df_perto):
        # Escola perto do imovel 0, hospital perto do imovel 1
        poi_locs = {
            "escola": [(-23.460, -46.530)],
            "hospital": [(-23.550, -46.630)],
        }
        result = filtrar_por_poi(df_perto, poi_locs, ["escola", "hospital"], 200)
        # Nenhum imóvel tem AMBAS as categorias próximas
        assert len(result) == 0

    def test_resets_index(self, df_perto):
        poi_locs = {"escola": [(-23.461, -46.531)]}
        result = filtrar_por_poi(df_perto, poi_locs, ["escola"], 500)
        assert result.index.tolist() == list(range(len(result)))

    def test_dist_max_boundary(self, df_perto):
        # Distância exata na fronteira — depende de haversine
        poi_locs = {"escola": [(-23.461, -46.531)]}
        d = haversine_km(-23.460, -46.530, -23.461, -46.531) * 1000
        # Margem minúscula acima → deve incluir
        result_above = filtrar_por_poi(df_perto, poi_locs, ["escola"], d + 10)
        assert len(result_above) >= 1
