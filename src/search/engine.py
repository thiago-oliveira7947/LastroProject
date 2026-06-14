"""Motor de busca e filtragem sobre o dataset de imoveis.

Todas as operacoes sao em memoria (pandas) — sem banco de dados.
"""
from __future__ import annotations

import math

import pandas as pd

from src import config


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


_cache: pd.DataFrame | None = None


def carregar_dataset() -> pd.DataFrame:
    global _cache
    if _cache is None:
        _cache = pd.read_csv(config.PROCESSED_CSV)
    return _cache


def buscar(
    lat: float | None = None,
    lon: float | None = None,
    raio_km: float = 10.0,
    tipos: list[str] | None = None,
    quartos_min: int = 0,
    quartos_max: int = 10,
    banheiros_min: int = 0,
    banheiros_max: int = 10,
    area_min: float = 0.0,
    area_max: float = 5000.0,
    preco_min: float = 0.0,
    preco_max: float = 100_000_000.0,
    vagas_min: int = 0,
    vagas_max: int = 20,
    max_resultados: int = 300,
) -> pd.DataFrame:
    """Filtra o dataset e retorna ate max_resultados linhas, ordenadas por distancia."""
    df = carregar_dataset().copy()

    # Distancia ao ponto buscado
    if lat is not None and lon is not None:
        df["_dist_km"] = df.apply(
            lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]), axis=1
        )
        df = df[df["_dist_km"] <= raio_km].sort_values("_dist_km")
    else:
        df["_dist_km"] = 0.0

    if tipos:
        df = df[df["tipo"].isin(tipos)]

    df = df[
        df["quartos"].between(quartos_min, quartos_max)
        & df["banheiros"].between(banheiros_min, banheiros_max)
        & df["area"].between(area_min, area_max)
        & df["preco"].between(preco_min, preco_max)
        & df["vagas_garagem"].between(vagas_min, vagas_max)
    ]

    return df.head(max_resultados).reset_index(drop=True)


def filtrar_por_poi(
    df: pd.DataFrame,
    poi_locs: dict[str, list[tuple[float, float]]],
    categorias: list[str],
    dist_max_m: float,
) -> pd.DataFrame:
    """Remove linhas onde alguma categoria de POI fica fora de dist_max_m metros."""
    if not categorias or not poi_locs:
        return df

    def tem_todos_pois(row: pd.Series) -> bool:
        for cat in categorias:
            locs = poi_locs.get(cat, [])
            if not locs:
                continue  # sem dados para esta categoria -> nao filtra
            min_d = min(
                haversine_km(row["latitude"], row["longitude"], plat, plon) * 1000
                for plat, plon in locs
            )
            if min_d > dist_max_m:
                return False
        return True

    mask = df.apply(tem_todos_pois, axis=1)
    return df[mask].reset_index(drop=True)
