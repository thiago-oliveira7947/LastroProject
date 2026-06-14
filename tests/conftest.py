"""Fixtures compartilhadas entre todos os módulos de teste."""
from __future__ import annotations

import json
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock


# ─── Dataset mínimo ───────────────────────────────────────────────────────────

COLUNAS_DF = [
    "tipo", "estado", "cidade", "quartos", "banheiros", "area",
    "vagas_garagem", "condominio", "iptu", "latitude", "longitude",
    "preco", "bairro",
]


def _df_minimo(n: int = 10) -> pd.DataFrame:
    """DataFrame com as mesmas colunas do CSV processado."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "tipo":         ["apartment", "house", "commercial", "land"] * (n // 4) + ["apartment"] * (n % 4),
        "estado":       ["São Paulo"] * n,
        "cidade":       ["guarulhos"] * n,
        "quartos":      rng.integers(1, 5, n).tolist(),
        "banheiros":    rng.integers(1, 3, n).tolist(),
        "area":         rng.uniform(40, 200, n).round(2).tolist(),
        "vagas_garagem":rng.integers(0, 3, n).tolist(),
        "condominio":   rng.uniform(0, 1000, n).round(0).tolist(),
        "iptu":         rng.uniform(0, 5000, n).round(0).tolist(),
        "latitude":     rng.uniform(-23.6, -23.4, n).round(6).tolist(),
        "longitude":    rng.uniform(-46.7, -46.4, n).round(6).tolist(),
        "preco":        rng.integers(150_000, 1_500_000, n).tolist(),
        "bairro":       ["Centro"] * n,
    })


@pytest.fixture
def df_imoveis():
    return _df_minimo(20)


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


@pytest.fixture
def mock_requests_get():
    with __import__("unittest.mock", fromlist=["patch"]).patch("requests.get") as m:
        yield m


@pytest.fixture
def mock_requests_post():
    with __import__("unittest.mock", fromlist=["patch"]).patch("requests.post") as m:
        yield m
