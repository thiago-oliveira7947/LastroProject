"""Carrega o modelo treinado e faz previsoes de preco.

Pode ser usado pela API ou direto na linha de comando:
    python -m src.models.predict
"""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from src import config


@lru_cache(maxsize=1)
def carregar_modelo():
    """Carrega o pipeline uma unica vez (cacheado)."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado em {config.MODEL_PATH}. "
            "Treine primeiro:  python -m src.models.train"
        )
    return joblib.load(config.MODEL_PATH)


@lru_cache(maxsize=1)
def carregar_metadata() -> dict:
    if not config.METADATA_PATH.exists():
        return {}
    with open(config.METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def prever(imovel: dict) -> float:
    """Recebe um dict com as features e devolve o preco previsto."""
    modelo = carregar_modelo()
    df = pd.DataFrame([{col: imovel[col] for col in config.FEATURES}])
    preco = float(modelo.predict(df)[0])
    return round(preco, 2)


if __name__ == "__main__":
    exemplo = {
        "tipo": "apartment",
        "estado": "São Paulo",
        "cidade": "sao paulo",
        "quartos": 3,
        "banheiros": 2,
        "area": 90.0,
        "vagas_garagem": 2,
        "condominio": 700,
        "iptu": 4000,
        "latitude": -23.55,
        "longitude": -46.63,
    }
    print("Imovel de exemplo:")
    for k, v in exemplo.items():
        print(f"  {k}: {v}")
    print(f"\nPreco previsto: R$ {prever(exemplo):,.2f}")
