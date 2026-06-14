"""Testes basicos da API. Rode com:  pytest"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src import config

client = TestClient(app)

IMOVEL_EXEMPLO = {
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

modelo_existe = config.MODEL_PATH.exists()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


def test_validacao_rejeita_entrada_invalida():
    ruim = dict(IMOVEL_EXEMPLO, area=-5)  # area deve ser > 0
    r = client.post("/prever", json=ruim)
    assert r.status_code == 422


@pytest.mark.skipif(not modelo_existe, reason="modelo ainda nao treinado")
def test_prever_retorna_preco():
    r = client.post("/prever", json=IMOVEL_EXEMPLO)
    assert r.status_code == 200
    body = r.json()
    assert body["preco_previsto"] > 0
    assert len(body["faixa_estimada"]) == 2
