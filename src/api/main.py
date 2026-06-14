"""API FastAPI para previsao de preco de imoveis.

Subir o servidor:
    uvicorn src.api.main:app --reload

Docs interativas:  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.models import predict
from src.api.schemas import Imovel, PrevisaoResposta


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega o modelo na subida do servidor (falha cedo se nao existir).
    try:
        predict.carregar_modelo()
        print("Modelo carregado com sucesso.")
    except FileNotFoundError as e:
        print(f"AVISO: {e}")
    yield


app = FastAPI(
    title="Lastro - Previsao de Preco de Imoveis",
    description="API que estima o preco de um imovel a partir de suas caracteristicas.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", tags=["info"])
def raiz():
    return {
        "servico": "Lastro - Previsao de Preco de Imoveis",
        "docs": "/docs",
        "endpoints": ["/health", "/info", "/prever"],
    }


@app.get("/health", tags=["info"])
def health():
    modelo_ok = predict.config.MODEL_PATH.exists()
    return {"status": "ok" if modelo_ok else "modelo_ausente", "modelo_treinado": modelo_ok}


@app.get("/info", tags=["info"])
def info():
    """Metadados do modelo: metricas, features e bairros conhecidos."""
    meta = predict.carregar_metadata()
    if not meta:
        raise HTTPException(status_code=404, detail="Modelo ainda nao foi treinado.")
    return meta


@app.post("/prever", response_model=PrevisaoResposta, tags=["previsao"])
def prever(imovel: Imovel):
    """Estima o preco de um imovel."""
    try:
        modelo = predict.carregar_modelo()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    preco = predict.prever(imovel.model_dump())

    # Faixa aproximada usando o MAE do modelo (se disponivel).
    meta = predict.carregar_metadata()
    mae = meta.get("metricas", {}).get("mae", 0.0)
    faixa = [round(max(preco - mae, 0), 2), round(preco + mae, 2)]

    return PrevisaoResposta(
        preco_previsto=preco,
        moeda=meta.get("moeda_preco", "BRL"),
        faixa_estimada=faixa,
    )
