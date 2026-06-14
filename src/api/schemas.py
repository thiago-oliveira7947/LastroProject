"""Schemas Pydantic: definem e validam a entrada/saida da API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Imovel(BaseModel):
    """Caracteristicas de um imovel para previsao de preco de venda (R$)."""

    tipo: Literal["apartment", "house", "commercial", "land"] = Field(
        ..., description="Tipo do imovel."
    )
    estado: str = Field(..., description="Estado (ex.: 'São Paulo').", examples=["São Paulo"])
    cidade: str = Field(..., description="Cidade (ex.: 'sao paulo').", examples=["sao paulo"])
    quartos: int = Field(..., ge=0, description="Numero de quartos.")
    banheiros: int = Field(..., ge=0, description="Numero de banheiros.")
    area: float = Field(..., gt=0, description="Area em m2.")
    vagas_garagem: int = Field(..., ge=0, description="Numero de vagas de garagem.")
    condominio: float = Field(0, ge=0, description="Valor do condominio (R$/mes).")
    iptu: float = Field(0, ge=0, description="Valor do IPTU (R$/ano).")
    latitude: float = Field(..., ge=-34, le=6, description="Latitude.")
    longitude: float = Field(..., ge=-74, le=-34, description="Longitude.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class PrevisaoResposta(BaseModel):
    """Resultado da previsao."""

    preco_previsto: float = Field(..., description="Preco estimado de venda.")
    moeda: str = Field("BRL", description="Moeda do preco previsto.")
    faixa_estimada: list[float] = Field(
        ..., description="Faixa aproximada [min, max] baseada no erro medio do modelo."
    )
