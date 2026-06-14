"""Configuracoes centrais do projeto (caminhos, features, hiperparametros).

Tudo que e' "conhecimento compartilhado" entre gerar dados -> preprocess ->
treino -> API mora aqui, para nao haver divergencia de nomes de colunas.

O projeto usa um dataset SINTETICO de imoveis brasileiros (precos em R$, areas
em m2), gerado com base nos padroes de uma amostra real de 20 anuncios.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

# Amostra real (20 linhas) usada apenas como referencia para a simulacao.
AMOSTRA_REAL_CSV = ROOT_DIR / "query_result_2026-06-14T13_32_13.86254108Z.csv"
AMOSTRA_ENCODING = "latin-1"

SIM_CSV = RAW_DIR / "imoveis_simulado.csv"        # dataset sintetico bruto (100k)
PROCESSED_CSV = PROCESSED_DIR / "imoveis.csv"     # dataset limpo completo
TRAIN_CSV = PROCESSED_DIR / "treino.csv"          # 80%
TEST_CSV = PROCESSED_DIR / "teste.csv"            # 20%

MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Geracao de dados sinteticos
# ---------------------------------------------------------------------------
N_LINHAS = 100_000

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
# Mapeia: coluna_no_csv_bruto -> nome_canonico_usado_no_projeto/API
COLUMN_MAP = {
    "type": "tipo",
    "neighborhood": "bairro",
    "city": "cidade",
    "state": "estado",
    "bedrooms": "quartos",
    "bathrooms": "banheiros",
    "area": "area",                 # m2
    "parking_spaces": "vagas_garagem",
    "condo_fees": "condominio",     # R$/mes
    "iptu": "iptu",                 # R$/ano
    "lat": "latitude",
    "long": "longitude",
    "price_buy": "preco",           # alvo (target), em R$
}

# Colunas que precisam de parsing especial (formato BR) ao ler o CSV bruto.
COLS_PRECO_BR = ["preco"]   # ponto = separador de milhar: "1.500.000" -> 1500000
COLS_AREA_BR = ["area"]     # virgula = decimal: "29,32" -> 29.32

TARGET = "preco"

# Features usadas pelo modelo.
CATEGORICAL_FEATURES = ["tipo", "estado", "cidade"]
NUMERIC_FEATURES = [
    "quartos",
    "banheiros",
    "area",
    "vagas_garagem",
    "condominio",
    "iptu",
    "latitude",
    "longitude",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Colunas mantidas no dataset processado (features + alvo + referencia).
COLUNAS_PROCESSADAS = FEATURES + [TARGET, "bairro"]

# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
MOEDA = "BRL"
UNIDADE_AREA = "m2"
