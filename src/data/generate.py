"""Gera um dataset SINTETICO de imoveis brasileiros (~100k linhas).

Foco em Grande Sao Paulo / Guarulhos, mais outras cidades presentes na amostra
real de 20 anuncios. Os precos sao gerados por formula correlacionada as
features para que o modelo de ML tenha sinal real para aprender.

Saida: data/raw/imoveis_simulado.csv

Uso:
    python -m src.data.generate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Catalogo de localizacoes
# preco_m2 = preco base de venda por m2 (R$).
# peso     = frequencia relativa no dataset (cidades-foco tem peso maior).
# ---------------------------------------------------------------------------
CIDADES = [
    # cidade,                    estado,               preco_m2,  lat,      long,    peso
    # --- Grande Sao Paulo (foco principal) ---
    ("guarulhos",                "São Paulo",           6500, -23.4628, -46.5333, 20),
    ("sao paulo",                "São Paulo",          10000, -23.5505, -46.6333, 20),
    ("osasco",                   "São Paulo",           5800, -23.5325, -46.7919,  8),
    ("mogi das cruzes",          "São Paulo",           4800, -23.5213, -46.1879,  6),
    ("diadema",                  "São Paulo",           5400, -23.6860, -46.6228,  5),
    ("sao bernardo do campo",    "São Paulo",           6500, -23.6939, -46.5650,  6),
    ("sao caetano do sul",       "São Paulo",           7800, -23.6228, -46.5750,  4),
    ("maua",                     "São Paulo",           5000, -23.6678, -46.4611,  4),
    ("ribeirao preto",           "São Paulo",           5500, -21.1784, -47.8097,  5),
    ("sao carlos",               "São Paulo",           5000, -22.0175, -47.8908,  3),
    ("taubate",                  "São Paulo",           4500, -23.0256, -45.5569,  3),
    ("juquia",                   "São Paulo",           3000, -24.3261, -47.6353,  2),
    # --- Outras regioes (presentes na amostra real) ---
    ("porto alegre",             "Rio Grande do Sul",   7000, -30.0346, -51.2177,  6),
    ("canoas",                   "Rio Grande do Sul",   4500, -29.9177, -51.1839,  3),
    ("gravatai",                 "Rio Grande do Sul",   4000, -29.9438, -50.9911,  2),
    ("pelotas",                  "Rio Grande do Sul",   4200, -31.7654, -52.3376,  2),
    ("sao leopoldo",             "Rio Grande do Sul",   4300, -29.7600, -51.1492,  2),
    ("belo horizonte",           "Minas Gerais",        7500, -19.9167, -43.9345,  6),
    ("alterosa",                 "Minas Gerais",        3000, -21.2322, -46.0497,  1),
    ("curitiba",                 "Paraná",              7000, -25.4297, -49.2711,  5),
    ("cuiaba",                   "Mato Grosso",         5000, -15.6014, -56.0979,  2),
]

# ---------------------------------------------------------------------------
# Bairros por cidade (mais realismo)
# ---------------------------------------------------------------------------
BAIRROS_POR_CIDADE: dict[str, list[str]] = {
    "guarulhos": [
        "Centro", "Jardim Maia", "Vila Augusta", "Vila Galvão",
        "Macedo", "Taboão", "Bom Clima", "Jardim São João",
        "Picanço", "Gopouva", "Cumbica", "Jardim Tranquilidade",
        "Vila Rio de Janeiro", "Cidade Serodio", "Portal dos Passaros",
        "Vila Barros", "Jardim Normandia", "Vila Leopoldina",
        "Jardim Zaira", "Parque Continental", "Vila Galvão",
    ],
    "sao paulo": [
        "Centro", "Mooca", "Tatuapé", "Vila Prudente", "Penha",
        "Santana", "Tucuruvi", "Tremembé", "Jardim Analia Franco",
        "Ermelino Matarazzo", "Itaquera", "São Miguel Paulista",
        "Bras", "Belem", "Agua Rasa", "Vila Formosa",
        "Aricanduva", "Carrão", "Vila Matilde", "Sapopemba",
        "Jardim Paulistano", "Consolacao", "Bela Vista",
    ],
    "osasco": [
        "Centro", "Jardim Veloso", "Presidente Altino",
        "Km 18", "Rochdale", "Vila Yara", "Jardim das Flores",
    ],
    "mogi das cruzes": [
        "Centro", "Vila Mogilar", "Jardim Casqueiro",
        "Cezar de Souza", "Jardim São Paulo",
    ],
    "diadema": [
        "Centro", "Piraporinha", "Eldorado", "Conceicao", "Taboao",
    ],
    "sao bernardo do campo": [
        "Centro", "Baeta Neves", "Jardim do Mar", "Rudge Ramos",
        "Nova Petropolis", "Cooperativa",
    ],
    "porto alegre": [
        "Centro", "Moinhos de Vento", "Bela Vista", "Medianeira",
        "Petropolis", "Santana", "Sarandi", "Jardim Botanico",
        "Rio Branco", "Santa Cecilia",
    ],
    "belo horizonte": [
        "Centro", "Mangabeiras", "Savassi", "Lourdes", "Funcionarios",
        "Sao Lucas", "Buritis", "Pampulha", "Contorno",
    ],
    "curitiba": [
        "Centro", "Bacacheri", "Batel", "Agua Verde", "Portao",
        "Boa Vista", "Santa Felicidade", "Cajuru",
    ],
    "_default": [
        "Centro", "Jardim America", "Vila Nova", "Boa Vista",
        "Sao Jose", "Industrial", "Jardim das Flores",
        "Vila Olimpia", "Parque Industrial", "Vila Popular",
    ],
}

TIPOS = ["apartment", "house", "commercial", "land"]
TIPO_PESOS = [0.50, 0.35, 0.10, 0.05]
TIPO_FATOR = {"apartment": 1.00, "house": 0.95, "commercial": 1.05, "land": 0.35}


def gerar(n: int = config.N_LINHAS, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Localizacao -------------------------------------------------------
    pesos = np.array([c[5] for c in CIDADES], dtype=float)
    pesos /= pesos.sum()
    idx = rng.choice(len(CIDADES), size=n, p=pesos)
    cidade = np.array([CIDADES[i][0] for i in idx])
    estado = np.array([CIDADES[i][1] for i in idx])
    preco_m2 = np.array([CIDADES[i][2] for i in idx], dtype=float)
    lat0 = np.array([CIDADES[i][3] for i in idx], dtype=float)
    long0 = np.array([CIDADES[i][4] for i in idx], dtype=float)

    latitude = (lat0 + rng.normal(0, 0.025, n)).round(6)
    longitude = (long0 + rng.normal(0, 0.025, n)).round(6)

    bairro = np.array([
        rng.choice(BAIRROS_POR_CIDADE.get(c, BAIRROS_POR_CIDADE["_default"]))
        for c in cidade
    ])

    # --- Tipo --------------------------------------------------------------
    tipo = rng.choice(TIPOS, size=n, p=TIPO_PESOS)
    is_apt = tipo == "apartment"
    is_house = tipo == "house"
    is_comm = tipo == "commercial"
    is_land = tipo == "land"

    # --- Quartos / banheiros / vagas ---------------------------------------
    quartos = np.zeros(n, dtype=int)
    quartos[is_apt] = rng.integers(1, 5, is_apt.sum())
    quartos[is_house] = rng.integers(2, 6, is_house.sum())
    quartos[is_comm] = rng.integers(0, 3, is_comm.sum())

    banheiros = np.maximum(0, quartos - 1 + rng.integers(0, 3, n))
    banheiros[is_land] = 0
    banheiros[is_comm] = rng.integers(1, 5, n)[is_comm]

    vagas = np.zeros(n, dtype=int)
    vagas[is_apt] = rng.integers(0, 4, is_apt.sum())
    vagas[is_house] = rng.integers(1, 7, is_house.sum())
    vagas[is_comm] = rng.integers(0, 6, is_comm.sum())

    # --- Area (m2) ---------------------------------------------------------
    area = np.empty(n, dtype=float)
    area[is_apt] = 30 + 22 * quartos[is_apt] + rng.normal(0, 12, is_apt.sum())
    area[is_house] = 70 + 45 * quartos[is_house] + rng.normal(0, 40, is_house.sum())
    area[is_comm] = rng.uniform(20, 500, is_comm.sum())
    area[is_land] = rng.uniform(150, 2000, is_land.sum())
    area = np.clip(area, 18, 2500).round(2)

    # --- Preco (R$) --------------------------------------------------------
    fator_tipo = np.array([TIPO_FATOR[t] for t in tipo])
    uplift = 1 + 0.03 * quartos + 0.04 * banheiros + 0.05 * vagas
    ruido = rng.lognormal(mean=0.0, sigma=0.18, size=n)
    preco = area * preco_m2 * fator_tipo * uplift * ruido
    preco[is_land] = (area * preco_m2 * TIPO_FATOR["land"] * rng.lognormal(0, 0.18, n))[is_land]
    preco = np.round(preco, -3).astype(np.int64)
    preco = np.maximum(preco, 50_000)

    # --- Condominio e IPTU -------------------------------------------------
    condominio = np.zeros(n)
    condominio[is_apt] = (area[is_apt] * rng.uniform(4, 10, is_apt.sum())).round(0)
    condominio[is_comm] = (area[is_comm] * rng.uniform(3, 8, is_comm.sum())).round(0)
    iptu = (0.006 * preco * rng.uniform(0.5, 1.5, n)).round(0)
    iptu[is_land] = (0.004 * preco * rng.uniform(0.5, 1.5, n)).round(0)[is_land]

    return pd.DataFrame({
        "type": tipo,
        "buy": True,
        "rent": False,
        "price_rent": 0,
        "price_buy": preco,
        "condo_fees": condominio,
        "iptu": iptu,
        "bedrooms": quartos,
        "bathrooms": banheiros,
        "area": area,
        "parking_spaces": vagas,
        "lat": latitude,
        "long": longitude,
        "neighborhood": bairro,
        "city": cidade,
        "state": estado,
        "country": "Brasil",
    })


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = gerar()
    df.to_csv(config.SIM_CSV, index=False, encoding="utf-8")
    print(f"Dataset sintetico gerado: {df.shape[0]:,} linhas x {df.shape[1]} colunas.")
    print(f"Salvo em: {config.SIM_CSV}")
    print("\nDistribuicao por cidade (top 10):")
    print(df["city"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
