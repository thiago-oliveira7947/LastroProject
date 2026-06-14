"""Limpa o CSV bruto de imoveis e divide em treino (80%) / teste (20%).

O parser de numeros e' tolerante ao formato BR (ponto = milhar, virgula =
decimal), entao funciona tanto no dataset sintetico quanto na amostra real.

Uso:
    python -m src.data.preprocess
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def _num_preco_br(serie: pd.Series) -> pd.Series:
    """'1.500.000' -> 1500000 ; '185000' -> 185000 (remove separador de milhar)."""
    txt = serie.astype(str).str.strip().str.replace(".", "", regex=False)
    return pd.to_numeric(txt, errors="coerce")


def _num_area_br(serie: pd.Series) -> pd.Series:
    """'29,32' -> 29.32 ; '68' -> 68.0 (virgula decimal)."""
    txt = serie.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(txt, errors="coerce")


def carregar_bruto(caminho=None, encoding="utf-8") -> pd.DataFrame:
    caminho = caminho or config.SIM_CSV
    if not caminho.exists():
        raise FileNotFoundError(
            f"CSV bruto nao encontrado em {caminho}. "
            "Gere primeiro:  python -m src.data.generate"
        )
    return pd.read_csv(caminho, encoding=encoding)


def processar(df: pd.DataFrame) -> pd.DataFrame:
    faltando = [c for c in config.COLUMN_MAP if c not in df.columns]
    if faltando:
        raise KeyError(f"Colunas esperadas ausentes no CSV bruto: {faltando}")

    df = df[list(config.COLUMN_MAP.keys())].rename(columns=config.COLUMN_MAP)

    # Parsing de numeros no formato BR.
    for col in config.COLS_PRECO_BR:
        df[col] = _num_preco_br(df[col])
    for col in config.COLS_AREA_BR:
        df[col] = _num_area_br(df[col])

    # Demais numericos (ja' costumam vir limpos).
    for col in ["quartos", "banheiros", "vagas_garagem", "condominio",
                "iptu", "latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # So' interessa venda com preco valido.
    df = df[df[config.TARGET].notna() & (df[config.TARGET] > 0)]

    # Remove linhas sem feature essencial; preenche extras numericos com 0.
    df = df.dropna(subset=["tipo", "estado", "cidade", "area"])
    df[["condominio", "iptu", "vagas_garagem"]] = (
        df[["condominio", "iptu", "vagas_garagem"]].fillna(0)
    )

    df = df[config.COLUNAS_PROCESSADAS].reset_index(drop=True)
    return df


def main() -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = processar(carregar_bruto())
    df.to_csv(config.PROCESSED_CSV, index=False, encoding="utf-8")

    treino, teste = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    treino.to_csv(config.TRAIN_CSV, index=False, encoding="utf-8")
    teste.to_csv(config.TEST_CSV, index=False, encoding="utf-8")

    print(f"Dataset processado: {len(df):,} linhas.")
    print(f"  treino (80%): {len(treino):,}  -> {config.TRAIN_CSV.name}")
    print(f"  teste  (20%): {len(teste):,}  -> {config.TEST_CSV.name}")
    print("\nPreview:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
