"""Treina o modelo de previsao de preco e salva o pipeline + metricas.

Le os arquivos ja' divididos (treino.csv / teste.csv) gerados pelo preprocess.
O modelo e' um Pipeline completo (pre-processamento + regressor), entao a API
so' precisa chamar .predict() com os dados crus.

Uso:
    python -m src.models.train
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config


def carregar_treino_teste() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.TRAIN_CSV.exists() or not config.TEST_CSV.exists():
        raise FileNotFoundError(
            "treino.csv/teste.csv nao encontrados. Rode antes:\n"
            "  python -m src.data.generate\n"
            "  python -m src.data.preprocess"
        )
    return pd.read_csv(config.TRAIN_CSV), pd.read_csv(config.TEST_CSV)


def construir_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", config.NUMERIC_FEATURES),
        ]
    )
    modelo = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=config.RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("modelo", modelo)])


def avaliar(pipeline, X, y) -> dict:
    pred = pipeline.predict(X)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(root_mean_squared_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "mape_pct": float(np.mean(np.abs((y - pred) / y)) * 100),
    }


def treinar() -> dict:
    treino, teste = carregar_treino_teste()
    X_train, y_train = treino[config.FEATURES], treino[config.TARGET]
    X_test, y_test = teste[config.FEATURES], teste[config.TARGET]

    pipeline = construir_pipeline()
    print(f"Treinando com {len(X_train):,} amostras...")
    pipeline.fit(X_train, y_train)

    metricas = avaliar(pipeline, X_test, y_test)
    metricas.update(n_treino=int(len(X_train)), n_teste=int(len(X_test)))

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)

    df_full = pd.concat([treino, teste])
    metadata = {
        "treinado_em": datetime.now(timezone.utc).isoformat(),
        "modelo": "HistGradientBoostingRegressor",
        "fonte_dados": "sintetico (baseado em amostra real de 20 anuncios)",
        "features": config.FEATURES,
        "target": config.TARGET,
        "metricas": metricas,
        "moeda_preco": config.MOEDA,
        "unidade_area": config.UNIDADE_AREA,
        "tipos_conhecidos": sorted(df_full["tipo"].unique().tolist()),
        "estados_conhecidos": sorted(df_full["estado"].unique().tolist()),
        "cidades_conhecidas": sorted(df_full["cidade"].unique().tolist()),
    }
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n=== Metricas (conjunto de teste - 20%) ===")
    print(f"  MAE  : R$ {metricas['mae']:,.0f}  (erro medio absoluto)")
    print(f"  RMSE : R$ {metricas['rmse']:,.0f}")
    print(f"  MAPE : {metricas['mape_pct']:.1f}%  (erro percentual medio)")
    print(f"  R2   : {metricas['r2']:.3f}  (1.0 = perfeito)")
    print(f"\nModelo salvo em : {config.MODEL_PATH}")
    print(f"Metadados em    : {config.METADATA_PATH}")
    return metadata


if __name__ == "__main__":
    treinar()
