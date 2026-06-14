# LastroProject — Previsão de Preço de Imóveis 🏠

Backend em Python + modelo de IA que estima o **preço de venda (R$)** de um imóvel
a partir de suas características: localização (estado/cidade/lat-long), tipo,
nº de quartos, nº de banheiros, metragem (m²) e nº de vagas de garagem.

> ⚠️ **Sobre os dados:** o modelo é treinado com um **dataset sintético de 100.000
> imóveis**, gerado a partir dos padrões de uma amostra real de 20 anúncios
> (`query_result_*.csv`). Os preços são *simulados* — servem para validar o
> pipeline completo. Para uso real, substitua o dataset por dados reais (basta
> que tenham as mesmas colunas) e treine novamente.

## Stack
- **ML:** scikit-learn (`HistGradientBoostingRegressor` dentro de um `Pipeline`)
- **API:** FastAPI + Uvicorn + Pydantic
- **Dados:** pandas / numpy

## Estrutura
```
src/
├── config.py            # caminhos, features e hiperparâmetros (fonte única de verdade)
├── data/
│   ├── generate.py      # gera o dataset sintético de 100k linhas
│   └── preprocess.py    # limpa, trata formato BR e divide 80% treino / 20% teste
├── models/
│   ├── train.py         # treina o pipeline e salva model.joblib + metadata.json
│   └── predict.py       # carrega o modelo e faz previsões
└── api/
    ├── schemas.py       # validação de entrada/saída (Pydantic)
    └── main.py          # endpoints FastAPI
data/raw/                # dataset sintético bruto
data/processed/          # imoveis.csv, treino.csv, teste.csv
models/                  # model.joblib + metadata.json (gerados pelo treino)
tests/                   # testes da API
```

## Como rodar

### 1. Ambiente e dependências
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell/CMD)
# source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

### 2. Gerar dados → treinar modelo
```bash
python -m src.data.generate     # cria data/raw/imoveis_simulado.csv (100k linhas)
python -m src.data.preprocess   # limpa e divide em treino.csv (80%) e teste.csv (20%)
python -m src.models.train      # treina e salva o modelo + métricas
```

### 3. Subir a API
```bash
uvicorn src.api.main:app --reload
```
Acesse a documentação interativa em **http://127.0.0.1:8000/docs**

## Endpoints
| Método | Rota       | Descrição                                  |
|--------|------------|--------------------------------------------|
| GET    | `/`        | Informações do serviço                     |
| GET    | `/health`  | Status (e se o modelo está treinado)       |
| GET    | `/info`    | Métricas do modelo, features e categorias  |
| POST   | `/prever`  | Estima o preço de um imóvel                |

### Exemplo de requisição
```bash
curl -X POST http://127.0.0.1:8000/prever \
  -H "Content-Type: application/json" \
  -d '{
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
    "longitude": -46.63
  }'
```
Resposta:
```json
{
  "preco_previsto": 936566.28,
  "moeda": "BRL",
  "faixa_estimada": [766075.27, 1107057.29]
}
```

## Desempenho atual (conjunto de teste, 20k imóveis)
| Métrica | Valor       | Significado                          |
|---------|-------------|--------------------------------------|
| R²      | **0.94**    | quanto da variação do preço é explicada (1.0 = perfeito) |
| MAPE    | **~13%**    | erro percentual médio                |
| MAE     | ~R$ 170 mil | erro médio absoluto                  |

> Como os dados são sintéticos, essas métricas medem só a capacidade do pipeline
> de aprender o padrão simulado — não a precisão em imóveis reais.

## Testes
```bash
pytest
```

## Próximos passos para uso real
1. Substituir o dataset sintético por dados reais (mesmo schema de colunas).
2. Reativar `bairro` como feature quando houver volume suficiente por bairro.
3. Adicionar histórico de transações (data/preço de vendas anteriores).
4. Tunar hiperparâmetros e validar com validação cruzada.
