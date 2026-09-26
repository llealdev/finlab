# FinLab

Sistema de análise financeira baseado em **RAG** (Retrieval-Augmented Generation) e **agentes de LLM**. O projeto coleta relatórios oficiais da SEC (10-K e 10-Q) e notícias do Yahoo Finance, indexa esse conteúdo no **Qdrant** com busca híbrida e expõe uma API em **FastAPI** capaz de:

- buscar trechos relevantes nos documentos (`/search`);
- responder perguntas com base nesses documentos (`/rag`);
- gerar uma recomendação de investimento estruturada (**BUY / HOLD / SELL**) a partir de análises fundamentalista, de momentum e de sentimento (`/agent`).

## Arquitetura

```
             ┌──────────────── Ingestão (ingestion/) ────────────────┐
 SEC EDGAR ──► 10-K / 10-Q ──► SemanticChunker (HDBSCAN) ──┐          │
 Yahoo     ──► notícias   ──► SimpleChunker ───────────────┼─► embeddings ──► Qdrant
             └───────────────────────────────────────────── │ dense + sparse + ColBERT
                                                                      │
             ┌──────────────── API (api/) ────────────────────────────▼──┐
 cliente ──► │ /search  → busca híbrida (dense + sparse → RRF → ColBERT) │
             │ /rag     → busca + resposta do LLM                        │
             │ /agent   → ticker → 3 análises em paralelo → recomendação │
             └───────────────────────────────────────────────────────────┘
```

### Busca híbrida

Cada trecho é armazenado com três vetores:

| Vetor     | Modelo padrão                    | Papel                                 |
|-----------|----------------------------------|---------------------------------------|
| `dense`   | `intfloat/multilingual-e5-large` | similaridade semântica (1024 dims)    |
| `sparse`  | `Qdrant/bm25`                    | correspondência por palavra-chave     |
| `colbert` | `colbert-ir/colbertv2.0`         | reranking por *late interaction*      |

Na consulta, os resultados de `dense` e `sparse` são combinados com **RRF** (Reciprocal Rank Fusion) e depois reordenados pelo **ColBERT**. Os scores são normalizados em relação ao melhor resultado.

Cada ponto no Qdrant tem o payload `{"text": <trecho>, "source": <metadados>}`. Os metadados incluem `ticker`, `form_type` (relatórios) ou `source="yahoo_finance"` (notícias), e são usados como filtros.

### Agente de recomendação (`/agent`)

1. **Extração do ticker** — primeiro por um mapeamento estático (ex.: "apple" → `AAPL`); se não encontrar, um LLM extrai o ticker. Sem ticker válido, a API retorna **400**.
2. **Três análises em paralelo**, cada uma com seu próprio filtro de busca e prompt:
   - **Fundamentalista** — baseada no 10-K (relatório anual);
   - **Momentum** — baseada no 10-Q (relatório trimestral);
   - **Sentimento** — baseada em notícias do Yahoo Finance.
3. **Agregação** — um último prompt combina as três análises em uma recomendação final com ação, confiança, justificativa, riscos, oportunidades e horizonte de tempo.

As saídas do LLM são validadas com **Pydantic** via [`instructor`](https://github.com/instructor-ai/instructor). O LLM é acessado por qualquer endpoint compatível com a API da OpenAI (por padrão, o opencode.ai Zen).

## Estrutura do projeto

```
api/            API FastAPI (routers → services → Qdrant/LLM)
  config/       configurações, prompts e mapeamento empresa → ticker
  models/       schemas Pydantic de requisição, resposta e saídas do LLM
  routers/      endpoints /search, /rag e /agent
  services/     embeddings, busca, RAG, extração de ticker e agente
ingestion/      scripts de criação da collection e ingestão de dados
  utils/        clientes EDGAR / Yahoo Finance e chunkers
evaluations/    avaliações em 4 níveis (testes, integração, Langfuse, LLM-as-judge)
guardrails/     demos de guardrails com guardrails-ai
```

## Pré-requisitos

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- Uma instância do [Qdrant](https://qdrant.tech/) (local ou Qdrant Cloud)
- Uma chave de API para um LLM compatível com a API da OpenAI
- (Opcional) Conta no [Langfuse](https://langfuse.com/) para as avaliações de níveis 3 e 4

## Instalação

```bash
# Apenas a API + ferramentas de desenvolvimento (pytest, ruff)
uv sync

# Tudo: ingestão (torch/CUDA, ~5 GB) e avaliações (langfuse)
uv sync --extra ingestion --extra evaluations
```

As dependências foram separadas de propósito: a API usa `fastembed` (ONNX, CPU) e não depende de torch, o que mantém a imagem Docker pequena.

## Configuração

Crie um arquivo `.env` na raiz (usado pela ingestão e pelas avaliações) e outro em `api/.env` (usado pela API e pelo Docker Compose):

```env
QDRANT_URL=https://seu-cluster.qdrant.io
QDRANT_API_KEY=...
LLM_API_KEY=...
HF_TOKEN=...

# Apenas para as avaliações
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Variáveis opcionais da API (com seus padrões em `api/config/settings.py`): `COLLECTION_NAME`, `DENSE_MODEL`, `SPARSE_MODEL`, `COLBERT_MODEL`, `BASE_URL_API_LLM`, `LLM_MODEL`.

## Uso

### 1. Ingestão de dados

Os scripts devem ser executados **de dentro da pasta `ingestion/`**. O ticker está fixo em cada script; altere-o antes de rodar.

```bash
cd ingestion
uv run python create_collection.py   # cria a collection "financial" (executar uma vez)
uv run python create_indexes.py      # cria índices em source.ticker, source.form_type e source.source
uv run python ingestion.py           # baixa 10-K e 10-Q da SEC e indexa
uv run python news_ingestion.py      # baixa notícias do Yahoo Finance e indexa
```

> Os modelos de embedding da ingestão precisam ser os mesmos da API. Se trocar algum modelo, atualize os scripts de ingestão, `api/config/settings.py` e o tamanho dos vetores em `create_collection.py`. Depois recrie a collection e reindexe os dados.

### 2. Subir a API

A API deve ser executada **de dentro da pasta `api/`**:

```bash
cd api
uv run uvicorn main:app --reload
```

Ou com Docker:

```bash
docker compose up --build
```

A primeira inicialização é lenta porque os três modelos de embedding são baixados (no Docker, ficam em cache no volume `model-cache`). A documentação interativa fica em `http://localhost:8000/docs`.

### 3. Exemplos de requisição

```bash
# Busca
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "risk factors", "limit": 3, "filter": {"ticker": "MSFT", "form_type": "10-K"}}'

# RAG
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "Quais são os principais riscos da Microsoft?", "limit": 3}'

# Agente
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Should I invest in Microsoft?", "limit": 3}'
```

## Avaliações

As avaliações ficam em `evaluations/` e devem ser executadas **de dentro dessa pasta**. Como os nomes dos arquivos têm hífens, passe o arquivo explicitamente ao pytest.

| Nível | Arquivo                         | O que avalia                                               |
|-------|---------------------------------|------------------------------------------------------------|
| 1     | `level-1-unit-tests.py`         | extração de ticker (mapeamento estático e fallback do LLM) |
| 2     | `level-2-unit-tests.py`         | endpoint `/agent` via HTTP (requer a API rodando)          |
| 3     | `level-3-human-annotation.py`   | mesmo do nível 2, com traces enviados ao Langfuse          |
| 4     | `level-4-llm-as-judge.py`       | um LLM avalia a qualidade dos traces do nível 3            |

```bash
cd evaluations
uv run pytest level-1-unit-tests.py
uv run pytest level-1-unit-tests.py::test_llm_fallback_ibm   # um teste específico
uv run pytest level-2-unit-tests.py                          # API em http://localhost:8000 (ou API_BASE_URL)
uv run python level-3-human-annotation.py
uv run python level-4-llm-as-judge.py
```

## Desenvolvimento

```bash
uv run ruff check .    # lint
uv run ruff format .   # formatação
```
