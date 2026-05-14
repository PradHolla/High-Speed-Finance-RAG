# High-Speed Financial Document RAG System

High-speed RAG pipeline for SEC filing analysis (10-K / 10-Q style PDFs) with:
- page-grounded retrieval evidence,
- citation-focused answer generation,
- automated retrieval and generation evaluation.

## What This Project Does
- Ingests financial filing PDFs into Qdrant with metadata and page spans.
- Retrieves relevant chunks using Bedrock embeddings + vector search.
- Generates responses with three strategies:
- `standard`: grounded QA with inline citations
- `comparison`: structured side-by-side synthesis
- `extraction`: strict JSON-style extraction
- Evaluates:
- retrieval quality (`Recall@k`, `Context Precision`, `MRR`)
- generation quality via LLM-as-a-Judge (factuality/citation/hallucination)

## Current Stack
- Python + `uv` dependency management
- AWS Bedrock (`amazon.titan-embed-text-v2:0` + chat model via `RAG_CHAT_MODEL`)
- Qdrant vector DB (Docker)
- FastAPI backend (`/health`, `/ingest`, `/chat`)
- React + Vite frontend

## Repository Layout
- `ingestion.py`: PDF extraction, semantic chunk merge, embedding upsert
- `retrieval_engine.py`: retrieval + prompting strategies
- `eval_retrieval.py`: retrieval metrics runner
- `eval_generation.py`: LLM-as-a-judge generation evaluator
- `backend_api.py`: FastAPI service
- `frontend/`: chat UI
- `data/`: golden datasets
- `artifacts/`: evaluation outputs

## Prerequisites
- Python 3.10+
- `uv`
- Docker + Docker Compose
- AWS Bedrock access for embedding model + chat/judge model

## Environment
Use either:
1. AWS profile (`~/.aws`) with `AWS_PROFILE`, or
2. Direct env vars (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`)

Optional model overrides:
- `RAG_CHAT_MODEL` (default: `zai.glm-5`)
- `RAG_EMBEDDING_MODEL` (default: `amazon.titan-embed-text-v2:0`)

## Quickstart

### 1) Install Python dependencies
```bash
uv sync
```

### 2) Start backend services
```bash
docker compose up -d --build qdrant backend
```

Health checks:
```bash
curl http://localhost:6333/healthz
curl http://localhost:8000/health
```

### 3) (Optional) Start frontend
```bash
docker compose up -d --build frontend
```

App URLs:
- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- Qdrant Dashboard: `http://localhost:6333/dashboard`

### 4) Ingest a filing
CLI path:
```bash
uv run ingestion.py
```

Or use frontend ingest form / backend `/ingest` endpoint.

## Evaluation

### Retrieval evaluation
```bash
uv run eval_retrieval.py \
  --golden-path data/golden_dataset_pratheek_25.jsonl \
  --top-k 5 \
  --output artifacts/retrieval_eval_pratheek25.json
```

### Generation evaluation (LLM-as-a-Judge)
```bash
uv run eval_generation.py \
  --golden-path data/golden_dataset_aditya_25.jsonl \
  --top-k 5 \
  --strategy standard \
  --judge-model zai.glm-5 \
  --judge-temperature 0.0 \
  --judge-max-tokens 1200 \
  --output artifacts/generation_eval_aditya25_full_rerun.json
```

```bash
uv run eval_generation.py \
  --golden-path data/golden_dataset_pratheek_25.jsonl \
  --top-k 5 \
  --strategy standard \
  --judge-model zai.glm-5 \
  --judge-temperature 0.0 \
  --judge-max-tokens 1200 \
  --output artifacts/generation_eval_pratheek25_full.json
```

## Latest Metrics

### Retrieval (combined run)
Source: `artifacts/retrieval_eval_45.json`
- Questions: `45`
- Top-k: `5`
- Recall@5: `0.8889`
- Context Precision: `0.5156`
- MRR: `0.7889`

### Generation (Aditya 25)
Source: `artifacts/generation_eval_aditya25_full_rerun.json`
- Questions: `25`
- Avg Final Score: `91.6`
- Avg Factual Accuracy: `4.76 / 5`
- Avg Citation Quality: `4.8 / 5`
- Hallucination Rate: `0.04`
- Judge success/repair/failure: `1.0 / 0.0 / 0.0`

### Generation (Pratheek 25)
Source: `artifacts/generation_eval_pratheek25_full.json`
- Questions: `25`
- Avg Final Score: `91.2`
- Avg Factual Accuracy: `4.72 / 5`
- Avg Citation Quality: `4.8 / 5`
- Hallucination Rate: `0.0`
- Judge success/repair/failure: `1.0 / 0.0 / 0.0`

## Notes
- Retrieval is strong on recall; precision is the main improvement area.
- Judge reliability hardening is implemented (schema validation + repair path + telemetry).
- Current benchmark scope is centered on available filings/datasets in `data/`.
