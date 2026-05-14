# FE524 Final Project Report
## High-Speed Financial Document RAG System

### Team Members
- Pradhyumna (Core Architect and Data Pipeline)
- Pallavi (Prompt Engineering and Generation)
- Pratheek (Retrieval Evaluation)
- Aditya (Generation Evaluation, LLM-as-a-Judge)

## Abstract
This project develops a Retrieval-Augmented Generation (RAG) system for financial document analysis, focused on SEC filings (10-K and 10-Q style reports). The system ingests filing PDFs, indexes semantically meaningful chunks in a vector database, retrieves evidence using query embeddings, and generates answers with inline source citations. To ensure reliability, we evaluate the pipeline in two stages: retrieval quality (Recall@k, Context Precision, MRR) and generation quality (LLM-as-a-Judge with factuality, citation quality, and hallucination penalties). Current experiments on two 25-question golden datasets show stable judge reliability (100% successful parse rate) and strong answer quality (average final scores above 91/100), with retrieval performance indicating good recall but moderate context precision.

## 1. Introduction
Financial analysts spend substantial effort manually extracting figures and risk signals from dense filings. This introduces latency, inconsistency, and human error in comparative analysis workflows. The objective of this project is to produce a practical AI system that can answer complex, natural-language financial questions while remaining grounded in document evidence and page-level traceability.

Our target behavior is:
- evidence-first retrieval from filing text,
- grounded synthesis with explicit citations,
- controlled hallucination risk via prompt and evaluation design,
- reproducible benchmarking for both retrieval and generation.

## 2. Data and Task Definition
The current implementation uses:
- `10-K.pdf` (Microsoft fiscal 2025 filing) for ingestion and system testing.
- Two role-owned golden datasets in JSONL format:
- `data/golden_dataset_pratheek_25.jsonl` (25 questions, IDs `001` to `025`)
- `data/golden_dataset_aditya_25.jsonl` (25 questions, IDs `021` to `045`)

Each row contains a question, expected answer/facts, expected page references, and retrieval keywords. This supports both retrieval-stage and generation-stage evaluation.

## 3. Methodology
### 3.1 Retrieval-Augmented Pipeline
The implemented pipeline is:
1. PDF parsing and chunk construction (`ingestion.py`)
2. Embedding generation using Bedrock Titan embeddings
3. Vector indexing and search in Qdrant
4. Prompted answer generation (`retrieval_engine.py`)
5. Automated evaluation (`eval_retrieval.py`, `eval_generation.py`)

Chunks carry metadata (`company`, `filing_year`, `filing_type`, page spans), enabling filtered retrieval and citation formatting.

### 3.2 Prompting Strategies
Three generation strategies are integrated:
- `standard`: grounded narrative QA with citation constraints.
- `comparison`: structured comparative reasoning format.
- `extraction`: strict JSON extraction for machine-readable outputs.

For `extraction`, schema validation is enforced in code, not prompt text alone.

### 3.3 Generation Evaluation Reliability Design
The LLM judge output is normalized through:
- JSON object extraction from potentially noisy outputs,
- strict key-set and type validation,
- coercion/clamping for numeric fields,
- repair pass for malformed judge outputs,
- telemetry fields (`ok`, `repaired`, `failed`) for auditability.

This was added to prevent silent scoring failures and to make evaluation reproducible.

## 4. Experimental Setup
### 4.1 Retrieval Evaluation
Script: `eval_retrieval.py`  
Metrics:
- Recall@k
- Context Precision
- Mean Reciprocal Rank (MRR)

Relevance is determined by page overlap and/or keyword matches against golden references.

### 4.2 Generation Evaluation (LLM-as-a-Judge)
Script: `eval_generation.py`  
Primary judge fields:
- factual_accuracy (1-5)
- citation_quality (1-5)
- hallucination (boolean)
- missing_facts (list)
- unsupported_claims (list)
- final_score (0-100)

Full runs were executed on both 25-question datasets with `top_k=5` and `strategy=standard`.

## 5. Results
### 5.1 Retrieval Results
Artifact: `artifacts/retrieval_eval_45.json`  
- Questions evaluated: 45
- Recall@5: 0.8889
- Context Precision: 0.5156
- MRR: 0.7889

Interpretation: the retriever usually surfaces relevant context, but top-k includes non-trivial noise, reducing precision.

### 5.2 Generation Results
Artifact: `artifacts/generation_eval_aditya25_full_rerun.json`  
- Questions evaluated: 25
- Avg final score: 91.6
- Avg factual accuracy: 4.76/5
- Avg citation quality: 4.8/5
- Hallucination rate: 0.04
- Judge success/repair/failure: 1.0 / 0.0 / 0.0

Artifact: `artifacts/generation_eval_pratheek25_full.json`  
- Questions evaluated: 25
- Avg final score: 91.2
- Avg factual accuracy: 4.72/5
- Avg citation quality: 4.8/5
- Hallucination rate: 0.0
- Judge success/repair/failure: 1.0 / 0.0 / 0.0

Interpretation: answer quality is consistently high on both sets, and judge reliability controls are functioning as intended.

## 6. Role-wise Contributions
### Pradhyumna
- Initialized and stabilized the core infrastructure: `uv`-managed environment, Qdrant setup, PDF ingestion, embeddings pipeline, metadata-aware retrieval, FastAPI integration, and Dockerized runtime.

### Pallavi
- Implemented and integrated multi-strategy prompting (`standard`, `comparison`, `extraction`) with source-grounding instructions and extraction schema discipline.

### Pratheek
- Built retrieval-oriented golden data and retrieval benchmark logic (Recall@k, Context Precision, MRR), enabling quantitative retriever diagnosis.

### Aditya
- Built generation benchmark data and LLM-as-a-Judge scoring pipeline, including reliability hardening for malformed judge outputs.

## 7. Completion Status and Remaining Scope
### Completed
- End-to-end RAG pipeline from ingestion to cited answers.
- Prompt strategy integration in runnable code.
- Retrieval and generation evaluation scripts with persisted artifacts.
- Two separate 25-question golden datasets with role ownership.
- Full generation evaluation runs on both datasets.

### Remaining (Explicit and Bounded)
1. Expand corpus breadth beyond a single filing source (additional companies and 10-Q sets).
2. Improve retrieval context precision (currently ~0.52 on combined run) through chunking/reranking refinements.
3. Define final acceptance thresholds for demo readiness (target values for Recall@k, hallucination rate, and final score).
4. Add brief error-analysis appendix for low-score rows to support presentation defense.

## 8. Reproducibility Commands
```bash
uv run eval_retrieval.py --golden-path data/golden_dataset_pratheek_25.jsonl --top-k 5 --output artifacts/retrieval_eval_pratheek25.json

uv run eval_generation.py --golden-path data/golden_dataset_aditya_25.jsonl --top-k 5 --strategy standard --judge-model zai.glm-5 --judge-temperature 0.0 --judge-max-tokens 1200 --output artifacts/generation_eval_aditya25_full_rerun.json

uv run eval_generation.py --golden-path data/golden_dataset_pratheek_25.jsonl --top-k 5 --strategy standard --judge-model zai.glm-5 --judge-temperature 0.0 --judge-max-tokens 1200 --output artifacts/generation_eval_pratheek25_full.json
```

## 9. FE524 Requirement Mapping
- Group members: listed in Team Members section.
- Business problem: Sections Abstract and Introduction.
- Documents/data used: Section 2.
- Model output evaluation for accuracy: Sections 4 and 5 with concrete metrics and artifacts.
