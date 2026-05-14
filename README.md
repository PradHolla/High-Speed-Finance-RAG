# High-Speed Financial Document RAG System


## Project Overview
Financial analysts, equity researchers, and risk officers spend hundreds of hours each quarter manually reviewing dense SEC filings (such as 10-K and 10-Q reports) to extract critical performance metrics, identify emerging risk factors, and compare supply chain vulnerabilities. This manual extraction is tedious, error-prone, and acts as a significant bottleneck in financial decision-making. 

This project solves this business problem by implementing a High-Speed Retrieval-Augmented Generation (RAG) system. The pipeline ingests complex financial PDFs, processes dense tabular data, and allows users to query the documents using natural language. The system synthesizes cross-document comparisons, retrieves exact numerical figures, and explicitly cites the source document and page number to completely eliminate hallucinations in a strict business context.

## Completed Features (Core Architecture)
The backend data ingestion and retrieval engine has been fully scaffolded, providing a plug-and-play foundation for prompt engineering and evaluation loops.

* **High-Speed Vector Database Setup:** Configured a local Qdrant instance via Docker, initialized with `int8` scalar quantization to keep quantized vectors in RAM for lightning-fast, memory-efficient retrieval.
* **Strict Dependency Management:** Initialized the project environment utilizing the `uv` package manager for clean, reproducible builds.
* **Automated Data Ingestion Pipeline:** Built a Python ingestion script using `pdfplumber` to extract text from SEC filings while preserving structural integrity.
* **Semantic Chunking Logic:** Implemented a targeted chunking strategy that splits documents based on structural line breaks (double newlines) rather than arbitrary character counts, ensuring financial tables and distinct paragraphs remain intact.
* **Cloud Embedding Integration:** Wired the ingestion pipeline to AWS Bedrock to generate high-quality 1024-dimensional vectors using the Amazon Titan Text v2 embedding model.
* **LangChain Retrieval Bridge:** Developed the `ask_financial_system` core wrapper utilizing `langchain-aws`. This function seamlessly embeds user queries, executes vector similarity searches using Qdrant's `query_points` method, formats the retrieved context blocks with metadata citations, and passes the payload to the LLM generation chain.
* **Web Application MVP:** Added a FastAPI backend (`/health`, `/ingest`, `/chat`) and React chat frontend with one-file ingestion, strategy selection, inline citations, and expandable retrieval evidence panel.

## Yet to be Done (Generation, Evaluation, & UI)
With the core backend engine operational, the remaining tasks focus on the language model's reasoning capabilities, strict accuracy evaluation, and the user interface.

* **Advanced Prompt Engineering:** Implement multi-stage prompting strategies (e.g., Few-Shot extraction for strict JSON outputs and Chain-of-Thought reasoning for comparative queries) to force the model to explicitly cite its source text and prevent hallucinations.
* **Automated SEC Data Scraping:** Integrate the `sec-edgar-downloader` to programmatically fetch batches of 10-K and 10-Q reports directly from the SEC database for target companies.
* **Golden Dataset Creation:** Manually review the ingested SEC filings to construct a robust evaluation dataset consisting of 50 highly complex, verified financial questions and their exact textual or numerical answers.
* **Retrieval Evaluation Loop (Context Precision):** Build an automated testing script to calculate how accurately the Qdrant database fetches the correct text chunks before the language model processes them.
* **Generation Evaluation Loop (LLM-as-a-Judge):** Develop an automated evaluation pipeline utilizing a frontier reasoning model to score the RAG system's final outputs against the Golden Dataset, measuring Exact Match (for numbers) and Faithfulness (penalizing ungrounded claims).
* **Frontend UX Expansion:** Add richer session management, stronger ingest progress reporting, and a run-history view for evaluation traces.

## Setup & Installation Instructions

### Prerequisites
Before running this project, ensure you have the following installed:
* **Python 3.10+**
* **Docker Desktop** (Required for the local Qdrant vector database)
* **uv** (The extremely fast Python package installer and resolver. Install via `curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS/Linux)
* **AWS Account** with Bedrock access explicitly granted for `amazon.titan-embed-text-v2:0` (for embeddings) and your chosen LLM (e.g., `meta.llama3-70b-instruct-v1:0`).

### 1. Initialize the Environment
Clone the repository and navigate into the project folder. Initialize the environment and install the required dependencies using `uv`:

```bash
# Initialize project and create virtual environment
uv init
# Add required dependencies
uv add boto3 python-dotenv qdrant-client pdfplumber langchain-aws langchain-core
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory of the project. This file is required to securely connect to AWS Bedrock via Boto3. 

Add your AWS IAM programmatic access keys to the `.env` file:

```env
AWS_ACCESS_KEY_ID="your_access_key_here"
AWS_SECRET_ACCESS_KEY="your_secret_key_here"
AWS_DEFAULT_REGION="us-east-1"
```
*(Note: Do not commit this file to version control. Ensure `.env` is added to your `.gitignore`).*

### 3. Start the Vector Database
We use Qdrant for lightning-fast, local vector retrieval. Start the database using Docker Compose. The `docker-compose.yml` file is configured for native volume binding to keep your quantized vectors secure locally.

```bash
docker-compose up -d
```
You can verify Qdrant is running by visiting `http://localhost:6333/dashboard` in your browser.

### 4. Run the Data Ingestion Pipeline
Place your target financial documents (e.g., `10-K.pdf`) in the root directory. Run the ingestion script. This script now preserves page-aware metadata in each chunk payload (`page_numbers`, `page_start`, `page_end`, `source_file`, `filing_type`, and stable `chunk_id`), generates embeddings via AWS Titan, and populates the Qdrant collection.

```bash
uv run ingestion.py
```
*Expected output: You should see the terminal log the chunking process and confirm that the ingestion to Qdrant is complete.*

### 5. Test the Retrieval & Generation Engine
Once the database is populated, you can test the core RAG bridge. The retrieval wrapper now returns structured evidence with rank and score (`retrieve_chunks`) and supports metadata filters (`company`, `filing_year`, `filing_type`). Generation supports three strategies: `standard`, `extraction` (strict JSON), and `comparison`.

```bash
uv run retrieval_engine.py
```
*Expected output: The terminal will print the embedded query logs, followed by the synthesized financial answer from the language model.*

### 6. Run Retrieval Evaluation (Context Precision / Recall@k)
Run the retrieval evaluation loop against the golden dataset:

```bash
uv run eval_retrieval.py --golden-path data/golden_dataset.jsonl --top-k 5 --output artifacts/retrieval_eval.json
```

*Expected output: Summary metrics printed in terminal and detailed per-question retrieval diagnostics written to `artifacts/retrieval_eval.json`.*

### 7. Run Generation Evaluation (LLM-as-a-Judge)
Run generation scoring against the same dataset:

```bash
uv run eval_generation.py --golden-path data/golden_dataset.jsonl --top-k 5 --strategy standard --output artifacts/generation_eval.json
```

*Expected output: Judge summary metrics printed in terminal and detailed per-question scoring written to `artifacts/generation_eval.json`.*

### 8. Run the Full Chat Application (FastAPI + React + Qdrant)
Start all services (Qdrant, backend API, and frontend UI):

```bash
docker compose up -d --build
```

Service URLs:
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- Qdrant dashboard: `http://localhost:6333/dashboard`

MVP flow:
1. Open the frontend.
2. Ingest one PDF with company/year/type metadata.
3. Ask questions in chat with strategy + top-k controls.
4. Expand evidence to inspect retrieved chunk rank/score/page citations.
