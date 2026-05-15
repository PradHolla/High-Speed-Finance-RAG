import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ingestion import process_and_ingest_pdf, setup_qdrant_collection
from retrieval_engine import generate_from_chunks, retrieve_chunks


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3)
    strategy: str = "standard"
    top_k: int = Field(default=5, ge=1, le=20)
    collection_name: str = "financial_reports"
    filters: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    answer: str
    strategy: str
    top_k: int
    retrieved_chunks: list[dict]


app = FastAPI(title="High-Speed Finance RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(
    file: UploadFile = File(...),
    company: str = Form(...),
    filing_year: str = Form(...),
    filing_type: str = Form(default="10-K"),
    collection_name: str = Form(default="financial_reports"),
) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    setup_qdrant_collection(collection_name)

    with tempfile.TemporaryDirectory(prefix="rag_upload_") as temp_dir:
        upload_path = Path(temp_dir) / file.filename
        with upload_path.open("wb") as dest:
            shutil.copyfileobj(file.file, dest)

        try:
            process_and_ingest_pdf(
                file_path=str(upload_path),
                collection_name=collection_name,
                company=company,
                year=filing_year,
                filing_type=filing_type,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return {
        "status": "ok",
        "message": "File ingested successfully.",
        "source_file": file.filename,
        "company": company,
        "filing_year": filing_year,
        "filing_type": filing_type,
        "collection_name": collection_name,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        chunks = retrieve_chunks(
            user_query=payload.query,
            top_k=payload.top_k,
            filters=payload.filters,
            collection_name=payload.collection_name,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    try:
        answer = generate_from_chunks(chunks=chunks, user_query=payload.query, strategy=payload.strategy)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return ChatResponse(
        answer=answer,
        strategy=payload.strategy,
        top_k=payload.top_k,
        retrieved_chunks=chunks,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
