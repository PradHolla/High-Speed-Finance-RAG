import pdfplumber
import boto3
import json
import uuid
import re
import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    ScalarQuantization, 
    ScalarQuantizationConfig, 
    ScalarType,
    PointStruct
)
from dotenv import load_dotenv

load_dotenv()

# Initialize Qdrant Client (pointing to your local Docker container)
qdrant = QdrantClient(os.getenv("QDRANT_URL", "http://localhost:6333"))

# We use Titan for embeddings since Bedrock is already wired up
embedding_model_id = "amazon.titan-embed-text-v2:0"
bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

def setup_qdrant_collection(collection_name: str):
    """Creates a new collection in Qdrant with Titan's vector size and int8 quantization."""
    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99, # Keeps the top 99% of values to maintain high accuracy
                    always_ram=True # Keeps quantized vectors in RAM for ultra-fast retrieval
                )
            )
        )
        print(f"Collection '{collection_name}' created with int8 quantization.")
    else:
        print(f"Collection '{collection_name}' already exists.")

def get_embedding(text: str) -> list:
    """Generates a vector embedding using AWS Titan."""
    payload = {
        "inputText": text,
        "dimensions": 1024,
        "normalize": True
    }
    
    # Manual bytes conversion for robust Boto3 encoding
    body_bytes = json.dumps(payload).encode('utf-8')
    
    response = bedrock_client.invoke_model(
        modelId=embedding_model_id,
        contentType="application/json",
        accept="application/json",
        body=body_bytes
    )
    
    response_body = json.loads(response.get('body').read())
    return response_body['embedding']

def _normalize_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return normalized.strip("_").lower()


def _extract_page_chunks(file_path: str) -> list[dict]:
    raw_chunks: list[dict] = []
    with pdfplumber.open(file_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            extracted = page.extract_text() or ""
            for paragraph in extracted.split("\n\n"):
                text = paragraph.strip()
                if len(text) > 100:
                    raw_chunks.append({"text": text, "pages": [page_idx]})
    return raw_chunks


def _merge_chunks(raw_chunks: list[dict], max_chars: int = 2000) -> list[dict]:
    merged: list[dict] = []
    current_text = ""
    current_pages: set[int] = set()

    for chunk in raw_chunks:
        text = chunk["text"]
        pages = set(chunk["pages"])
        if len(current_text) + len(text) < max_chars:
            current_text += (" " + text if current_text else text)
            current_pages.update(pages)
            continue

        if current_text:
            merged.append({
                "text": current_text.strip(),
                "pages": sorted(current_pages),
            })
        current_text = text
        current_pages = set(pages)

    if current_text:
        merged.append({
            "text": current_text.strip(),
            "pages": sorted(current_pages),
        })
    return merged


def process_and_ingest_pdf(
    file_path: str,
    collection_name: str,
    company: str,
    year: str,
    filing_type: str = "10-K",
):
    """Extracts text, applies semantic chunking, and upserts page-aware chunks to Qdrant."""
    print(f"Processing {file_path}...")

    raw_chunks = _extract_page_chunks(file_path)
    merged_chunks = _merge_chunks(raw_chunks)
    print(f"Generated {len(merged_chunks)} chunks. Generating embeddings...")

    source_file = Path(file_path).name
    doc_prefix = f"{_normalize_id(company)}_{_normalize_id(year)}_{_normalize_id(filing_type)}_{_normalize_id(source_file)}"

    points = []
    for idx, chunk in enumerate(merged_chunks):
        vector = get_embedding(chunk["text"])
        page_numbers = chunk["pages"]
        stable_chunk_id = f"{doc_prefix}_c{idx:04d}"

        payload = {
            "text": chunk["text"],
            "company": company,
            "filing_year": str(year),
            "year": str(year),  # Backward-compatible alias for older code paths.
            "filing_type": filing_type,
            "source_file": source_file,
            "chunk_id": stable_chunk_id,
            "page_numbers": page_numbers,
            "page_start": page_numbers[0],
            "page_end": page_numbers[-1],
        }

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
        )

    print("Upserting to Qdrant...")
    qdrant.upsert(collection_name=collection_name, points=points)
    print("Ingestion complete.")


def ingest_documents(collection_name: str, documents: list[dict]):
    """Batch-ingest multiple documents into the same collection."""
    setup_qdrant_collection(collection_name)
    for doc in documents:
        process_and_ingest_pdf(
            file_path=doc["file_path"],
            collection_name=collection_name,
            company=doc["company"],
            year=str(doc["filing_year"]),
            filing_type=doc.get("filing_type", "10-K"),
        )

if __name__ == "__main__":
    collection = "financial_reports"
    ingest_documents(
        collection_name=collection,
        documents=[
            {
                "file_path": "10-K.pdf",
                "company": "Microsoft",
                "filing_year": "2025",
                "filing_type": "10-K",
            }
        ],
    )
