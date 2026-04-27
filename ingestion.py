import pdfplumber
import boto3
import json
import uuid
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
qdrant = QdrantClient("http://localhost:6333")

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

def process_and_ingest_pdf(file_path: str, collection_name: str, company: str, year: str):
    """Extracts text, applies semantic chunking, and upserts to Qdrant."""
    print(f"Processing {file_path}...")
    
    full_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n\n"

    # Basic Semantic Chunking: Split by double newlines to preserve paragraphs/tables
    raw_chunks = [chunk.strip() for chunk in full_text.split('\n\n') if len(chunk.strip()) > 100]
    
    # Further group smaller chunks to avoid overwhelming the DB with tiny vectors
    # Further group smaller chunks to avoid overwhelming the DB with tiny vectors
    merged_chunks = []
    current_chunk = ""
    
    for chunk in raw_chunks:
        if len(current_chunk) + len(chunk) < 2000:  # Roughly 400-500 tokens
            # Only add a space if current_chunk isn't empty
            current_chunk += (" " + chunk if current_chunk else chunk)
        else:
            # Prevent appending an empty string if a single chunk is > 2000 chars
            if current_chunk:
                merged_chunks.append(current_chunk.strip())
            current_chunk = chunk
            
    if current_chunk:
        merged_chunks.append(current_chunk.strip())

    print(f"Generated {len(merged_chunks)} chunks. Generating embeddings...")

    points = []
    for idx, chunk in enumerate(merged_chunks):
        vector = get_embedding(chunk)
        
        # Attach rich metadata for the Retrieval & QA Leads to filter by
        payload = {
            "text": chunk,
            "company": company,
            "year": year,
            "chunk_id": idx
        }
        
        points.append(
            PointStruct(
                id=str(uuid.uuid4()), 
                vector=vector, 
                payload=payload
            )
        )

    print("Upserting to Qdrant...")
    qdrant.upsert(
        collection_name=collection_name,
        points=points
    )
    print("Ingestion complete.")

if __name__ == "__main__":
    collection = "financial_reports"
    setup_qdrant_collection(collection)
    
    # Example execution for your downloaded Microsoft 10-K
    process_and_ingest_pdf(
        file_path="10-K.pdf", 
        collection_name=collection,
        company="Microsoft",
        year="2025"
    )