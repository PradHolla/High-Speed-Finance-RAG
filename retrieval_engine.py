import json
import os
from typing import Any

import boto3
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

qdrant = QdrantClient(os.getenv("QDRANT_URL", "http://localhost:6333"))
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

chat_model_id = os.getenv("RAG_CHAT_MODEL", "zai.glm-5")
embedding_model_id = os.getenv("RAG_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")

llm = ChatBedrockConverse(
    client=bedrock_client,
    model=chat_model_id,
    temperature=0.1,
    max_tokens=1000,
)

STANDARD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert equity researcher. Answer questions based ONLY on provided context.

RULES:
1. Every factual claim must be followed by an inline source citation.
2. Use this exact format: [Source: Company Year FilingType p.X] or [Source: Company Year FilingType p.X-Y]
3. Quote exact figures from context. Do not infer missing numbers.
4. If context is insufficient, respond with: "This information is not available in the provided documents." """),
    ("human", "Context:\n{context}\n\nQuery: {query}"),
])

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial extraction engine. Return ONLY valid JSON using this exact schema:
{{
  "answer": "<string or null>",
  "source_document": "<string or null>",
  "source_pages": "<array of integers or null>",
  "direct_quote": "<string or null>",
  "confidence": "HIGH | MEDIUM | LOW"
}}

If answer is not found, return:
{{"answer": null, "source_document": null, "source_pages": null, "direct_quote": null, "confidence": "LOW"}}"""),
    ("human", """Context:
{context}

Query: {query}

Example output:
{{"answer":"$281.724 billion","source_document":"Microsoft 2025 10-K","source_pages":[71],"direct_quote":"Total revenue 281,724","confidence":"HIGH"}}"""),
])

COT_COMPARISON_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert equity researcher performing a structured comparative analysis.

Use this exact format:
STEP 1 — EVIDENCE EXTRACTION
STEP 2 — SIDE-BY-SIDE COMPARISON
STEP 3 — SYNTHESIS & CONCLUSION

Rules:
- Ground every claim in context.
- Cite sources inline in Step 3.
- If one side has no supporting data, explicitly say so."""),
    ("human", "Context:\n{context}\n\nQuery: {query}"),
])


def get_embedding(text: str) -> list:
    payload = {"inputText": text, "dimensions": 1024, "normalize": True}
    body_bytes = json.dumps(payload).encode("utf-8")
    response = bedrock_client.invoke_model(
        modelId=embedding_model_id,
        contentType="application/json",
        accept="application/json",
        body=body_bytes,
    )
    return json.loads(response.get("body").read())["embedding"]


def _build_query_filter(filters: dict[str, Any] | None) -> Filter | None:
    if not filters:
        return None

    conditions = []
    if filters.get("company"):
        conditions.append(FieldCondition(key="company", match=MatchValue(value=filters["company"])))
    if filters.get("filing_year"):
        value = str(filters["filing_year"])
        conditions.append(FieldCondition(key="filing_year", match=MatchValue(value=value)))
    if filters.get("filing_type"):
        conditions.append(FieldCondition(key="filing_type", match=MatchValue(value=filters["filing_type"])))

    if not conditions:
        return None
    return Filter(must=conditions)


def _format_pages(page_numbers: list[int]) -> str:
    if not page_numbers:
        return "p.?"
    if len(page_numbers) == 1:
        return f"p.{page_numbers[0]}"
    return f"p.{page_numbers[0]}-{page_numbers[-1]}"


def retrieve_chunks(
    user_query: str,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    collection_name: str = "financial_reports",
) -> list[dict]:
    query_vector = get_embedding(user_query)
    query_filter = _build_query_filter(filters)

    search_results = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
    ).points

    rows: list[dict] = []
    for rank, result in enumerate(search_results, start=1):
        payload = result.payload or {}
        rows.append(
            {
                "rank": rank,
                "score": result.score,
                "id": str(result.id),
                "chunk_id": payload.get("chunk_id"),
                "text": payload.get("text", ""),
                "company": payload.get("company", "Unknown"),
                "filing_year": str(payload.get("filing_year", payload.get("year", "Unknown"))),
                "filing_type": payload.get("filing_type", "10-K"),
                "source_file": payload.get("source_file"),
                "page_numbers": payload.get("page_numbers", []),
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
            }
        )
    return rows


def _select_prompt(strategy: str) -> ChatPromptTemplate:
    if strategy == "extraction":
        return EXTRACTION_PROMPT
    if strategy == "comparison":
        return COT_COMPARISON_PROMPT
    return STANDARD_PROMPT


def _build_context_string(chunks: list[dict]) -> str:
    context_blocks = []
    for chunk in chunks:
        pages = _format_pages(chunk.get("page_numbers", []))
        source = (
            f"[Source: {chunk.get('company', 'Unknown')} "
            f"{chunk.get('filing_year', 'Unknown')} "
            f"{chunk.get('filing_type', '10-K')} {pages}]"
        )
        context_blocks.append(f"{source}\n{chunk.get('text', '')}")
    return "\n\n---\n\n".join(context_blocks)


def _validate_extraction_output(text: str) -> str:
    required_keys = {"answer", "source_document", "source_pages", "direct_quote", "confidence"}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Extraction strategy did not return valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Extraction output must be a JSON object.")
    if set(parsed.keys()) != required_keys:
        raise ValueError(f"Extraction JSON keys must match exactly: {sorted(required_keys)}")
    if parsed["confidence"] not in {"HIGH", "MEDIUM", "LOW"}:
        raise ValueError("Extraction JSON confidence must be HIGH, MEDIUM, or LOW.")
    if parsed["source_pages"] is not None:
        if not isinstance(parsed["source_pages"], list) or not all(isinstance(x, int) for x in parsed["source_pages"]):
            raise ValueError("Extraction JSON source_pages must be null or a list of integers.")

    return json.dumps(parsed, ensure_ascii=True)


def generate_from_chunks(chunks: list[dict], user_query: str, strategy: str = "standard") -> str:
    if not chunks:
        return "This information is not available in the provided documents."

    context_string = _build_context_string(chunks)
    prompt = _select_prompt(strategy)
    chain = prompt | llm

    print("Generating response via Bedrock Converse...")
    response = chain.invoke({"context": context_string, "query": user_query})
    text_response = response.content if isinstance(response.content, str) else str(response.content)
    if strategy == "extraction":
        return _validate_extraction_output(text_response)
    return text_response


def ask_financial_system(
    user_query: str,
    top_k: int = 5,
    strategy: str = "standard",
    filters: dict[str, Any] | None = None,
    collection_name: str = "financial_reports",
) -> str:
    print(f"[Strategy: {strategy}] Embedding query and fetching from Qdrant...")

    chunks = retrieve_chunks(
        user_query=user_query,
        top_k=top_k,
        filters=filters,
        collection_name=collection_name,
    )
    if not chunks:
        return "This information is not available in the provided documents."

    print("Generating response via Bedrock Converse...")
    try:
        return generate_from_chunks(chunks=chunks, user_query=user_query, strategy=strategy)
    except Exception as exc:
        return f"API Error during generation: {exc}"


if __name__ == "__main__":
    print("\n=== STANDARD: General Q&A ===")
    print(ask_financial_system("What are the primary macroeconomic risks mentioned?", strategy="standard"))

    print("\n=== EXTRACTION: Structured JSON Output ===")
    print(ask_financial_system("What was the total revenue reported?", strategy="extraction"))

    print("\n=== COMPARISON: Chain-of-Thought Analysis ===")
    print(ask_financial_system("Compare risk factors and key financial growth drivers.", strategy="comparison"))
