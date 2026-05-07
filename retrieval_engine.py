import json
import boto3
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Initialize Backend Clients
qdrant = QdrantClient("http://localhost:6333")
bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

# Initialize LangChain LLM
# Swap this with whichever model you DO have access to (e.g., Llama 3, Amazon Nova, Mistral)
llm = ChatBedrockConverse(
    client=bedrock_client,
    model="zai.glm-5",
    temperature=0.1,
    max_tokens=1000,
)

# ─────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# Three strategies selectable via the `strategy` param in ask_financial_system().
#   "standard"   → Strict inline citations for general Q&A
#   "extraction" → Few-Shot examples forcing structured JSON output
#   "comparison" → Chain-of-Thought for multi-company / cross-period analysis
# ─────────────────────────────────────────────────────────────

# --- Strategy 1: Standard ---
STANDARD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert equity researcher. Answer questions based ONLY on the provided context.

RULES:
1. Every factual claim must be followed immediately by its source in this exact format: [Source: Company Year 10-K/10-Q]
2. Quote exact figures — never round, estimate, or infer.
3. If the answer is not present in the context, respond with: "This information is not available in the provided documents." """),
    ("human", "Context:\n{context}\n\nQuery: {query}")
])

# --- Strategy 2: Few-Shot Extraction (returns JSON) ---
EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial data extraction engine. Your sole job is to extract a specific numerical or factual answer from the provided context and return it as valid JSON.

The JSON must follow this exact schema:
{{
  "answer": "<the direct answer>",
  "source_document": "<Company Year 10-K or 10-Q>",
  "direct_quote": "<verbatim sentence from the source>",
  "confidence": "HIGH | MEDIUM | LOW"
}}

# Examples

<example id="1">
<input>
Query: What was Microsoft's total revenue for fiscal year 2024?
Context: [Source: Microsoft 2024 10-K]
Revenue: $245.1 billion, an increase of 16% compared to fiscal year 2023.
</input>
<assistant_response>
{{
  "answer": "$245.1 billion",
  "growth": "16% increase from fiscal year 2023",
  "source_document": "Microsoft 2024 10-K",
  "direct_quote": "Revenue: $245.1 billion, an increase of 16% compared to fiscal year 2023",
  "confidence": "HIGH"
}}
</assistant_response>
</example>

<example id="2">
<input>
Query: What was Apple's operating income in Q3 2024?
Context: [Source: Apple 2024 10-Q]
Operating income was $29.6 billion for the third fiscal quarter of 2024, compared to $28.3 billion in the prior year quarter.
</input>
<assistant_response>
{{
  "answer": "$29.6 billion",
  "period": "Q3 Fiscal 2024",
  "prior_year": "$28.3 billion",
  "source_document": "Apple 2024 10-Q",
  "direct_quote": "Operating income was $29.6 billion for the third fiscal quarter of 2024",
  "confidence": "HIGH"
}}
</assistant_response>
</example>

If the figure cannot be found, return: {{"answer": null, "source_document": null, "direct_quote": null, "confidence": "LOW"}}
Return ONLY the JSON object. No explanation, no preamble."""),
    ("human", "Context:\n{context}\n\nQuery: {query}\n\nReturn ONLY valid JSON.")
])

# --- Strategy 3: Chain-of-Thought Comparison ---
COT_COMPARISON_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert equity researcher performing a structured comparative analysis.

You MUST respond using this exact three-step format:

STEP 1 — EVIDENCE EXTRACTION:
List every relevant fact from the context for each company or time period. After each fact, write its source inline: [Source: Company Year 10-K/10-Q]

STEP 2 — SIDE-BY-SIDE COMPARISON:
Organize the extracted evidence into a clear comparison. Use a table or parallel bullet points.

STEP 3 — SYNTHESIS & CONCLUSION:
State your conclusion based solely on the evidence above. Cite each source inline.

RULES:
- Never invent data. If a document does not contain relevant information for one side of the comparison, write: "No information found in [Source] for this metric."
- Every claim in Step 3 must have an inline citation."""),
    ("human", "Context:\n{context}\n\nQuery: {query}")
])


def get_embedding(text: str) -> list:
    """Generates a vector embedding using AWS Titan via Boto3."""
    payload = {"inputText": text, "dimensions": 1024, "normalize": True}
    body_bytes = json.dumps(payload).encode('utf-8')

    response = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=body_bytes
    )
    return json.loads(response.get('body').read())['embedding']


def retrieve_chunks(user_query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieves the top_k most relevant chunks from Qdrant for a given query.
    Returns a list of chunk payloads (text, company, year, chunk_id) for evaluation use.

    Used by Pratheek's Context Precision evaluation script to inspect which chunks
    Qdrant returned before the LLM processes them.
    """
    query_vector = get_embedding(user_query)

    search_results = qdrant.query_points(
        collection_name="financial_reports",
        query=query_vector,
        limit=top_k
    ).points

    return [result.payload for result in search_results]


def ask_financial_system(user_query: str, top_k: int = 5, strategy: str = "standard") -> str:
    """
    Core RAG wrapper. Retrieves context from Qdrant and generates a cited answer via LangChain.

    Args:
        user_query: The natural language question to answer.
        top_k: Number of context chunks to retrieve from Qdrant.
        strategy: Prompting strategy to use.
                  "standard"   - General Q&A with strict inline citations.
                  "extraction" - Returns a structured JSON object with the exact figure and source.
                  "comparison" - Chain-of-Thought reasoning for cross-company or cross-period analysis.
    """
    print(f"[Strategy: {strategy}] Embedding query and fetching from Qdrant...")

    # 1. Retrieve the most relevant chunks
    chunks = retrieve_chunks(user_query, top_k)

    # 2. Assemble the context blocks with metadata citations
    context_blocks = []
    for chunk in chunks:
        text = chunk.get("text", "")
        company = chunk.get("company", "Unknown")
        year = chunk.get("year", "Unknown")
        context_blocks.append(f"[Source: {company} {year} 10-K]\n{text}")

    context_string = "\n\n---\n\n".join(context_blocks)

    # 3. Select prompt strategy
    if strategy == "extraction":
        prompt = EXTRACTION_PROMPT
    elif strategy == "comparison":
        prompt = COT_COMPARISON_PROMPT
    else:
        prompt = STANDARD_PROMPT

    # 4. Chain execution
    chain = prompt | llm

    print("Generating response via Bedrock Converse...")
    try:
        response = chain.invoke({
            "context": context_string,
            "query": user_query
        })
        return response.content
    except Exception as e:
        return f"API Error during generation: {e}"


if __name__ == "__main__":
    # --- Test 1: Standard — general risk question with inline citations ---
    print("\n=== STANDARD: General Q&A ===")
    print(ask_financial_system(
        "What are the primary macroeconomic risks mentioned?",
        strategy="standard"
    ))

    # --- Test 2: Extraction — forces a JSON response with exact figure + source ---
    print("\n=== EXTRACTION: Structured JSON Output ===")
    print(ask_financial_system(
        "What was the total revenue reported?",
        strategy="extraction"
    ))

    # --- Test 3: Comparison — triggers Chain-of-Thought Step 1 / Step 2 / Step 3 format ---
    print("\n=== COMPARISON: Chain-of-Thought Analysis ===")
    print(ask_financial_system(
        "Compare the risk factors described across different sections of this filing.",
        strategy="comparison"
    ))
