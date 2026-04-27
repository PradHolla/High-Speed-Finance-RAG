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

def ask_financial_system(user_query: str, top_k: int = 5) -> str:
    """
    The core wrapper function. 
    Retrieves context from Qdrant and generates an answer using LangChain Bedrock Converse.
    """
    print("Embedding query and fetching from Qdrant...")
    query_vector = get_embedding(user_query)
    
    # 1. Retrieve the most relevant chunks
    search_results = qdrant.query_points(
        collection_name="financial_reports",
        query=query_vector,
        limit=top_k
    ).points
    
    # 2. Assemble the context blocks with metadata citations
    context_blocks = []
    for result in search_results:
        text = result.payload.get("text", "")
        company = result.payload.get("company", "Unknown")
        year = result.payload.get("year", "Unknown")
        context_blocks.append(f"[Source: {company} {year} 10-K]\n{text}")
        
    context_string = "\n\n---\n\n".join(context_blocks)
    
    # 3. LangChain Prompting
    # Your Prompt Engineering Lead will spend most of their time tweaking this exact template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert equity researcher. Answer questions based ONLY on the provided context. Always explicitly cite the Source Document inline."),
        ("human", "Context:\n{context}\n\nQuery: {query}")
    ])
    
    # 4. Chain execution
    chain = prompt | llm
    
    print("Generating reasoning via Bedrock Converse...")
    try:
        response = chain.invoke({
            "context": context_string,
            "query": user_query
        })
        return response.content
    except Exception as e:
        return f"API Error during generation: {e}"

if __name__ == "__main__":
    # Test execution
    answer = ask_financial_system("What are the primary macroeconomic risks mentioned?")
    
    print("\n=== SYSTEM ANSWER ===")
    print(answer)