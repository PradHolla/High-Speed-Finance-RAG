import argparse
import json
import os
from datetime import datetime

import boto3
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate

from retrieval_engine import ask_financial_system

load_dotenv()

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
JUDGE_MODEL_ID = os.getenv("JUDGE_MODEL_ID", "zai.glm-5")
DEFAULT_COLLECTION = "financial_reports"

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=REGION)
judge_llm = ChatBedrockConverse(
    client=bedrock_client,
    model=JUDGE_MODEL_ID,
    temperature=0.0,
    max_tokens=800,
)

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert financial analyst and strict evaluation judge for a Retrieval-Augmented Generation system.
Your job is to compare a proposed answer against a verified reference answer, judge the factual accuracy, and identify any unsupported or hallucinated claims.

Return ONLY valid JSON with the following keys:
{
  "score": <integer 0-5>,
  "accuracy": "HIGH | MEDIUM | LOW",
  "hallucination": "NONE | MINOR | MAJOR",
  "missing_information": "<brief summary of missing facts if any>",
  "incorrect_claims": "<brief summary of incorrect claims if any>",
  "notes": "<short reasoning summary>"
}

Evaluation definitions:
- 5: Fully accurate, all key facts present, no unsupported claims.
- 4: Mostly accurate, minor wording or detail differences, no substantive hallucinations.
- 3: Partially correct, some important facts are missing or slightly inaccurate.
- 2: Significant factual errors or unsupported content, but some correct detail remains.
- 1: Largely incorrect or hallucinated; not faithful to the reference.
- 0: Incorrect answer or answer not based on the provided documents.

Do not add any extra text, explanations, or markdown outside the JSON object."""
    ),
    (
        "human",
        """Question: {question}

Reference Answer: {reference_answer}

System Answer: {system_answer}

Evaluate this response strictly against the reference answer. If the system answer omits key factual items from the reference, mark them in missing_information. If the system answer contains claims not supported by the reference, mark them in incorrect_claims and hallucination.

Return only the JSON object."""
    ),
])


def load_golden_dataset(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def simple_exact_match(reference_answer: str, system_answer: str) -> bool:
    ref = " ".join(reference_answer.lower().split())
    sys = " ".join(system_answer.lower().split())
    return ref in sys or sys in ref


def build_judge_input(question: str, reference_answer: str, system_answer: str) -> dict:
    return {
        "question": question,
        "reference_answer": reference_answer,
        "system_answer": system_answer,
    }


def judge_answer(question: str, reference_answer: str, system_answer: str) -> dict:
    chain = JUDGE_PROMPT | judge_llm
    payload = build_judge_input(question, reference_answer, system_answer)
    response = chain.invoke(payload)

    raw_output = response.content.strip()
    try:
        judged = json.loads(raw_output)
    except json.JSONDecodeError:
        judged = {
            "score": 0,
            "accuracy": "LOW",
            "hallucination": "MAJOR",
            "missing_information": "Failed to parse judge output.",
            "incorrect_claims": "Judge output not valid JSON.",
            "notes": raw_output.replace("\n", " ")[:800],
        }
    return judged


def evaluate_dataset(dataset_path: str, top_k: int, strategy: str, output_path: str) -> dict:
    dataset = load_golden_dataset(dataset_path)
    results = []
    total_score = 0

    for item in dataset.get("questions", []):
        question_id = item.get("id")
        question = item.get("question")
        reference_answer = item.get("answer")

        print(f"Evaluating {question_id}: {question}")
        system_answer = ask_financial_system(question, top_k=top_k, strategy=strategy)

        judge_result = judge_answer(question, reference_answer, system_answer)
        baseline_match = simple_exact_match(reference_answer, system_answer)

        record = {
            "id": question_id,
            "question": question,
            "reference_answer": reference_answer,
            "system_answer": system_answer,
            "judge_result": judge_result,
            "baseline_exact_match": baseline_match,
        }
        results.append(record)
        total_score += int(judge_result.get("score", 0))

    average_score = round(total_score / max(len(results), 1), 2)
    summary = {
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "dataset_path": os.path.abspath(dataset_path),
        "model": JUDGE_MODEL_ID,
        "strategy": strategy,
        "top_k": top_k,
        "questions_evaluated": len(results),
        "average_score": average_score,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge evaluation for the High-Speed Finance RAG system.")
    parser.add_argument("--dataset", default="golden_dataset.json", help="Path to the golden dataset JSON file.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of Qdrant chunks to retrieve for each query.")
    parser.add_argument("--strategy", choices=["standard", "extraction", "comparison"], default="standard", help="Generation strategy for the financial system answers.")
    parser.add_argument("--output", default="generation_evaluation_results.json", help="Output file path for the judge evaluation results.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluation = evaluate_dataset(args.dataset, args.top_k, args.strategy, args.output)
    print(f"Evaluation complete. Results written to {args.output}")
    print(f"Average judge score: {evaluation['average_score']}")
