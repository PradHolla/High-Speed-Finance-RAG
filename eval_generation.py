import argparse
import json
import os
from pathlib import Path
from typing import Any

import boto3
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate

from retrieval_engine import ask_financial_system, retrieve_chunks


REQUIRED_KEYS = {
    "factual_accuracy",
    "citation_quality",
    "hallucination",
    "missing_facts",
    "unsupported_claims",
    "final_score",
}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_json_object(raw_text: str) -> dict:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[idx:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError("Judge output does not contain a valid JSON object.")


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return int(stripped)
    return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return True
        if lowered in {"false", "no"}:
            return False
    return None


def _as_string_list(value: Any) -> list[str] | None:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return None


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def validate_and_coerce_judge_output(candidate: dict) -> dict:
    if not isinstance(candidate, dict):
        raise ValueError("Judge output must be a JSON object.")
    if set(candidate.keys()) != REQUIRED_KEYS:
        raise ValueError(f"Judge output keys must match exactly: {sorted(REQUIRED_KEYS)}")

    factual_accuracy = _as_int(candidate.get("factual_accuracy"))
    citation_quality = _as_int(candidate.get("citation_quality"))
    final_score = _as_int(candidate.get("final_score"))
    hallucination = _as_bool(candidate.get("hallucination"))
    missing_facts = _as_string_list(candidate.get("missing_facts"))
    unsupported_claims = _as_string_list(candidate.get("unsupported_claims"))

    if factual_accuracy is None or citation_quality is None or final_score is None:
        raise ValueError("Judge numeric fields must be integers or integer-like strings.")
    if hallucination is None:
        raise ValueError("Judge hallucination field must be boolean or boolean-like string.")
    if missing_facts is None or unsupported_claims is None:
        raise ValueError("Judge list fields must be arrays.")

    normalized = {
        "factual_accuracy": _clamp(factual_accuracy, 1, 5),
        "citation_quality": _clamp(citation_quality, 1, 5),
        "hallucination": hallucination,
        "missing_facts": missing_facts,
        "unsupported_claims": unsupported_claims,
        "final_score": _clamp(final_score, 0, 100),
    }
    return normalized


def parse_judge_json(raw_text: str) -> dict:
    parsed = _extract_json_object(raw_text)
    return validate_and_coerce_judge_output(parsed)


def build_judge_chain(model_id: str, temperature: float, max_tokens: int):
    client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    llm = ChatBedrockConverse(client=client, model=model_id, temperature=temperature, max_tokens=max_tokens)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are grading a finance RAG answer for factual grounding.
Return ONLY valid JSON in this exact schema:
{{
  "factual_accuracy": 1,
  "citation_quality": 1,
  "hallucination": false,
  "missing_facts": [],
  "unsupported_claims": [],
  "final_score": 0
}}

Rules:
- factual_accuracy is 1-5.
- citation_quality is 1-5.
- final_score is 0-100.
- hallucination must be true if claims are unsupported by provided context.
- missing_facts contains key expected facts absent from answer.
- unsupported_claims contains answer claims not grounded in context.
- Do not include markdown fences or any prose.
"""),
        ("human", """Question:
{question}

Expected answer:
{expected_answer}

Required facts:
{required_facts}

Expected source pages:
{source_pages}

Retrieved context:
{retrieved_context}

Model answer:
{model_answer}
"""),
    ])
    return prompt | llm


def build_repair_chain(model_id: str, temperature: float, max_tokens: int):
    client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    llm = ChatBedrockConverse(client=client, model=model_id, temperature=temperature, max_tokens=max_tokens)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You fix malformed judge output.
Return ONLY valid JSON with exactly these keys:
{{
  "factual_accuracy": 1,
  "citation_quality": 1,
  "hallucination": false,
  "missing_facts": [],
  "unsupported_claims": [],
  "final_score": 0
}}
No markdown fences. No extra keys."""),
        ("human", """Malformed judge output:
{raw_output}

Repair it into valid JSON with the exact schema only."""),
    ])
    return prompt | llm


def _fallback_score() -> dict:
    return {
        "factual_accuracy": 1,
        "citation_quality": 1,
        "hallucination": True,
        "missing_facts": ["Judge parsing failed"],
        "unsupported_claims": [],
        "final_score": 0,
    }


def _judge_with_repair(judge_chain, repair_chain, payload: dict) -> tuple[dict, str, str | None, str]:
    judged = judge_chain.invoke(payload)
    raw = judged.content if isinstance(judged.content, str) else str(judged.content)
    try:
        return parse_judge_json(raw), "ok", None, raw
    except Exception as primary_exc:
        repaired = repair_chain.invoke({"raw_output": raw})
        repaired_raw = repaired.content if isinstance(repaired.content, str) else str(repaired.content)
        try:
            return parse_judge_json(repaired_raw), "repaired", None, repaired_raw
        except Exception as repair_exc:
            error = f"Primary parse error: {primary_exc}; Repair parse error: {repair_exc}"
            return _fallback_score(), "failed", error, repaired_raw


def evaluate(
    golden_path: str,
    top_k: int,
    output_path: str,
    strategy: str,
    judge_model: str,
    judge_temperature: float,
    judge_max_tokens: int,
) -> dict:
    dataset = load_jsonl(golden_path)
    judge_chain = build_judge_chain(judge_model, judge_temperature, judge_max_tokens)
    repair_chain = build_repair_chain(judge_model, judge_temperature, judge_max_tokens)
    rows = []

    for item in dataset:
        question = item["question"]
        filters = {
            "company": item.get("company"),
            "filing_year": item.get("filing_year"),
            "filing_type": item.get("filing_type"),
        }
        chunks = retrieve_chunks(question, top_k=top_k, filters=filters)
        retrieved_context = "\n\n".join(
            f"[rank={c['rank']} score={c['score']:.5f} pages={c.get('page_numbers', [])}] {c.get('text', '')}"
            for c in chunks
        )
        model_answer = ask_financial_system(
            user_query=question,
            top_k=top_k,
            strategy=strategy,
            filters=filters,
        )

        payload = {
            "question": question,
            "expected_answer": item.get("expected_answer", ""),
            "required_facts": json.dumps(item.get("required_facts", []), ensure_ascii=True),
            "source_pages": json.dumps(item.get("source_pdf_pages", []), ensure_ascii=True),
            "retrieved_context": retrieved_context,
            "model_answer": model_answer,
        }

        score, judge_status, judge_error, judge_raw_output = _judge_with_repair(
            judge_chain=judge_chain,
            repair_chain=repair_chain,
            payload=payload,
        )

        rows.append(
            {
                "id": item["id"],
                "question": question,
                "strategy": strategy,
                "model_answer": model_answer,
                "judge_score": score,
                "judge_status": judge_status,
                "judge_error": judge_error,
                "judge_raw_output": judge_raw_output,
            }
        )

    final_scores = [r["judge_score"]["final_score"] for r in rows]
    factual_scores = [r["judge_score"]["factual_accuracy"] for r in rows]
    citation_scores = [r["judge_score"]["citation_quality"] for r in rows]
    hallucination_rate = sum(1 for r in rows if r["judge_score"]["hallucination"]) / len(rows) if rows else 0.0
    judge_success_rate = sum(1 for r in rows if r["judge_status"] == "ok") / len(rows) if rows else 0.0
    judge_repair_rate = sum(1 for r in rows if r["judge_status"] == "repaired") / len(rows) if rows else 0.0
    judge_failure_rate = sum(1 for r in rows if r["judge_status"] == "failed") / len(rows) if rows else 0.0

    summary = {
        "questions_evaluated": len(rows),
        "top_k": top_k,
        "strategy": strategy,
        "judge_model": judge_model,
        "judge_temperature": judge_temperature,
        "judge_max_tokens": judge_max_tokens,
        "avg_final_score": sum(final_scores) / len(final_scores) if final_scores else 0.0,
        "avg_factual_accuracy": sum(factual_scores) / len(factual_scores) if factual_scores else 0.0,
        "avg_citation_quality": sum(citation_scores) / len(citation_scores) if citation_scores else 0.0,
        "hallucination_rate": hallucination_rate,
        "judge_success_rate": judge_success_rate,
        "judge_repair_rate": judge_repair_rate,
        "judge_failure_rate": judge_failure_rate,
    }

    result = {"summary": summary, "rows": rows}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG generation quality with LLM-as-a-judge.")
    parser.add_argument("--golden-path", default="data/golden_dataset.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--strategy", default="standard")
    parser.add_argument("--output", default="artifacts/generation_eval.json")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", os.getenv("RAG_CHAT_MODEL", "zai.glm-5")))
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=1200)
    args = parser.parse_args()

    result = evaluate(
        golden_path=args.golden_path,
        top_k=args.top_k,
        output_path=args.output,
        strategy=args.strategy,
        judge_model=args.judge_model,
        judge_temperature=args.judge_temperature,
        judge_max_tokens=args.judge_max_tokens,
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote detailed generation results to {args.output}")


if __name__ == "__main__":
    main()
