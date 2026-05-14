import argparse
import json
from pathlib import Path

from retrieval_engine import retrieve_chunks


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chunk_is_relevant(chunk: dict, expected_pages: list[int], keywords: list[str]) -> bool:
    pages = set(chunk.get("page_numbers") or [])
    expected = set(expected_pages or [])
    if pages and expected and pages.intersection(expected):
        return True

    text = (chunk.get("text") or "").lower()
    return any(keyword.lower() in text for keyword in keywords or [])


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(golden_path: str, top_k: int, output_path: str) -> dict:
    dataset = load_jsonl(golden_path)
    rows = []

    recall_hits = []
    precision_scores = []
    reciprocal_ranks = []

    for item in dataset:
        question = item["question"]
        expected_pages = item.get("source_pdf_pages", [])
        keywords = item.get("retrieval_keywords", [])

        filters = {
            "company": item.get("company"),
            "filing_year": item.get("filing_year"),
            "filing_type": item.get("filing_type"),
        }
        chunks = retrieve_chunks(question, top_k=top_k, filters=filters)

        relevance = [chunk_is_relevant(c, expected_pages, keywords) for c in chunks]
        relevant_count = sum(1 for x in relevance if x)
        recall_hit = relevant_count > 0
        precision = relevant_count / top_k if top_k else 0.0

        rr = 0.0
        for idx, rel in enumerate(relevance, start=1):
            if rel:
                rr = 1.0 / idx
                break

        recall_hits.append(1.0 if recall_hit else 0.0)
        precision_scores.append(precision)
        reciprocal_ranks.append(rr)

        rows.append(
            {
                "id": item["id"],
                "question": question,
                "top_k": top_k,
                "recall_hit": recall_hit,
                "context_precision": precision,
                "mrr_component": rr,
                "relevant_chunks": relevant_count,
                "retrieved_chunk_ids": [c.get("chunk_id") for c in chunks],
                "retrieved_pages": [c.get("page_numbers", []) for c in chunks],
            }
        )

    summary = {
        "questions_evaluated": len(dataset),
        "top_k": top_k,
        "recall_at_k": mean(recall_hits),
        "context_precision": mean(precision_scores),
        "mrr": mean(reciprocal_ranks),
    }

    result = {"summary": summary, "rows": rows}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on golden dataset.")
    parser.add_argument("--golden-path", default="data/golden_dataset.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="artifacts/retrieval_eval.json")
    args = parser.parse_args()

    result = evaluate(args.golden_path, args.top_k, args.output)
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote detailed retrieval results to {args.output}")


if __name__ == "__main__":
    main()
