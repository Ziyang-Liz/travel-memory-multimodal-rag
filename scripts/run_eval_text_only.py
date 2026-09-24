import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from dotenv import load_dotenv

from src.config import EVAL_DIR
from src.retrieval.retrievers import TravelRetrievers
from src.agent.answer import generate_answer


def main():
    
    load_dotenv()

    test_path = EVAL_DIR / "test_questions.csv"
    df = pd.read_csv(test_path)

    retrievers = TravelRetrievers()
    rows = []

    for _, row in df.iterrows():
        question = row["question"]
        expected_image_id = str(row["expected_image_id"])

        start = time.time()

        evidence = retrievers.text_retriever(question, top_k=8)

        answer = generate_answer(
            question=question,
            evidence=evidence,
            route="text_only",
            history=""
        )

        latency = time.time() - start
        evidence_ids = [item.doc_id for item in evidence]

        rows.append({
            "question_id": row["question_id"],
            "family": row["family"],
            "question": question,
            "system_variant": "text_only_rag",
            "expected_image_id": expected_image_id,
            "retrieved_evidence_ids": "|".join(evidence_ids),
            "retrieval_hit": int(expected_image_id in evidence_ids),
            "latency_seconds": round(latency, 3),
            "answer": answer,
            "manual_answer_score": ""
        })

    result_df = pd.DataFrame(rows)

    out = EVAL_DIR / "evaluation_results_text_only.csv"
    result_df.to_csv(out, index=False, encoding="utf-8-sig")

    summary = result_df.groupby("family").agg(
        questions=("question_id", "count"),
        retrieval_accuracy=("retrieval_hit", "mean"),
        avg_latency=("latency_seconds", "mean")
    ).reset_index()

    summary_out = EVAL_DIR / "evaluation_summary_text_only.csv"
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

    print("Text-only evaluation complete.")
    print(summary)
    print(f"\nSaved detailed results to: {out}")
    print(f"Saved summary to: {summary_out}")


if __name__ == "__main__":
    main()