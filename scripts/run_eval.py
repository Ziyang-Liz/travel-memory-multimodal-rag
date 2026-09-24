import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import time
import pandas as pd

from src.agent.graph import ask_agent
from src.config import EVAL_DIR
import time
import pandas as pd

from src.agent.graph import ask_agent
from src.config import EVAL_DIR


def contains_expected_image(answer: str, expected_image_id: str) -> bool:
    if not expected_image_id:
        return False
    return expected_image_id.lower() in answer.lower()


def main():
    test_path = EVAL_DIR / "test_questions.csv"
    df = pd.read_csv(test_path)

    rows = []
    history = ""

    for _, row in df.iterrows():
        question = row["question"]
        expected_image_id = str(row["expected_image_id"])

        start = time.time()
        result = ask_agent(question, history="")
        latency = time.time() - start

        evidence_ids = [e.doc_id for e in result["evidence"]]
        retrieved_expected = expected_image_id in evidence_ids
        answer_mentions_expected = contains_expected_image(result["answer"], expected_image_id)

        rows.append({
            "question_id": row["question_id"],
            "family": row["family"],
            "question": question,
            "route": result["route"],
            "expected_image_id": expected_image_id,
            "retrieved_evidence_ids": "|".join(evidence_ids),
            "retrieval_hit": int(retrieved_expected),
            "answer_mentions_expected_id": int(answer_mentions_expected),
            "latency_seconds": round(latency, 3),
            "answer": result["answer"],
            "manual_answer_score": ""
        })

        history += f"\nUser: {question}\nAssistant: {result['answer']}\n"

    out = EVAL_DIR / "evaluation_results.csv"
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out, index=False, encoding="utf-8-sig")

    summary = result_df.groupby("family").agg(
        questions=("question_id", "count"),
        retrieval_accuracy=("retrieval_hit", "mean"),
        avg_latency=("latency_seconds", "mean"),
    ).reset_index()

    summary_out = EVAL_DIR / "evaluation_summary.csv"
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

    print("Evaluation complete.")
    print(summary)
    print(f"\nSaved detailed results to: {out}")
    print(f"Saved summary to: {summary_out}")


if __name__ == "__main__":
    main()
