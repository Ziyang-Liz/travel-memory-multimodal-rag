import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.config import EVAL_DIR


def ask_plain_llm(client, question: str) -> str:
    prompt = f"""
You are answering a question without access to the user's private travel memory knowledge base.

Question:
{question}

If the question requires private travel details that are not provided, clearly say that the information is not available.
Do not invent personal travel facts.
"""

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": "You are a careful assistant. Do not invent private personal memories."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Please add it to your .env file before running this baseline."
        )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    test_path = EVAL_DIR / "test_questions.csv"
    df = pd.read_csv(test_path)

    rows = []

    for _, row in df.iterrows():
        question = row["question"]

        start = time.time()
        answer = ask_plain_llm(client, question)
        latency = time.time() - start

        rows.append({
            "question_id": row["question_id"],
            "family": row["family"],
            "question": question,
            "system_variant": "plain_llm",
            "expected_image_id": row["expected_image_id"],
            "retrieved_evidence_ids": "",
            "retrieval_hit": "N/A",
            "latency_seconds": round(latency, 3),
            "answer": answer,
            "manual_answer_score": ""
        })

    result_df = pd.DataFrame(rows)

    out = EVAL_DIR / "evaluation_results_plain_llm.csv"
    result_df.to_csv(out, index=False, encoding="utf-8-sig")

    summary = result_df.groupby("family").agg(
        questions=("question_id", "count"),
        avg_latency=("latency_seconds", "mean")
    ).reset_index()

    summary_out = EVAL_DIR / "evaluation_summary_plain_llm.csv"
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

    print("Plain LLM evaluation complete.")
    print(summary)
    print(f"\nSaved detailed results to: {out}")
    print(f"Saved summary to: {summary_out}")


if __name__ == "__main__":
    main()