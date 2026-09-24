import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.ingestion.build_kb import build_documents_jsonl
from src.retrieval.vector_store import build_and_save_vector_store

if __name__ == "__main__":
    docs = build_documents_jsonl()
    print(f"Step 1 complete: built {len(docs)} documents.")
    build_and_save_vector_store()
    print("Step 2 complete: vector index saved in storage/.")