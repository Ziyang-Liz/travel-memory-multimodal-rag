from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TEXT_DIR = DATA_DIR / "text"
METADATA_DIR = DATA_DIR / "metadata"
EVAL_DIR = DATA_DIR / "evaluation"
STORAGE_DIR = PROJECT_ROOT / "storage"

DOCUMENTS_JSONL = STORAGE_DIR / "documents.jsonl"
VECTOR_STORE_PATH = STORAGE_DIR / "tfidf_vector_store.joblib"
