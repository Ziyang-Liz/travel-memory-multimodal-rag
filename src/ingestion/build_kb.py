import json
from pathlib import Path
import pandas as pd

from src.config import TEXT_DIR, METADATA_DIR, STORAGE_DIR, DOCUMENTS_JSONL


def _safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def load_text_documents():
    docs = []
    for path in sorted(TEXT_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append({
            "doc_id": f"text_{path.stem}",
            "source": str(path),
            "modality": "text",
            "city": "",
            "place": "",
            "category": "notes",
            "content": text,
            "metadata": {"file_name": path.name},
        })
    return docs


def load_image_caption_documents():
    csv_path = METADATA_DIR / "image_captions.csv"
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        content = (
            f"Image caption: {_safe_str(row.get('caption'))}\n"
            f"Visual tags: {_safe_str(row.get('visual_tags'))}\n"
            f"Emotion tags: {_safe_str(row.get('emotion_tags'))}\n"
            f"Related memory: {_safe_str(row.get('related_memory'))}\n"
            f"City: {_safe_str(row.get('city'))}\n"
            f"Place: {_safe_str(row.get('place'))}\n"
            f"Image path: {_safe_str(row.get('image_path'))}"
        )
        docs.append({
            "doc_id": _safe_str(row["image_id"]),
            "source": _safe_str(row["image_path"]),
            "modality": "image_caption",
            "city": _safe_str(row.get("city")),
            "place": _safe_str(row.get("place")),
            "category": "image",
            "content": content,
            "metadata": row.to_dict(),
        })
    return docs


def load_memory_metadata_documents():
    csv_path = METADATA_DIR / "travel_memory_metadata.csv"
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        content = (
            f"Memory note: {_safe_str(row.get('memory_note'))}\n"
            f"Caption: {_safe_str(row.get('caption'))}\n"
            f"Tags: {_safe_str(row.get('tags'))}\n"
            f"Emotion: {_safe_str(row.get('emotion_tags'))}\n"
            f"City: {_safe_str(row.get('city'))}\n"
            f"Place: {_safe_str(row.get('place'))}\n"
            f"Category: {_safe_str(row.get('category'))}\n"
            f"Cost: {_safe_str(row.get('cost'))}\n"
            f"Weather: {_safe_str(row.get('weather'))}\n"
            f"Image path: {_safe_str(row.get('image_path'))}"
        )
        docs.append({
            "doc_id": _safe_str(row["memory_id"]),
            "source": _safe_str(row.get("image_path")),
            "modality": "metadata_memory",
            "city": _safe_str(row.get("city")),
            "place": _safe_str(row.get("place")),
            "category": _safe_str(row.get("category")),
            "content": content,
            "metadata": row.to_dict(),
        })
    return docs


def build_documents_jsonl():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    docs = []
    docs.extend(load_text_documents())
    docs.extend(load_image_caption_documents())
    docs.extend(load_memory_metadata_documents())

    with DOCUMENTS_JSONL.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return docs


if __name__ == "__main__":
    docs = build_documents_jsonl()
    print(f"Built {len(docs)} documents at {DOCUMENTS_JSONL}")
