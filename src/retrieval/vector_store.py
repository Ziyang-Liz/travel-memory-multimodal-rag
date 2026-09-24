import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DOCUMENTS_JSONL, VECTOR_STORE_PATH


@dataclass
class SearchResult:
    doc_id: str
    score: float
    content: str
    modality: str
    source: str
    city: str
    place: str
    category: str
    metadata: Dict[str, Any]


class LocalTfidfVectorStore:

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.documents: List[Dict[str, Any]] = []

    def build(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        corpus = [doc["content"] for doc in documents]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=12000
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    def save(self, path=VECTOR_STORE_PATH):
        joblib.dump({
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
            "documents": self.documents,
        }, path)

    @classmethod
    def load(cls, path=VECTOR_STORE_PATH):
        data = joblib.load(path)
        store = cls()
        store.vectorizer = data["vectorizer"]
        store.matrix = data["matrix"]
        store.documents = data["documents"]
        return store

    def query(
        self,
        query: str,
        top_k: int = 5,
        modality: Optional[str] = None,
        city: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[SearchResult]:
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("Vector store is not built or loaded.")

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()

        results = []
        for idx, score in enumerate(scores):
            doc = self.documents[idx]
            if modality and doc.get("modality") != modality:
                continue
            if city and city.lower() not in str(doc.get("city", "")).lower():
                continue
            if category and category.lower() not in str(doc.get("category", "")).lower():
                continue
            results.append(SearchResult(
                doc_id=doc["doc_id"],
                score=float(score),
                content=doc["content"],
                modality=doc.get("modality", ""),
                source=doc.get("source", ""),
                city=doc.get("city", ""),
                place=doc.get("place", ""),
                category=doc.get("category", ""),
                metadata=doc.get("metadata", {}),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


def load_documents_jsonl(path=DOCUMENTS_JSONL):
    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))
    return documents


def build_and_save_vector_store():
    documents = load_documents_jsonl()
    store = LocalTfidfVectorStore()
    store.build(documents)
    store.save()
    return store


if __name__ == "__main__":
    store = build_and_save_vector_store()
    print(f"Saved vector store to {VECTOR_STORE_PATH}")
