from typing import Dict, List, Optional
import pandas as pd

from src.config import VECTOR_STORE_PATH, METADATA_DIR
from src.retrieval.vector_store import LocalTfidfVectorStore, SearchResult


class TravelRetrievers:
    def __init__(self):
        self.store = LocalTfidfVectorStore.load(VECTOR_STORE_PATH)
        self.metadata_df = pd.read_csv(METADATA_DIR / "travel_memory_metadata.csv")

    def text_retriever(self, query: str, top_k: int = 4) -> List[SearchResult]:
        return self.store.query(query, top_k=top_k, modality="text")

    def image_caption_retriever(self, query: str, top_k: int = 4) -> List[SearchResult]:
        return self.store.query(query, top_k=top_k, modality="image_caption")

    def memory_metadata_retriever(self, query: str, top_k: int = 4) -> List[SearchResult]:
        return self.store.query(query, top_k=top_k, modality="metadata_memory")

    def hybrid_retriever(self, query: str, top_k: int = 6) -> List[SearchResult]:
        candidates = []
        candidates.extend(self.store.query(query, top_k=top_k, modality="text"))
        candidates.extend(self.store.query(query, top_k=top_k, modality="image_caption"))
        candidates.extend(self.store.query(query, top_k=top_k, modality="metadata_memory"))

        by_id = {}
        for item in candidates:
            if item.doc_id not in by_id or item.score > by_id[item.doc_id].score:
                by_id[item.doc_id] = item

        results = list(by_id.values())
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def metadata_filter(
        self,
        city: Optional[str] = None,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict]:
        df = self.metadata_df.copy()

        if city:
            df = df[df["city"].astype(str).str.contains(city, case=False, na=False)]
        if category:
            df = df[df["category"].astype(str).str.contains(category, case=False, na=False)]
        if tag:
            df = df[df["tags"].astype(str).str.contains(tag, case=False, na=False)]

        return df.head(top_k).to_dict(orient="records")


def format_results(results: List[SearchResult]) -> str:
    chunks = []
    for i, r in enumerate(results, start=1):
        chunks.append(
            f"[Evidence {i}] doc_id={r.doc_id}; modality={r.modality}; "
            f"city={r.city}; place={r.place}; score={r.score:.3f}; source={r.source}\n"
            f"{r.content}"
        )
    return "\n\n".join(chunks)
