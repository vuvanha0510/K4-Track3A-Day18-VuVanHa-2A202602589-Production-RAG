from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        # underthesea chưa tải model / lỗi mạng → fallback về text gốc.
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [segment_vietnamese(c["text"]).split() for c in chunks]
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []
        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for i in top_indices:
            if scores[i] > 0:
                doc = self.documents[i]
                results.append(SearchResult(
                    text=doc["text"],
                    score=float(scores[i]),
                    metadata=doc.get("metadata", {}),
                    method="bm25"
                ))
        if not results and top_indices:
            # Fallback if query terms have 0 scores but results are requested
            for i in top_indices[:top_k]:
                doc = self.documents[i]
                results.append(SearchResult(
                    text=doc["text"],
                    score=float(scores[i]),
                    metadata=doc.get("metadata", {}),
                    method="bm25"
                ))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        if not chunks:
            return
        from qdrant_client.models import Distance, PointStruct, VectorParams
        try:
            self.client.recreate_collection(collection, vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
        except Exception:
            try:
                self.client.delete_collection(collection)
            except Exception as del_err:
                # Collection chưa tồn tại → bỏ qua, create phía dưới sẽ tạo mới.
                print(f"  ℹ️  delete_collection({collection}) skipped: {del_err}")
            self.client.create_collection(collection, vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))

        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=False)
        points = [
            PointStruct(
                id=i,
                vector=v.tolist(),
                payload={**c.get("metadata", {}), "text": c["text"]}
            )
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection, points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        try:
            if not self.client.collection_exists(collection):
                return []
        except Exception as ex:
            # Client cũ hoặc collection check không hỗ trợ → bỏ qua,
            # để query_points() bên dưới quyết định.
            print(f"  ℹ️  collection_exists({collection}) check skipped: {ex}")
        query_vector = self._get_encoder().encode(query).tolist()
        try:
            response = self.client.query_points(collection, query=query_vector, limit=top_k)
            points = response.points
        except Exception:
            try:
                # Fallback for older qdrant_client versions if search() is available
                points = self.client.search(collection_name=collection, query_vector=query_vector, limit=top_k)
            except Exception:
                return []

        results = []
        for pt in points:
            payload = getattr(pt, "payload", {}) or {}
            score = getattr(pt, "score", 0.0)
            text = payload.get("text", "")
            meta = {k: v for k, v in payload.items() if k != "text"}
            results.append(SearchResult(text=text, score=float(score), metadata=meta, method="dense"))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid"
        )
        for item in sorted_items
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original: Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
