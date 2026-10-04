from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
        model = self._load_model()
        pairs = [(query, doc["text"]) for doc in documents]
        scores = model.predict(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]
        scored = sorted(zip(scores, documents), key=lambda x: float(x[0]), reverse=True)

        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """
    Lightweight ONNX alternative (<5ms vs ~150ms của bge-reranker-v2-m3).

    Dùng khi cần latency thấp (production, real-time) và chấp nhận chất lượng
    thấp hơn một chút so với cross-encoder 2.2B tham số.
    Model: flashrank ms-marco-MultiBERT-L-12 (đã nằm trong requirements.txt).
    """
    MODEL_NAME = "ms-marco-MultiBERT-L-12"

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from flashrank import Ranker
            self._model = Ranker(model_name=self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        try:
            from flashrank import RerankRequest
            model = self._load_model()
            passages = [{"id": i, "text": d["text"]} for i, d in enumerate(documents)]
            ranked = model.rerank(RerankRequest(query=query, passages=passages))
        except Exception as e:
            print(f"  ⚠️  Flashrank failed ({e}) — giữ nguyên thứ tự gốc.")
            ranked = [{"id": i, "score": documents[i].get("score", 0.0)}
                      for i in range(len(documents))]

        results = []
        for rank, item in enumerate(ranked[:top_k]):
            idx = item["id"] if isinstance(item.get("id"), int) else rank
            doc = documents[idx]
            results.append(RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(item.get("score", 0.0)),
                metadata=doc.get("metadata", {}),
                rank=rank,
            ))
        return results


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
