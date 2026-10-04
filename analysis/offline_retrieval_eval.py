"""
Offline Retrieval Evaluation — đo Context Recall/Precision KHÔNG cần LLM.

Vì Gemini free tier chỉ 20 req/ngày nên RAGAS (gọi LLM judge cho mỗi câu)
không chạy được. Script này đo chất lượng retrieval bằng token overlap
giữa context trả về và ground_truth — chạy offline, miễn phí, kết quả tái lập được.

Cách đo:
- context_recall   = tỉ lệ token của ground_truth xuất hiện trong context
- context_precision = tỉ lệ token của context có trong ground_truth
- hit              = ground_truth có mặt (>= 0.7 recall) trong top-k context nào

Usage: python analysis/offline_retrieval_eval.py
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch, BM25Search, DenseSearch
from src.m4_eval import load_test_set
from config import HYBRID_TOP_K, RERANK_TOP_K

STOPWORDS = {
    "và", "của", "là", "có", "được", "cho", "với", "những", "các", "một", "này",
    "đó", "khi", "nếu", "để", "từ", "theo", "về", "trong", "ngoài", "ra", "vào",
    "như", "hoặc", "thì", "sẽ", "đã", "đang", "bị", "phải", "mỗi", "mà", "ở",
}


def tokenize(text: str) -> set:
    """Tách từ tiếng Việt, bỏ stopword, lowercase."""
    from src.m2_search import segment_vietnamese
    tokens = segment_vietnamese(text.lower()).split()
    return {t.strip(".,;:!?()\"'") for t in tokens if t.strip(".,;:!?()\"'") and t not in STOPWORDS}


def overlap_scores(contexts: list[str], ground_truth: str) -> tuple[float, float]:
    gt = tokenize(ground_truth)
    if not gt:
        return 0.0, 0.0
    ctx = set()
    for c in contexts:
        ctx |= tokenize(c)
    recall = len(gt & ctx) / len(gt)
    precision = len(gt & ctx) / len(ctx) if ctx else 0.0
    return recall, precision


def build_index(strategy: str = "hierarchical"):
    docs = load_documents()
    chunks = []
    for doc in docs:
        if strategy == "hierarchical":
            _, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
            for c in children:
                chunks.append({"text": c.text, "metadata": {**c.metadata, "parent_id": c.parent_id}})
        else:
            for c in chunk_hierarchical(doc["text"], metadata=doc["metadata"])[1]:
                chunks.append({"text": c.text, "metadata": c.metadata})
    return chunks


def evaluate(method: str, chunks: list[dict], top_k: int = 3) -> dict:
    test_set = load_test_set()

    if method == "hybrid":
        search = HybridSearch()
        search.index(chunks)

        def retrieve(q):
            return [r.text for r in search.search(q, top_k=HYBRID_TOP_K)[:top_k]]
    elif method == "bm25":
        search = BM25Search()
        search.index(chunks)

        def retrieve(q):
            return [r.text for r in search.search(q, top_k=top_k)]
    else:
        search = DenseSearch()
        search.index(chunks)

        def retrieve(q):
            return [r.text for r in search.search(q, top_k=top_k)]

    rows = []
    for item in test_set:
        contexts = retrieve(item["question"])
        recall, precision = overlap_scores(contexts, item["ground_truth"])
        rows.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "context_recall": round(recall, 4),
            "context_precision": round(precision, 4),
            "hit": recall >= 0.7,
        })

    n = len(rows)
    return {
        "method": method,
        "num_questions": n,
        "context_recall": round(sum(r["context_recall"] for r in rows) / n, 4),
        "context_precision": round(sum(r["context_precision"] for r in rows) / n, 4),
        "hit_rate": round(sum(r["hit"] for r in rows) / n, 4),
        "rows": rows,
    }


def main():
    print("=" * 70)
    print("OFFLINE RETRIEVAL EVAL (không cần LLM)")
    print("=" * 70)
    chunks = build_index()
    print(f"\nIndexed {len(chunks)} chunks\n")

    results = {}
    for method in ["bm25", "dense", "hybrid"]:
        print(f"--- {method.upper()} ---", flush=True)
        results[method] = evaluate(method, chunks, top_k=RERANK_TOP_K)
        r = results[method]
        print(f"  context_recall    = {r['context_recall']:.4f}")
        print(f"  context_precision = {r['context_precision']:.4f}")
        print(f"  hit_rate (>=0.7)  = {r['hit_rate']:.4f}\n")

    # In bảng so sánh + bottom-5
    print("=" * 70)
    print(f"{'Method':<12}{'Recall':>10}{'Precision':>12}{'Hit rate':>12}")
    print("-" * 70)
    for m, r in results.items():
        print(f"{m:<12}{r['context_recall']:>10.4f}{r['context_precision']:>12.4f}{r['hit_rate']:>12.4f}")

    print("\n" + "=" * 70)
    print("BOTTOM-5 WORST QUESTIONS (theo hybrid context_recall)")
    print("=" * 70)
    worst = sorted(results["hybrid"]["rows"], key=lambda r: r["context_recall"])[:5]
    for i, r in enumerate(worst, 1):
        print(f"\n#{i} recall={r['context_recall']:.4f} precision={r['context_precision']:.4f}")
        print(f"  Q: {r['question']}")
        print(f"  GT: {r['ground_truth'][:110]}")

    # Lưu kết quả để dùng cho failure_analysis.md
    import json
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "offline_retrieval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()