from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class _ThrottledLLMWrapper:
    """
    Bọc RAGAS LLM để giới hạn tốc độ gọi (requests per minute).

    Gemini free tier: 15 req/phút/model. RAGAS chạy metric song song (dù
    max_workers=1 vẫn có các call lồng nhau qua asyncio) nên có thể vượt trần
    và nhận429. Wrapper này giữ khoảng cách tối thiểu giữa 2 lần gọi.

    RAGAS gọi `self.llm.agenerate()` / `generate()` (không phải invoke),
    nên phải throttle ở đúng 2 method này.
    """

    def __init__(self, inner, rpm: int = 9):
        self._inner = inner
        self._interval = 60.0 / max(1, rpm)
        self._last = 0.0
        import asyncio as _asyncio
        # Lock để các coroutine gọi song song vẫn được xếp hàng tuần tự.
        self._alock = _asyncio.Lock()

    def set_run_config(self, run_config):
        return self._inner.set_run_config(run_config)

    def _sync_wait(self):
        import time as _t
        gap = _t.monotonic() - self._last
        if gap < self._interval:
            _t.sleep(self._interval - gap)
        self._last = _t.monotonic()

    def generate(self, *args, **kwargs):
        self._sync_wait()
        return self._inner.generate(*args, **kwargs)

    async def agenerate(self, *args, **kwargs):
        # Phải dùng asyncio.sleep + Lock: time.sleep() sẽ block event loop
        # khiến các request đang chờ khác bị TimeoutError.
        import asyncio as _asyncio
        import time as _t
        async with self._alock:
            gap = _t.monotonic() - self._last
            if gap < self._interval:
                await _asyncio.sleep(self._interval - gap)
            self._last = _t.monotonic()
            return await self._inner.agenerate(*args, **kwargs)

    def __getattr__(self, name):
        # Chuyển tiếp mọi thuộc tính khác (is_finished, temperature, ...).
        return getattr(self._inner, name)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    import pandas as pd

    from config import (
        GEMINI_API_KEY,
        GEMINI_BASE_URL,
        OPENAI_API_KEY,
        RAGAS_CHAT_MODEL,
        RAGAS_EMBED_MODEL,
        RAGAS_RPM_LIMIT,
    )
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or OPENAI_API_KEY or GEMINI_API_KEY
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        if not api_key.startswith("sk-"):
            os.environ["OPENAI_BASE_URL"] = GEMINI_BASE_URL

    try:
        # Phải ép kiểu cột "contexts" là Sequence[string].
        # Nếu để datasets tự suy luận, list 1 phần tử sẽ bị gán kiểu Value
        # → ValueError: Dataset feature "contexts" should be of type Sequence[string].
        from datasets import Dataset, Features
        from datasets import Sequence as DatasetsSequence
        from datasets import Value as DatasetsValue
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        features = Features({
            "question": DatasetsValue("string"),
            "answer": DatasetsValue("string"),
            "contexts": DatasetsSequence(DatasetsValue("string")),
            "ground_truth": DatasetsValue("string"),
        })
        dataset = Dataset.from_dict({
            "question": list(questions),
            "answer": list(answers),
            "contexts": [list(c) if c is not None else [""] for c in contexts],
            "ground_truth": list(ground_truths),
        }, features=features)
        eval_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        # Configure model if gemini / custom base URL.
        # RAGAS yêu cầu object có set_run_config() → phải bọc qua
        # LangchainLLMWrapper / LangchainEmbeddingsWrapper, không gán ChatOpenAI thô.
        if api_key and not api_key.startswith("sk-"):
            try:
                from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                from ragas.embeddings import LangchainEmbeddingsWrapper
                from ragas.llms import LangchainLLMWrapper

                class _SingleCandidateChatOpenAI(ChatOpenAI):
                    """
                    Gemini OpenAI-compat endpoint có 2 hạn chế:
                    1. Từ chối request có n>1:
                       400 - 'Multiple candidates is not enabled for this model'
                       → ChatOpenAI mặc định gửi "n": self.n, ép về 1.
                    2. Rate limit theo phút (15 req/phút free tier).
                       → throttle trước mỗi lần gọi để không chạm trần.
                    """

                    # 15 req/phút → chừa biên dùng 9 req/phút (~6.7s/lần gọi)
                    MIN_INTERVAL_S = 60.0 / RAGAS_RPM_LIMIT
                    _last_call = 0.0

                    def _get_request_payload(self, *args, **kwargs):
                        payload = super()._get_request_payload(*args, **kwargs)
                        if isinstance(payload, dict):
                            payload["n"] = 1
                        return payload

                    def _throttle(self) -> None:
                        import time as _t
                        wait = self.MIN_INTERVAL_S - (_t.monotonic() - type(self)._last_call)
                        if wait > 0:
                            _t.sleep(wait)
                        type(self)._last_call = _t.monotonic()

                    def invoke(self, *args, **kwargs):
                        self._throttle()
                        return super().invoke(*args, **kwargs)

                    async def ainvoke(self, *args, **kwargs):
                        self._throttle()
                        return await super().ainvoke(*args, **kwargs)

                llm = _SingleCandidateChatOpenAI(model=RAGAS_CHAT_MODEL, api_key=api_key,
                                                 base_url=GEMINI_BASE_URL, temperature=0, n=1)
                # Gemini OpenAI-compat endpoint không có text-embedding-3-small (404),
                # dùng gemini-embedding-001 thay thế.
                embeddings = OpenAIEmbeddings(model=RAGAS_EMBED_MODEL, api_key=api_key,
                                              base_url=GEMINI_BASE_URL, check_embedding_ctx_length=False)
                llm_w = _ThrottledLLMWrapper(LangchainLLMWrapper(llm), rpm=RAGAS_RPM_LIMIT)
                emb_w = LangchainEmbeddingsWrapper(embeddings)
                for metric in eval_metrics:
                    metric.llm = llm_w
                    if hasattr(metric, "embeddings"):
                        metric.embeddings = emb_w
            except Exception as cfg_e:
                print(f"  ⚠️  Cannot configure custom LLM/embeddings ({cfg_e}) — falling back to RAGAS defaults.")

        # max_workers=1: tránh chạy song song gây 429.
        # timeout phải lớn: _ThrottledLLMWrapper xếp hàng các call LLM, và thời
        # gian chờ trong hàng CŨNG được tính vào timeout của RAGAS. 180s là quá
        # ngắn khi 20 câu × 4 metric đều phải xếp hàng (~6.7s/lần gọi).
        from ragas.run_config import RunConfig
        run_config = RunConfig(max_workers=1, timeout=900, max_retries=3,
                               max_wait=20, log_tenacity=False)

        result = evaluate(dataset, metrics=eval_metrics, run_config=run_config)
        df = result.to_pandas()

        def _clean(val) -> float:
            """NaN (do LLM call fail) về 0.0 để không lan sang JSON/so sánh."""
            try:
                f = float(val)
            except (TypeError, ValueError):
                return 0.0
            return 0.0 if pd.isna(f) else f

        per_question = []
        for _, row in df.iterrows():
            f_val = _clean(row["faithfulness"]) if "faithfulness" in row else 0.0
            ar_val = _clean(row["answer_relevancy"]) if "answer_relevancy" in row else 0.0
            cp_val = _clean(row["context_precision"]) if "context_precision" in row else 0.0
            cr_val = _clean(row["context_recall"]) if "context_recall" in row else 0.0

            per_question.append(EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]),
                ground_truth=str(row["ground_truth"]),
                faithfulness=f_val,
                answer_relevancy=ar_val,
                context_precision=cp_val,
                context_recall=cr_val,
            ))

        def _get_score(name):
            """Ưu tiên trung bình từ per_question để NaN của một câu không
            làm hỏng cả aggregate (result[name] có thể là NaN nếu mọi câu lỗi)."""
            vals = [getattr(pq, name, 0.0) for pq in per_question]
            if vals:
                return float(sum(vals) / len(vals))
            try:
                return _clean(result[name])
            except Exception:
                return 0.0

        return {
            "faithfulness": _get_score("faithfulness"),
            "answer_relevancy": _get_score("answer_relevancy"),
            "context_precision": _get_score("context_precision"),
            "context_recall": _get_score("context_recall"),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {type(e).__name__}: {e}")
        per_question = [
            EvalResult(q, a, c, gt, 0.0, 0.0, 0.0, 0.0)
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": per_question,
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    analyzed = []
    for item in eval_results:
        metrics = {
            "faithfulness": item.faithfulness,
            "answer_relevancy": item.answer_relevancy,
            "context_precision": item.context_precision,
            "context_recall": item.context_recall,
        }
        avg_score = sum(metrics.values()) / len(metrics) if metrics else 0.0
        worst_metric = min(metrics, key=metrics.get) if metrics else "faithfulness"
        worst_score = metrics.get(worst_metric, 0.0)
        diag, fix = diagnostic_tree.get(worst_metric, ("Unknown failure", "Review pipeline"))

        analyzed.append({
            "question": item.question,
            "answer": item.answer,
            "ground_truth": item.ground_truth,
            "worst_metric": worst_metric,
            "score": float(worst_score),
            "avg_score": float(avg_score),
            "diagnosis": diag,
            "suggested_fix": fix,
        })

    analyzed.sort(key=lambda x: (x["avg_score"], x["score"]))
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
