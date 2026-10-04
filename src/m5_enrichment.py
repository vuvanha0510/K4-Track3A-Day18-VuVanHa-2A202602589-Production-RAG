from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ENRICH_CACHE_PATH,
    ENRICH_MAX_CHUNKS,
    disable_llm,
    get_llm_client,
    is_quota_error,
)


def _handle_llm_error(prefix: str, exc: Exception) -> None:
    """Log lỗi LLM; nếu là hết quota thì tắt LLM cho các call sau."""
    if is_quota_error(exc):
        disable_llm(f"{prefix}: quota/rate limit")
        print(f"  ⚠️  {prefix} failed (quota) — chuyển sang fallback offline cho các chunk còn lại.")
    else:
        print(f"  ⚠️  {prefix} failed: {type(exc).__name__}: {exc}")


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    client, model = get_llm_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            summary = resp.choices[0].message.content.strip()
            # Một bản tóm tắt dài hơn bản gốc là vô nghĩa (gemini-flash-lite
            # đôi khi diễn giải thêm thay vì rút gọn) → cắt lại theo câu.
            if summary and len(summary) > len(text):
                kept = []
                for s in summary.split(". "):
                    if s.strip():
                        kept.append(s.strip())
                    if sum(len(x) for x in kept) > len(text):
                        break
                summary = ". ".join(kept).rstrip(".") + "." if kept else text
            return summary
        except Exception as e:
            _handle_llm_error("LLM summarize", e)

    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    client, model = get_llm_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Mỗi câu hỏi nằm trên 1 dòng và BẮT BUỘC kết thúc bằng dấu '?'."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            questions = resp.choices[0].message.content.strip().split("\n")
            cleaned = []
            for q in questions:
                q = q.strip().lstrip("0123456789.-) ").strip()
                if not q:
                    continue
                # Luôn chuẩn hoá về dạng câu hỏi (Gemini đôi khi trả về mệnh đề
                # không có dấu "?" → làm HyQA mất ý nghĩa và test fail).
                if not q.endswith("?"):
                    q += "?"
                cleaned.append(q)
            return cleaned[:n_questions]
        except Exception as e:
            _handle_llm_error("LLM HyQA", e)

    import re
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    client, model = get_llm_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
            )
            context = resp.choices[0].message.content.strip()
            return f"{context}\n\n{text}"
        except Exception as e:
            _handle_llm_error("LLM contextual", e)

    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    client, model = get_llm_client()
    if client:
        try:
            import json as _json
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            return _json.loads(content)
        except Exception as e:
            _handle_llm_error("LLM metadata", e)

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    client, model = get_llm_client()
    if client:
        try:
            import json as _json
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": """Phân tích đoạn văn và trả về JSON DUY NHẤT (không markdown, không giải thích):
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1 kết thúc bằng dấu ?", "câu hỏi 2 kết thúc bằng dấu ?", "câu hỏi 3 kết thúc bằng dấu ?"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=400,
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            return _json.loads(content)
        except Exception as e:
            _handle_llm_error("Enrichment API", e)

    # LLM unavailable (thiếu key / hết quota) → dùng heuristic offline.
    # Không gọi tiếp 4 hàm riêng lẻ vì mỗi hàm đều thử 1 API call rồi lỗi,
    # gây tốn quota/thời gian vô ích.
    return _enrich_offline(text, source)


# ─── Offline Enrichment (không tốn API quota) ──────────────


def _enrich_offline(text: str, source: str) -> dict:
    """
    Enrichment hoàn toàn offline bằng heuristic (không gọi LLM).

    Dùng khi đã hết quota Gemini: vẫn giữ đúng schema đầu ra để pipeline
    không đổi hành vi, chỉ là chất lượng enrichment thấp hơn.
    """
    import re as _re

    # Summary: lấy 2 câu đầu.
    sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+|\n+', text) if len(s.strip()) > 15]
    summary = ". ".join(sentences[:2])
    if summary and not summary.endswith((".", "!", "?")):
        summary += "."

    # HyQA: biến câu khẳng định thành câu hỏi.
    questions = []
    for s in sentences[:3]:
        q = s.rstrip(".!?").strip()
        if not q:
            continue
        # "Nhân viên được nghỉ 12 ngày" → "Nhân viên được nghỉ bao nhiêu ngày?"
        q = _re.sub(r"\b\d[\d.,]*\b", "bao nhiêu", q)
        questions.append(q + "?")
    if not questions and sentences:
        questions = [sentences[0].rstrip(".!?") + "?"]

    context = f"Trích từ tài liệu \"{source}\". " if source else ""
    meta = {
        "topic": "general",
        "entities": [],
        "category": "policy",
        "language": "vi",
    }
    return {"summary": summary, "questions": questions[:3], "context": context, "metadata": meta}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    # --- Cache: khóa theo (text, source, methods) để chạy lại không tốn quota ---
    cache = {}
    if ENRICH_CACHE_PATH and os.path.exists(ENRICH_CACHE_PATH):
        try:
            import json as _json
            with open(ENRICH_CACHE_PATH, encoding="utf-8") as f:
                cache = _json.load(f)
        except Exception:
            cache = {}

    def _cache_key(text, source):
        import hashlib
        raw = f"{source}\x00{'+'.join(methods)}\x00{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _save_cache():
        if not ENRICH_CACHE_PATH:
            return
        try:
            import json as _json
            with open(ENRICH_CACHE_PATH, "w", encoding="utf-8") as f:
                _json.dump(cache, f, ensure_ascii=False)
        except Exception as cache_err:
            # Cache chỉ là tối ưu — lỗi ghi cache không được làm hỏng pipeline.
            print(f"  ℹ️  Không lưu được enrichment cache: {cache_err}")

    # Số chunk được gọi LLM thật (giới hạn quota); phần còn lại dùng fallback offline.
    budget = ENRICH_MAX_CHUNKS if ENRICH_MAX_CHUNKS and ENRICH_MAX_CHUNKS > 0 else len(chunks)
    api_calls = 0

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")
        key = _cache_key(text, source)
        cached = cache.get(key)
        # Mặc định an toàn cho mọi nhánh (nhánh non-combined không dùng context_line).
        context_line = ""

        if cached is not None:
            summary = cached.get("summary", "")
            questions = cached.get("questions", [])
            context_line = cached.get("context", "")
            auto_meta = cached.get("metadata", {})
            if use_combined:
                enriched_text = f"{context_line}\n\n{text}" if context_line else text
            else:
                # Mode riêng lẻ: enriched_text do contextual_prepend quyết định,
                # không dùng context_line của combined.
                enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            method_used = "+".join(methods) + "+cached"
        elif api_calls < budget:
            api_calls += 1
            method_used = "+".join(methods)
            if use_combined:
                result = _enrich_single_call(text, source)
                summary = result.get("summary", "")
                questions = result.get("questions", [])
                context_line = result.get("context", "")
                enriched_text = f"{context_line}\n\n{text}" if context_line else text
                auto_meta = result.get("metadata", {})
            else:
                summary = summarize_chunk(text) if "summary" in methods else ""
                questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
                enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
                auto_meta = extract_metadata(text) if "metadata" in methods else {}
            cache[key] = {"summary": summary, "questions": questions,
                          "context": context_line, "metadata": auto_meta}
        else:
            # Hết quota → fallback offline (không gọi API), vẫn giữ cấu trúc đầu ra.
            if use_combined:
                result = _enrich_offline(text, source)
                summary = result.get("summary", "")
                questions = result.get("questions", [])
                context_line = result.get("context", "")
                enriched_text = f"{context_line}\n\n{text}" if context_line else text
                auto_meta = result.get("metadata", {})
            else:
                summary = ""
                questions = []
                enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
                auto_meta = {}
            method_used = "+".join(methods) + "+offline"

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method=method_used,
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    _save_cache()
    if len(chunks) > budget:
        print(f"  ℹ️  Chỉ {budget}/{len(chunks)} chunk dùng LLM (ENRICH_MAX_CHUNKS); "
              f"phần còn lại dùng fallback offline.", flush=True)
    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
