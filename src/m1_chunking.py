from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import os
import re
import sys
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────

_semantic_model = None

def _get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_model


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "strategy": "semantic", "chunk_index": 0})]

    import numpy as np
    model = _get_semantic_model()
    embeddings = model.encode(sentences)

    groups = [[sentences[0]]]
    for i in range(1, len(sentences)):
        vec1 = embeddings[i - 1]
        vec2 = embeddings[i]
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        sim = float(np.dot(vec1, vec2) / (norm1 * norm2 + 1e-9))

        if sim < threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    chunks = []
    for idx, group in enumerate(groups):
        chunk_text = " ".join(group)
        chunks.append(Chunk(text=chunk_text, metadata={**metadata, "strategy": "semantic", "chunk_index": idx}))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        if text.strip():
            paragraphs = [text.strip()]
        else:
            return ([], [])

    parents: list[Chunk] = []
    current_parent_paras = []
    current_parent_len = 0

    for para in paragraphs:
        if current_parent_len + len(para) > parent_size and current_parent_paras:
            pid = f"parent_{len(parents)}"
            parent_text = "\n\n".join(current_parent_paras)
            p_meta = {**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": len(parents)}
            parents.append(Chunk(text=parent_text, metadata=p_meta))
            current_parent_paras = [para]
            current_parent_len = len(para)
        else:
            current_parent_paras.append(para)
            current_parent_len += len(para) + 2

    if current_parent_paras:
        pid = f"parent_{len(parents)}"
        parent_text = "\n\n".join(current_parent_paras)
        p_meta = {**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": len(parents)}
        parents.append(Chunk(text=parent_text, metadata=p_meta))

    children: list[Chunk] = []
    for parent in parents:
        pid = parent.metadata["parent_id"]
        p_text = parent.text
        lines = [line.strip() for line in re.split(r'(?<=[.!?])\s+|\n+', p_text) if line.strip()]

        current_child_lines = []
        current_child_len = 0

        for line in lines:
            if current_child_len + len(line) > child_size and current_child_lines:
                c_text = " ".join(current_child_lines)
                c_meta = {**metadata, "chunk_type": "child", "chunk_index": len(children)}
                children.append(Chunk(text=c_text, metadata=c_meta, parent_id=pid))
                current_child_lines = [line]
                current_child_len = len(line)
            else:
                current_child_lines.append(line)
                current_child_len += len(line) + 1

        if current_child_lines:
            c_text = " ".join(current_child_lines)
            c_meta = {**metadata, "chunk_type": "child", "chunk_index": len(children)}
            children.append(Chunk(text=c_text, metadata=c_meta, parent_id=pid))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    parts = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks = []
    current_section = ""
    current_body = ""

    for part in parts:
        part_str = part.strip()
        if not part_str:
            continue
        if re.match(r'^#{1,3}\s+.+$', part_str):
            if current_body or current_section:
                chunk_text = f"{current_section}\n\n{current_body}".strip() if current_section else current_body.strip()
                if chunk_text:
                    meta = {**metadata, "strategy": "structure", "chunk_index": len(chunks)}
                    if current_section:
                        meta["section"] = current_section
                    chunks.append(Chunk(text=chunk_text, metadata=meta))
            current_section = part_str
            current_body = ""
        else:
            if current_body:
                current_body += "\n\n" + part_str
            else:
                current_body = part_str

    if current_body or current_section:
        chunk_text = f"{current_section}\n\n{current_body}".strip() if current_section else current_body.strip()
        if chunk_text:
            meta = {**metadata, "strategy": "structure", "chunk_index": len(chunks)}
            if current_section:
                meta["section"] = current_section
            chunks.append(Chunk(text=chunk_text, metadata=meta))

    if not chunks and text.strip():
        chunks.append(Chunk(text=text.strip(), metadata={**metadata, "strategy": "structure", "chunk_index": 0}))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
