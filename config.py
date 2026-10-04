"""Shared configuration for Lab 18."""

import os

from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") or GEMINI_API_KEY


# Circuit breaker: đặt True sau khi gặp lỗi quota (429) để các lời gọi
# sau đó lập tức dùng fallback offline thay vì chờ retry rồi mới fail.
_LLM_DISABLED = False
_LLM_DISABLE_REASON = ""


def disable_llm(reason: str = "") -> None:
    """Tắt LLM cho phần còn lại của tiến trình (dùng khi hết quota/rate limit)."""
    global _LLM_DISABLED, _LLM_DISABLE_REASON
    _LLM_DISABLED = True
    _LLM_DISABLE_REASON = reason


def llm_disabled_reason() -> str:
    return _LLM_DISABLE_REASON


def is_quota_error(exc: Exception) -> bool:
    """True nếu exception là lỗi quota/rate limit của Gemini."""
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()


def get_llm_client():
    """Get an OpenAI-compatible client and default model name."""
    if _LLM_DISABLED:
        return None, None
    openai_key = os.getenv("OPENAI_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if openai_key and openai_key.startswith("sk-"):
        from openai import OpenAI
        return OpenAI(api_key=openai_key), "gpt-4o-mini"
    elif gemini_key or openai_key:
        key = gemini_key or openai_key
        from openai import OpenAI
        return OpenAI(api_key=key, base_url=GEMINI_BASE_URL), RAGAS_CHAT_MODEL
    return None, None


def get_embedding_client():
    """Get (client, model) cho embeddings — dùng cho RAGAS.

    Gemini OpenAI-compat endpoint KHÔNG phục vụ `text-embedding-3-small`
    (trả về 404) → dùng `gemini-embedding-001`.
    """
    openai_key = os.getenv("OPENAI_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if openai_key and openai_key.startswith("sk-"):
        from openai import OpenAI
        return OpenAI(api_key=openai_key), "text-embedding-3-small"
    if gemini_key or openai_key:
        from openai import OpenAI
        return OpenAI(api_key=gemini_key or openai_key, base_url=GEMINI_BASE_URL), RAGAS_EMBED_MODEL
    return None, None

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Enrichment guardrails ---
# 1 API call/chunk × ~104 chunks vượt quota Gemini free tier (20 req/ngày/model).
# ENRICH_MAX_CHUNKS giới hạn số chunk được gọi LLM; các chunk còn lại dùng
# enrichment fallback offline (không tốn quota). ENRICH_CACHE_PATH lưu kết quả
# để chạy lại pipeline không phát sinh thêm API call.
ENRICH_MAX_CHUNKS = 20
ENRICH_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".enrich_cache.json")

# --- Evaluation (RAGAS) ---
# Gemini OpenAI-compat endpoint không phục vụ model của OpenAI,
# nên phải dùng đúng tên model Gemini cho cả chat lẫn embeddings.
# Gemini free tier chỉ 20 req/ngày/model, nhưng hạn mức tính RIÊNG cho từng model.
# gemini-3.8-flash đã hết quota (429) → dùng gemini-3.5-flash-lite (model nhẹ hơn,
# quota riêng, đủ làm LLM judge cho RAGAS). Đổi qua biến môi trường khi cần.
RAGAS_CHAT_MODEL = os.getenv("RAGAS_CHAT_MODEL", "gemini-3.5-flash-lite")
RAGAS_EMBED_MODEL = os.getenv("RAGAS_EMBED_MODEL", "gemini-embedding-001")
# Gemini free tier: 15 request/phút/model (ngoài 20 request/ngày).
# RAGAS gọi LLM judge nhiều lần nên cần throttle để không chạm trần phút.
# 10 req/phút chừa biên cho request nền (embeddings, health check).
RAGAS_RPM_LIMIT = int(os.getenv("RAGAS_RPM_LIMIT", "10"))
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")