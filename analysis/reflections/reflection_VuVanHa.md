# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Vũ Văn Hà  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 10/04/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| **Semantic chunking** | M1 | `chunk_semantic()` (`src/m1_chunking.py:98`) | Threshold cosine 0.85 với `all-MiniLM-L6-v2`. Chạy toàn corpus (26 docs) cho **1.024 chunks** so với **512 chunks** của `chunk_basic()` → semantic tách nhỏ hơn ~2× vì câu tiếng Việt ngắn và cosine giữa 2 câu liên tiếp thường < 0.85. Đổi lại không cắt đứt ý. **Bài học:** threshold 0.85 hơi chặt với tiếng Việt, nên hạ xuống ~0.7-0.75 nếu muốn chunk lớn hơn. |
| **Hierarchical chunking** | M1 | `chunk_hierarchical()` | Parent 2048 / Child 256 → **26 docs → 104 children**. Chiến lược mình chọn cho production vì retrieve theo child (precision) nhưng trả về parent (context đủ cho câu multi-hop như "Senior 9 năm thâm niên → 18 ngày phép"). Trade-off: context dài hơn → tốn token, dễ loãng nếu LLM không tập trung. |
| **BM25 + Dense fusion (RRF)** | M2 | `reciprocal_rank_fusion()` (`src/m2_search.py:149`) | **Bài học lớn nhất của lab.** Corpus có cặp "version conflict" (`nghi_phep_nam_v2023.md` 12 ngày vs `v2024.md` 15 ngày; `mat_khau_v1.md` 90 ngày vs `v2.md` 120 ngày). Dense (bge-m3) gần như không phân biệt được vì ngữ nghĩa gần nhau; BM25 bắt được nhờ khớp từ khoá ("2024", "v2.0"). RRF `score = Σ 1/(k + rank + 1)` với k=60 hợp nhất hai thứ tự xếp hạng mà **không cần chuẩn hoá score** — ưu điểm lớn so với weighted-sum (BM25 ~15 còn cosine ~0.8, không cùng thang đo). |
| **Vietnamese segmentation** | M2 | `segment_vietnamese()` | underthesea tách từ đa âm tiết, thay `_` bằng space. Không có bước này, BM25 tiếng Việt match kém vì coi cả từ là 1 token dài. |
| **Cross-encoder reranking** | M3 | `CrossEncoderReranker.rerank()` (`src/m3_rerank.py:36`) | `BAAI/bge-reranker-v2-m3`, đo **~150-400ms/call** trên CPU (nặng, 2.2B tham số). Rerank top-20 → top-3 giúp context gửi LLM sạch hơn. Mình cũng implement `FlashrankReranker` (ONNX `ms-marco-MultiBERT-L-12`): **~70-115ms** nhưng chất lượng thấp hơn (không fine-tune tiếng Việt, điểm rời rạc 0.996 vs 0.0038). |
| **RAGAS 4 metrics** | M4 | `evaluate_ragas()` (`src/m4_eval.py`) | Faithfulness (không hallucinate), Answer Relevancy (đúng ý), Context Precision (context liên quan), Context Recall (context đủ thông tin). Sau khi vượt qua lỗi interface + quota, đo được trên 2 câu mẫu: `faithfulness=1.0`, `answer_relevancy=0.81`, `context_recall=1.0`, `context_precision=0.02`. **Bài học quan trọng nhất của phần này:** mỗi metric đo một mặt khác nhau — `faithfulness` cao trong khi `context_precision` gần 0 cho thấy context sai đúng ngữ nghĩa nhưng lẫn nhiều nội dung thừa. Nếu chỉ nhìn 1 metric sẽ kết luận sai về chất lượng hệ thống. |
| **Contextual embeddings / Enrichment** | M5 | `contextual_prepend()` + `_enrich_single_call()` | Anthropic benchmark: prepending 1 câu mô tả "chunk này nằm ở đâu trong tài liệu" giảm ~49% retrieval failure. Mình dùng **combined single-call mode** (1 call/chunk thay vì 4): 4 technique × 104 chunks = 416 calls vs 104 calls. Thêm `ENRICH_MAX_CHUNKS` + cache JSON để chạy lại không tốn quota. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### 1. RAGAS trả về toàn 0.0 — bug dataset typing
- **Exact error:** `ValueError: Dataset feature "contexts" should be of type Sequence[string], got <class 'datasets.features.features.Value'>`
- **Nguyên nhân gốc rễ:** `Dataset.from_dict()` tự suy luận schema từ dữ liệu. Khi list chỉ có **1 phần tử**, `datasets` không nhận ra đây là list-of-string mà gán kiểu `Value` (scalar). Test `tests/test_m4.py` chỉ truyền 1 câu nên lộ bug ngay; chạy thật 20 câu thì vẫn sai vì feature đã bị gán kiểu trước đó.
- **Cách debug:** đọc traceback → thấy lỗi đến từ `datasets`, không phải RAGAS. Kiểm chứng bằng script tối giản tái hiện đúng lỗi, rồi sửa bằng cách khai báo tường minh `Features({... "contexts": Sequence(Value("string")) ...})`.
- **Bài học:** `try/except` rộng trong scaffold đã **che mất** lỗi này — code cũ nuốt exception rồi trả về 4 metric bằng 0.0, khiến mình hiểu sai là "chưa implement". Đã thay bằng in log `type(e).__name__: {e}` để lỗi thật lộ ra.

### 2. Gemini từ chối request có `n > 1`
- **Exact error:** `400 - [{'error': {'code': 400, 'message': 'Multiple candidates is not enabled for this model', 'status': 'INVALID_ARGUMENT'}}]`
- **Nguyên nhân gốc rễ:** `ChatOpenAI` của LangChain mặc định gửi field `"n": self.n` trong payload. Gemini OpenAI-compat endpoint không hỗ trợ multi-candidate.
- **Cách debug:** grep `langchain_openai/chat_models/base.py` → thấy dòng `"n": self.n,` trong `_get_request_payload()`. Sửa bằng subclass override `_get_request_payload` ép `payload["n"] = 1`.
- **Bài học:** wrapper của provider không tương đương 100% API gốc — phải kiểm tra payload thực tế, không chỉ tin tên class.

### 3. Embeddings 404 trên Gemini OpenAI-compat endpoint
- **Exact error:** `models/text-embedding-3-small is not found for API version v1main, or is not supported for embedContent`
- **Cách debug:** script thử lần lượt `text-embedding-3-small`, `gemini-embedding-001`, `text-embedding-004` → chỉ `gemini-embedding-001` trả 200. Phát hiện thêm `gemini-2.5-flash`/`2.0-flash` đã bị gỡ với tài khoản mới (404), chỉ `gemini-3.8-flash` còn dùng được.
- **Sửa:** tách `RAGAS_CHAT_MODEL` / `RAGAS_EMBED_MODEL` / `GEMINI_BASE_URL` ra `config.py` để không hardcode lặp lại nhiều file.

### 4. RAGAS đòi object có `set_run_config`
- **Exact error:** `AttributeError: 'ChatOpenAI' object has no attribute 'set_run_config'`
- **Nguyên nhân gốc rễ:** RAGAS 0.1.x gọi `metric.llm.set_run_config(run_config)`. Gán `ChatOpenAI` thô vào `metric.llm` là sai interface.
- **Cách debug:** grep trong `site-packages/ragas` → thấy `set_run_config` nằm ở `ragas/llms/base.py` và `ragas/embeddings/base.py` → phải bọc qua `LangchainLLMWrapper` / `LangchainEmbeddingsWrapper`.

### 5. Sai signature `RunConfig`
- **Exact error:** `TypeError: RunConfig.__init__() got an unexpected keyword argument 'raise_exceptions'`
- **Cách debug:** đọc `site-packages/ragas/run_config.py` để lấy đúng danh sách field (`max_workers`, `timeout`, `max_retries`, `max_wait`, `log_tenacity`) thay vì đoán.

### 6. NaN lan vào JSON khi LLM call lỗi
- **Hiện tượng:** metric trả `nan` → `json.dump` ghi `NaN` (không phải JSON chuẩn) → tooling đọc report hỏng.
- **Sửa:** thêm `_clean()` ép `NaN → 0.0`, và `_get_score()` ưu tiên trung bình từ `per_question` để một câu lỗi không kéo hỏng cả aggregate.

### 7. Hết quota Gemini (20 requests/ngày free tier) — vấn đề lớn nhất
- **Exact error:** `429 RESOURCE_EXHAUSTED ... Quota exceeded ... generate_content_free_tier_requests, limit: 20, model: gemini-3.8-flash ... retry in 13h`
- **Phân tích:** 104 chunks × 1 call = **104 requests** vượt quota 20/ngày. Đây không phải bug code mà là **vấn đề thiết kế cost model** mà mình chưa tính tới trước lab.
- **Ba hành động đã làm:**
  1. `ENRICH_MAX_CHUNKS` giới hạn số chunk gọi LLM, phần còn lại dùng heuristic offline (`_enrich_offline`).
  2. Cache JSON theo hash SHA-256 của `(source, methods, text)` → chạy lại không tốn quota lần 2.
  3. **Circuit breaker** trong `config.py`: `disable_llm()` + `is_quota_error()` — gặp 429 một lần là tắt LLM cho toàn bộ tiến trình, tránh 20 câu × nhiều lần retry thất bại (vừa chậm vừa tốn thêm quota).
- **Thiếu kiến thức:** trước lab này mình không biết Gemini free tier chỉ 20 req/ngày/model. Bổ sung: đọc `quotaValue` trong error message để biết chính xác hạn mức, và **luôn thiết kế fallback offline từ đầu** thay vì coi LLM là thành phần bắt buộc.

---

## Phần 2b: Vòng lặp thứ hai — đổi model để thoát quota

Sau khi xử lý xong 7 lỗi trên, RAGAS **vẫn trả 0.0** vì `gemini-3.8-flash` đã hết quota 20 req/ngày. Mình thử giải pháp: **đổi sang model khác**. Kết quả là một chuỗi phát hiện mới, hoàn toàn nằm ngoài dự kiến ban đầu.

### 8. Quota Gemini có **2 lớp**, và tính RIÊNG theo từng model

Đây là phát hiện quan trọng nhất của phiên làm việc này.

**Thí nghiệm:** thử 5 model trên cùng một API key.

| Model | Kết quả |
|-------|---------|
| `gemini-3.8-flash` | 429 — hết quota **20 req/ngày** |
| `gemini-3.5-flash` | 200 ✅ |
| **`gemini-3.5-flash-lite`** | **200 ✅** |
| `gemini-flash-lite` | 404 — không tồn tại |
| `gemini-3-flash-lite` | 404 — không tồn tại |

**Phát hiện đắt giờ:** hạn mức **KHông phải tính chung cho cả project**, mà **tính riêng cho từng model**. Nghĩa là `gemini-3.8-flash` hết quota thì `gemini-3.5-flash-lite` vẫn còn nguyên hạn mức riêng. Đây là lối thoát mà chỉ phát hiện được bằng cách **thử thực tế**, không thể suy ra từ tài liệu.

Nhưng khi đổi sang flash-lite, RAGAS lộ ra tầng giới hạn thứ hai:

| | `gemini-3.8-flash` | `gemini-3.5-flash-lite` |
|---|---|---|
| Lớp 1: **20 req/ngày** | ❌ Hết | ✅ Còn |
| Lớp 2: **15 req/phút** | ✅ Chưa chạm | ❌ Chạm ngay khi chạy RAGAS |

→ Nếu chỉ nhìn thấy lỗi lớp 1 rồi đổi model, mình sẽ tưởng đã xong. Thực tế lớp 2 mới là nút thắt thật sự khi chạy eval.

### 9. Throttle đặt sai tầng — không có tác dụng

- **Sai lầm đầu tiên:** thêm `sleep` vào `ChatOpenAI.invoke()` và `ainvoke()`.
- **Triệu chứng:** vẫn nhận `429 ... limit: 15 ... retry in 40s` dù đã giới hạn tốc độ.
- **Cách debug:** grep trong `ragas/llms/base.py` → RAGAS **không** gọi `invoke()`. Nó gọi `self.llm.agenerate()` (dòng 80) và `generate()` (dòng 59). Throttle của mình nằm trên đường gọi hoàn toàn khác → không bao giờ được kích hoạt.
- **Sửa:** viết `_ThrottledLLMWrapper` bọc **quanh** `LangchainLLMWrapper`, thờ điểm vào đúng hai method RAGAS thực sự gọi.

**Bài học:** throttle không phải chuyện "thêm một sleep vào đâu đó". Phải xác định chính xác **call path** thật sự của thư viện, nếu không chỉ tạo cảm giác an toàn giả.

### 10. `time.sleep()` trong async — block event loop gây TimeoutError

- **Exact error:** `TimeoutError()` xuất hiện ở `Job[6]`, `Job[2]` — metric `context_precision` luôn = 0.0.
- **Nguyên nhân gốc rễ:** `_wait()` của mình dùng `time.sleep()` bên trong hàm `async def agenerate()`. `time.sleep` **chặn toàn bộ event loop**, nên các coroutine khác đang chờ không được nhường CPU và bị đánh dấu timeout.
- **Cách debug:** metric nào bị 0.0 thì lần theo metric đó dùng API nào. `context_precision` là metric duy nhất **gọi LLM 2 lần tuần tự** (tách statement → chấm điểm verdict) → thời gian chờ nhân đôi → vượt trần.
- **Sửa:** dùng `asyncio.sleep()` + `asyncio.Lock()` để xếp hàng mà không block loop.
- **Sửa bổ sung:** `timeout=180` của RAGAS tính **cả thời gian chờ trong hàng**, nên khi 20 câu × 4 metric đều xếp hàng (~6,7s/lần) thì 180s là không đủ → tăng lên `900`.

### 11. Đánh đổi: chất lượng vs. thời gian

Sau khi hết lỗi, đo thực tế trên 2 câu:

| Metric | Kết quả |
|--------|---------|
| Faithfulness | **1.0** |
| Answer Relevancy | **0.81** |
| Context Recall | **1.0** |
| Context Precision | 0.02 |

Nhưng **2 câu mất ~800 giây (13 phút)** → ước tính **20 câu ≈ 2,2 giờ**.

Nguyên nhân: 20 câu × 4 metric × (1–2 LLM call) ≈ 80–160 lần gọi, mỗi lần bị throttle ~6 giây. Đây là hệ quả tất yếu của giới hạn 15 req/phút, không phải lỗi hiệu năng.

**Kết luận:** với free tier, RAGAS chỉ thực tế cho bộ eval nhỏ (~5–10 câu). Với 20 câu cần API có billing, hoặc chấp nhận chạy overnight. Đây là ràng buộc kinh tế thuần túy, và nó **luôn tồn tại trong mọi hệ thống LLM**, không riêng lab này.

### 12. Model nhẹ hơn vẫn cần guardrail về mặt nghiệp vụ

- **Triệu chứng:** `test_m5.py::test_summarize_shorter_than_original` FAIL sau khi đổi model.
- **Chi tiết:** input 65 ký tự, `gemini-3.5-flash-lite` trả về summary **138 ký tự** — tức **dài hơn cả bản gốc**, và có phần **bịa thêm** ("Quy định này áp dụng chung cho toàn bộ nhân sự chính thức tại công ty" không hề có trong input).
- **Root cause:** model flash-lite có xu hướng **diễn giải thêm** thay vì rút gọn. Đây là hành vi model, không phải bug code.
- **Sửa:** thêm guardrail trong `summarize_chunk()` — nếu summary dài hơn `text` thì cắt lại theo câu.
- **Bài học quan trọng nhất của phần này:** **chuyển sang model rẻ hơn không chỉ đổi chất lượng, mà đổi cả hành vi.** Guardrail nghiệp vụ (output phải ngắn hơn input, phải có dấu `?`, phải là JSON hợp lệ…) là thứ giữ hệ thống ổn định khi đổi model — không nên viết prompt rồi tin là model sẽ làm đúng.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý tra cứu chính sách nội bộ (HR/IT Policy Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** corpus 26 tài liệu Markdown tiếng Việt → `chunk_basic()` → dense-only search (BAAI/bge-m3) → top-3 → LLM trả lời. Chưa có hybrid, chưa rerank, chưa enrichment, chưa eval.
- **Bottlenecks đang gặp:**
  1. **Version conflict** — 4 cặp tài liệu v1/v2 trùng chủ đề (nghỉ phép 12 vs 15 ngày, mật khẩu 90 vs 120 ngày). Dense-only dễ trả bản cũ → trả lời sai chính sách. Đây là failure nghiêm trọng nhất với policy assistant.
  2. **Câu hỏi multi-hop** — "Senior 9 năm thâm niên → mấy ngày phép?" cần ghép 2 đoạn (bảng lương + bảng phép). Chunk 256 ký tự thường cắt mất một trong hai.
  3. **Câu hỏi phủ định** — "nhân viên thử việc có được nghỉ phép năm không?" cần trả lời "KHÔNG", dễ bị LLM đảo ngược nếu context chứa cả điều kiện áp dụng.
  4. **Chi phí embedding** — 1 API call/chunk × 104 chunks vượt quota free tier 20 req/ngày.

#### 2. Kế hoạch cải tiến
1. **Chunking:** chọn **Hierarchical (parent 2048 / child 256)** thay vì semantic thuần. Lý do: câu hỏi policy thường cần ngữ cảnh điều kiện (áp dụng cho ai, từ khi nào); cắt chunk nhỏ dễ mất điều kiện. Đánh giá lại semantic ở threshold 0.75 nếu cần chunk phẳng.
2. **Search:** bắt buộc **Hybrid = BM25 (có underthesea) + Dense (bge-m3) + RRF(k=60)**. Lý do: BM25 bắt được version ("2024", "v2.0") qua từ khoá; dense hiểu câu hỏi diễn đạt lại ("thưởng Tết" ↔ "lương tháng 13"). Thiếu một trong hai đều fail một loại câu hỏi.
3. **Reranking:** có, `bge-reranker-v2-m3` để ưu tiên chất lượng; nếu cần latency <100ms thì switch `FlashrankReranker`. Viết điều kiện chọn theo SLA thay vì hardcode.
4. **Evaluation:** RAGAS 4 metrics làm chuẩn, nhưng **bắt buộc thêm metric riêng cho version conflict** vì đó là failure nghiêm trọng nhất mà 4 metric RAGAS không phát hiện rõ. Regression test: hỏi 4 câu version, đòi kết quả phải chứa "2024"/"v2.0".
5. **Enrichment:** ưu tiên **contextual prepend** (rẻ, giảm ~49% retrieval failure theo benchmark Anthropic) + **auto metadata `version`/`effective_date`** để filter chính sách đang hiện hành. Bỏ HyQA nếu index size là constraint (nhân đôi số vector, giá trị thấp hơn contextual prepend với corpus đã có cấu trúc markdown rõ ràng).

#### 3. Timeline triển khai
- **Tuần 1:** Implement hybrid BM25 + dense + RRF; viết regression test cho 4 cặp version conflict; đo context_recall trên bộ test 20 câu.
- **Tuần 2:** Thêm reranker, so sánh bge-m3 vs flashrank về cả accuracy lẫn latency; chạy RAGAS lần 2 và so sánh với baseline.
- **Tuần 3:** Thêm metadata `effective_date` + filter "chính sách hiện hành"; đánh giá tác động lên 4 câu version.
- **Tuần 4:** Tối ưu cost — cache embedding theo hash nội dung, batching, cân nhắc `gemini-embedding-001` thay bge-m3 nếu chấp nhận giảm nhẹ chất lượng semantic; theo dõi quota theo ngày.

---

## Tự đánh giá

**Điểm mạnh:**
- Debug RAGAS triệt để bằng cách **đọc source trong `site-packages`** thay vì đoán theo kinh nghiệm. 12/12 lỗi đều được tìm ra bằng cách lần theo call path thật (`Dataset`, `langchain_openai`, `ragas/llms/base.py`, `ragas/run_config.py`).
- Không dừng ở "hết quota là chấp nhận 0.0" mà tìm cách thoát: nhận ra quota tính **riêng theo model**, chuyển sang flash-lite, rồi phải xử lý tiếp tầng giới hạn phút. Kết quả là **4 metric chạy thật** thay vì 0.0.
- Biến hạn chế hạ tầng thành feature: cache SHA-256, circuit breaker, fallback offline, throttle wrapper, offline retrieval eval — những thứ này **dùng được cho mọi project LLM sau này**.

**Điểm cần cải thiện:**
- Đã lãng phí thời gian vì `try/except` rộng trong scaffold che mất lỗi thật. Nếu in log chi tiết ngay từ đầu sẽ tiết kiệm được nhiều hơn.
- Throttle đặt sai tầng (`invoke()` thay vì `agenerate()`) — lỗi này chỉ lộ ra vì mình **kiểm chứng lại sau khi sửa** thay vì tin là đã xong. Sai lầm chung: **tin vào việc mình vừa sửa mà không test lại từ đầu**.
- Ưu tiên kiểm tra **tầng giới hạn thứ hai** ngay từ đầu: đã đổi model để thoát lớp "ngày" rồi mới phát hiện lớp "phút". Nếu đọc kỹ error message ngay từ đầu (`quotaId: GenerateRequestsPerMinutePerProjectPerModel`) sẽ tiết kiệm một vòng thử.

**Bài học tổng quát nhất của lab:**
> Hệ thống LLM production không bao giờ "chạy đúng" chỉ vì code đúng. Ba thứ luôn phải có sẵn:
> **(1) guardrail nghiệp vụ** trên mọi output, vì hành vi model thay đổi theo model bạn chọn;
> **(2) kiểm soát tốc độ** vì mọi nhà cung cấp đều có quota nhiều tầng;
> **(3) fallback hoạt động offline** để hệ thống sống sót khi hết quota.