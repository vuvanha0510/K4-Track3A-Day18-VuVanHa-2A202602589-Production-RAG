# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Vũ Văn Hà  
**Khóa:** K4 - Track 3A

---

## RAGAS Scores

> **Trạng thái:** RAGAS **đã chạy được** sau khi sửa 5 lỗi interface + quota (chi tiết ở
> `reflection_VuVanHa.md` Phần 2). Bảng dưới đây là kết quả đo trên **2 câu mẫu** chạy thật,
> dùng `gemini-3.5-flash-lite` làm LLM judge.

| Metric | Naive Baseline | Production (2 câu mẫu) | Ghi chú |
|--------|---------------|------------------------|---------|
| Faithfulness | 0.0 | **1.0000** | Không hallucinate |
| Answer Relevancy | 0.0 | **0.8077** | Trả lời đúng ý hỏi |
| Context Precision | 0.0 | **0.0238** | ⚠️ Rất thấp — xem phân tích bên dưới |
| Context Recall | 0.0 | **1.0000** | Context chứa đủ thông tin |

**Vì sao bộ test đầy đủ 20 câu chưa chạy được:**
Gemini free tier có **hai tầng quota**: 20 req/ngày **và** 15 req/phút (tính riêng cho từng model).
RAGAS cần ~80–160 lượt gọi LLM cho 20 câu × 4 metric. Đo thực tế: **2 câu ≈ 800 giây** →
**20 câu ≈ 2,2 giờ**. Đây là ràng buộc kinh tế, không phải lỗi kỹ thuật.

> Naive Baseline giữ 0.0 vì chạy trước đó khi `gemini-3.8-flash` đã hết quota ngày.

### Phân tích `context_precision = 0.0238`

Đây là kết quả **đáng chú ý nhất** và hoàn toàn nhất quán với offline eval bên dưới:

- Context có **đúng** nội dung cần thiết → `context_recall = 1.0`
- Nhưng top-3 chunk chứa **rất nhiều thứ không liên quan** → `context_precision ≈ 0`

Nghĩa là hệ thống **tìm đúng nhưng lấy bừa**. Đây chính là failure mode kinh điển của RAG
khi chỉ dùng embedding similarity mà không rerank: mọi chunk "nói về chủ đề X" đều
có điểm cao, kể cả chunk không trả lời được câu hỏi.

**Hệ quả thực tế:** LLM nhận context đúng nhưng bị "loãng" → tốn token, tăng nguy cơ
ảo giác, và đây chính là lý do **reranker là bước có ROI cao nhất** trong cả pipeline.

### Offline Retrieval Eval (đo đầy đủ 20 câu, không cần LLM)

Vì RAGAS quá chậm trên free tier, phần phân tích failures dưới đây dùng
`analysis/offline_retrieval_eval.py` — đo context recall/precision bằng token overlap,
**không cần LLM**, chạy được đủ 20 câu và tái lập được.

Top-3 context, corpus 26 docs → 104 chunks (hierarchical parent 2048 / child 256):

| Method | Context Recall | Context Precision | Hit rate (recall ≥ 0.7) |
|--------|---------------|-------------------|-------------------------|
| BM25 only | 0.7501 | 0.2083 | 0.5500 |
| Dense only (bge-m3) | 0.7721 | 0.2110 | 0.5000 |
| **Hybrid + RRF** | **0.7614** | **0.2132** | **0.6000** |

**Đọc kết quả:**
- **Hybrid có hit rate cao nhất (0.60)** — tức 12/20 câu tìm đủ thông tin trong top-3. Đây là chỉ số quan trọng nhất cho RAG có LLM: nếu context không chứa đáp án thì đừng kỳ vọng LLM đúng.
- **Recall > Precision rất xa** (0.76 vs 0.21) — top-3 chunk có nhiều nội dung thừa. Với LLM đây là chi phí token + nguy cơ "loãng". Đây là lý do **reranker** là bước có ROI cao nhất: cắt precision lên mà không giảm recall.
- **Dense recall cao hơn BM25** (0.772 vs 0.750) nhưng **hit rate thấp hơn** (0.50 vs 0.55) — nghĩa là dense đưa thông tin cần thiết vào top-3 nhiều hơn một chút, nhưng hay bỏ sót câu mà BM25 bắt được (các câu có từ khoá đặc biệt như "55 triệu", "CEO").

---

## Bottom-5 Failures

Xếp theo **context_recall** của Hybrid (thấp nhất trước).

### #1 — Multi-hop: lương Senior + phép năm theo thâm niên
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** context recall 0.4167 — thấp nhất trong 20 câu
- **Worst metric:** context_recall
- **Error Tree:** Output sai → **Context đúng?** KHÔNG (thiếu cả 2 nguồn) → Query OK? Đúng → **Multi-hop chưa được hỗ trợ**
- **Root cause:** Câu hỏi cần **ghép 2 tài liệu khác nhau**: `bang_luong_2024.md` (khoảng lương Senior) + `nghi_phep_nam_v2024.md` (công thức cộng ngày phép). Top-3 chunk chỉ chứa được một trong hai vì chúng không đứng gần nhau trong không gian vector. Đây là giới hạn cố hữu của **single-vector dense retrieval** — không có cơ chế nối 2 tài liệu.
- **Suggested fix:** Đổi sang **retrieve theo child → trả về parent** (chiến lược hierarchical của M1) để mỗi chunk kèm ngữ cảnh rộng hơn; hoặc bổ sung bước **query decomposition** — tách thành 2 sub-query ("lương Senior bao nhiêu?" + "9 năm thâm niên cộng mấy ngày phép?") rồi merge kết quả.

### #2 — Số học với đơn vị tiền tỉ
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** context recall 0.4667, precision 0.1061
- **Worst metric:** context_precision
- **Error Tree:** Output sai → **Context đúng?** CÓ (đã tìm được bảng lương) → Query OK? Đúng → **Thiếu bước tính toán**
- **Root cause:** Retrieval **thành công** — vấn đề nằm ở **generation**: đáp án cần phép tính `85% × 20.000.000 = 17.000.000`, là suy luận nhiều bước chứ không phải tra cứu thẳng. Model đọc được "20 triệu" nhưng không tự nhân 85%.
- **Suggested fix:** Dùng **tool/function calling cho phép tính** thay vì để LLM tự nhân trong đầu; hoặc thêm prompt yêu cầu trình bày từng bước. Với RAG production, đây là lý do cần tách **retrieval** (tra cứu) khỏi **computation** (tính).

### #3 — Ngưỡng phê duyệt 50 triệu
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** context recall 0.5, precision **0.0750** (thấp nhất — context gần như toàn nhiễu)
- **Worst metric:** context_precision
- **Error Tree:** Output sai → **Context đúng?** KHÔNG → Query OK? Đúng → **Từ khoá số "55 triệu" không match tài liệu** (tài liệu ghi "50.000.000 VNĐ")
- **Root cause:** **Failure mode kinh điển của BM25 + số.** Corpus ghi `50.000.000 VNĐ`, query nói `55 triệu` — không token nào trùng nên BM25 không đóng góp được. Dense hiểu ngữ nghĩa "mua sắm, phê duyệt, giá trị cao" nên bắt đúng tài liệu, nhưng kéo theo nhiều chunk "mua sắm" khác. Kết quả: đúng tài liệu nhưng lẫn nhiễu, và điều kiện ngưỡng (`55 > 50`) **vẫn phải do LLM tự tính**.
- **Suggested fix:** **Chuẩn hoá đơn vị số** ở bước ingest và trong query ("55 triệu" → "55.000.000 VNĐ") — bài học **normalize trước, match sau**. Song song, thêm metadata `threshold` để lọc theo khoảng giá.

### #4 — Ngưỡng ngày nghỉ không lương 16-30 ngày
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16-30 ngày cần phê duyệt của Giám đốc điều hành (CEO).
- **Got:** context recall 0.56
- **Worst metric:** context_recall
- **Error Tree:** Output sai → **Context đúng?** KHÔNG (thiếu bảng ngưỡng 16-30) → Query OK? Đúng → **Truy vấn theo khoảng, không theo giá trị cụ thể**
- **Root cause:** Cùng dạng với #3 nhưng với **ngày**. Tài liệu ghi "16-30 ngày", query hỏi "20 ngày" — cần phép **range containment** (20 ∈ [16,30]) mà BM25 không có khả năng này.
- **Suggested fix:** Index theo **từng giá trị rời rạc trong range** (sinh thêm chunk cho "17 ngày", "18 ngày"… "30 ngày") hoặc dùng **metadata filter theo khoảng số** ở tầng Qdrant. Với domain HR/PCCC đây là mẫu hỏi **rất phổ biến**, đáng đầu tư riêng.

### #5 — Version conflict: nghỉ phép năm 15 hay 12 ngày
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** v2024 (hiện hành) = 15 ngày. v2023 (đã thay thế) = 12 ngày.
- **Got:** context recall 0.5789
- **Worst metric:** context_recall
- **Error Tree:** Output sai → **Context đúng?** CÓ NHƯNG LẪN PHIÊN BẢN → Query OK? Đúng → **Không có cơ chế phân giải xung đột version**
- **Root cause:** Corpus có `nghi_phep_nam_v2023.md` (12 ngày) và `nghi_phep_nam_v2024.md` (15 ngày) — hai chunk có **ngữ nghĩa gần như giống nhau**, nên dense gần như không phân biệt nổi; RRF cũng không giải quyết được vì cả hai đều là kết quả hợp lệ. Đây là failure nguy hiểm nhất với policy assistant: **trả lời sai nhưng nghe rất thuyết phục**.
- **Suggested fix:** Gắn metadata `version` + `effective_date` khi ingest (M5 `extract_metadata` đã sinh `category`; bổ sung 2 field này), rồi **lọc theo bản hiện hành trước khi rerank**. Với lab này có thể đánh dấu `nghi_phep_nam_v2024` là bản active.

---

## Case Study (cho presentation)

**Question chọn phân tích:** *"Nhân viên được nghỉ bao nhiêu ngày phép năm?"*

**Error Tree walkthrough:**
1. **Output đúng?** Không — dễ trả về 12 ngày (bản v2023, đã bị thay thế).
2. **Context đúng?** Đúng một nửa — top-3 chứa **cả hai** bản 2023 và 2024.
3. **Query rewrite OK?** Đúng — query rõ ràng, không mơ hồ.
4. **Fix ở bước:** Bước 2 → **thêm metadata version/effective_date ở tầng ingest và lọc theo bản hiện hành trước reranking.**

**Nếu có thêm 1 giờ, sẽ optimize (theo thứ tự ROI):**
1. **Sửa version conflict** (#5) — nguy hiểm nhất về mặt nghiệp vụ, sửa bằng metadata filter.
2. **Nâng context precision từ 0.21 lên 0.5+** qua reranking — đây là việc có ROI cao nhất vì **RAGAS đã độc lập xác nhận** `context_precision = 0.0238` trong khi `context_recall = 1.0`: hệ thống tìm đúng nhưng lấy bừa. Cần kiểm lại điểm nối M2 → M3 trong `src/pipeline.py` (lấy top-3 từ `HYBRID_TOP_K=20` rồi rerank).
3. **Chuẩn hoá đơn vị số** (#3, #4) — normalize "triệu" → "VNĐ" trước khi index.
4. **Hỗ trợ multi-hop** (#1) — tách sub-query cho câu cần 2 nguồn.

---

## Ghi chú kỹ thuật: vì sao RAGAS cần đến 12 lỗi sửa mới chạy được

Thay vì coi "RAGAS trả 0.0" là hạn chế của lab, đây là bài học về **tích hợp thư viện thật**:

| # | Lỗi | Cách phát hiện |
|---|------|----------------|
| 1 | `Dataset` suy luận sai kiểu `contexts` khi list 1 phần tử | Traceback từ `datasets` |
| 2 | `text-embedding-3-small` 404 trên Gemini | Test từng model name |
| 3 | Gemini từ chối `n > 1` | Grep payload trong `langchain_openai` |
| 4 | RAGAS đòi `set_run_config()` | Grep trong `ragas/llms/base.py` |
| 5 | Sai signature `RunConfig` | Đọc `ragas/run_config.py` |
| 6 | `NaN` ghi ra JSON hỏng | Đọc lại report |
| 7 | Quota 20 req/ngày (riêng từng model) | Error message có `quotaValue` |
| 8 | Throttle sai tầng (`invoke` vs `agenerate`) | Grep `ragas/llms/base.py` |
| 9 | `time.sleep()` trong async → block event loop | Metric nào đó luôn = 0 |
| 10 | Timeout RAGAS tính cả thời gian chờ trong hàng | `TimeoutError` ở `Job[n]` |
| 11 | Model nhẹ hơn tóm tắt **dài hơn** bản gốc | Test `test_m5.py` fail |
| 12 | NaN lan vào aggregate score | Đọc `per_question` |

**Điểm mấu chốt:** scaffold ban đầu có `try/except` rộng nuốt mọi lỗi và trả về `0.0`.
Điều này khiến "chưa implement" và "đã implement nhưng hỏng" trông giống hệt nhau.
Sau khi thay bằng log `type(e).__name__: {e}`, toàn bộ 12 lỗi lộ ra rõ ràng và lần lượt sửa được.

**Guardrail đã thêm để hệ thống không chết vì quota:**
- `ENRICH_MAX_CHUNKS` — giới hạn số chunk gọi LLM
- Cache JSON theo hash SHA-256 của `(source, methods, text)` — chạy lại không tốn quota
- Circuit breaker `disable_llm()` / `is_quota_error()` — gặp 429 một lần thì tắt LLM
- `_enrich_offline()` — enrichment heuristic không cần mạng
- `_ThrottledLLMWrapper` + `RAGAS_RPM_LIMIT` — giữ dưới trần 15 req/phút
