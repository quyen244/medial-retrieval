# Bước 4-5: lấy ứng viên, chấm utility hai tầng

**Trạng thái: THIẾT KẾ, CHƯA CODE.** Viết doc trước để thống nhất input/output và
luồng dữ liệu trước khi bắt tay viết `src/rag/labeling/candidates.py` và
`src/rag/labeling/judge.py`.

## Tổng quan

| Bước | Input | Output | Cần LLM |
|---|---|---|---|
| 4 | ~6.000 query giữ lại (từ bước 3) | mỗi query kèm 50 ứng viên chưa gắn nhãn (EN 30 · VI 15 · ZH 5) | không |
| 5 | query + 50 ứng viên | mỗi ứng viên có điểm utility 0-3 | **có**, 2 tầng |

```mermaid
flowchart LR
    A["gen_queries_kept.jsonl<br/>~6.000 query"] --> B["BƯỚC 4<br/>lấy ứng viên theo ngôn ngữ"]
    C[("Qdrant<br/>corpus")] --> B
    B --> D["query + 50 ứng viên<br/>EN 30 · VI 15 · ZH 5 · ẩn danh, đã xáo"]
    D --> E["BƯỚC 5a<br/>tầng 1: CE rẻ"]
    E --> F["50 ứng viên có t1_score"]
    F --> G["BƯỚC 5b<br/>tầng 2: LLM đắt<br/>chỉ 15/50 (10 cao nhất + 5 quanh median)"]
    G --> H["judged.jsonl<br/>mỗi ứng viên có t2_score"]
```

---

## Bước 4 - Lấy ứng viên

**Input:** `data/labeling/gen_queries_kept.jsonl` (~6.000 dòng, mỗi dòng có `query`, `src_chunk_id`, `src_language`).
**Output:** `data/labeling/candidates.jsonl` — mỗi dòng 1 query kèm 50 ứng viên (EN 25 · VI 15 · ZH 10).

```mermaid
flowchart TD
    A["1 query từ gen_queries_kept"] --> B["3 lần hybrid_search riêng,<br/>lọc theo ngôn ngữ"]
    B --> B1["EN top_k=25"]
    B --> B2["VI top_k=15"]
    B --> B3["ZH top_k=10"]
    B1 --> C{"src_chunk_id có trong<br/>nhóm NGÔN NGỮ CỦA NÓ không?"}
    B2 --> C
    B3 --> C
    C -- có --> D["giữ nguyên nhóm đó"]
    C -- không --> E["chèn src_chunk_id vào đúng nhóm ngôn ngữ của nó,<br/>bỏ 1 cái điểm thấp nhất CÙNG NHÓM"]
    D --> F["gộp 3 nhóm = 50 ứng viên"]
    E --> F
    F --> G["xáo thứ tự cả 50"]
    G --> H["ghi ra, chunk gốc KHÔNG mang nhãn gì khác biệt"]
```

**Vì sao chia theo ngôn ngữ, không lấy top-50 thô trên toàn corpus:** corpus
lệch (en 99,8%, vi và zh thấp hơn), nên top-k thô sẽ bị en áp đảo tự nhiên —
ứng viên vi/zh dễ không lọt vào pool dù mục đích chính là dạy reranker so
khớp xuyên ngôn ngữ cho **cả 3**, không chỉ en. `candidate_lang_quota`
(50/30/20, áp lên 50 → EN 25 / VI 15 / ZH 10) đảm bảo mọi query đều có đủ 3
ngôn ngữ để chấm. Tỷ lệ zh nâng từ 10% lên 20% so với lần chốt trước — corpus
zh đã crawl lên ~30% (trước đó 4,1%), không còn cần dè sẻn slot như lúc thưa.

**Vì sao luôn đưa `src_chunk_id` vào pool dù nó không nằm sẵn trong top-k của
nhóm nó:** query được sinh ra TỪ chunk đó, nên nó luôn là ứng viên hợp lệ về
mặt logic — "liệu chunk sinh ra câu hỏi này có tự trả lời được câu hỏi của
chính nó không" là câu cần tầng 2 chấm, không phải câu cần bước 4 quyết định
trước. Nhưng nó **không được gắn nhãn positive trước**, không được đặt ở vị
trí cố định, và bị xáo lẫn vào 50 ứng viên như mọi chunk khác. Luật chèn áp
**đúng nhóm ngôn ngữ của chunk gốc** — ví dụ chunk gốc là `en` thì chỉ
kiểm tra/chèn trong nhóm EN (25 cái), không đụng vào nhóm vi/zh.

**Đây là điểm khác quan trọng nhất so với cách làm thông thường** (nhắc lại từ
spec gốc): nếu mặc định chunk gốc = positive, reranker học được đúng một việc —
đếm độ trùng từ vựng với chunk nguồn, vì query được viết ra từ chính chunk đó
nên luôn mượn thuật ngữ của nó. Trộn ẩn danh buộc tầng 5 phải tự đánh giá nó
dựa trên nội dung, không dựa trên xuất xứ.

**Theo dõi sau pilot:** nếu phần lớn ứng viên ZH vẫn luôn bị chấm 0 dù corpus
đã khá hơn, reranker có thể học đường tắt "thấy tiếng Trung → chấm thấp" thay
vì đọc nội dung — đo bằng `t2_positive_rate_by_language` ở bước 5.

**Hard negative cùng chương ICD (tuỳ chọn, TẮT mặc định):** cắm thêm 2 ứng
viên lấy từ bệnh cùng chương ICD nhưng khác mã với bệnh trong query, dùng bảng
`icd_bilingual.tsv` đã có. Việc này **đợi sau khi pilot của bước 4-5 chạy ổn**
mới bật — thêm biến số sớm làm khó debug nếu keep_rate hay tỷ lệ positive lệch
bất thường.

Một dòng output:

```json
{"qid": "g000123_1", "query": "...",
 "candidates": [
   {"chunk_id": "...", "language": "en", "content": "...", "is_src": false},
   {"chunk_id": "src_chunk_cua_qid_nay", "language": "en", "content": "...", "is_src": true},
   ...48 cái nữa...
 ]}
```

`is_src` được ghi lại **chỉ để audit sau này** (ví dụ: đo xem chunk gốc thắng
hay thua bao nhiêu % thời gian ở tầng 5), **không được đưa vào prompt chấm
điểm ở bước 5** — model chấm không được biết cái nào là chunk gốc.

---

## Bước 5 - Chấm utility hai tầng

**Input:** `candidates.jsonl` từ bước 4.
**Output:** `data/labeling/judged.jsonl`, mỗi ứng viên có `t1_score` và (một phần) `t2_score`.

### 5a - Tầng 1: cross-encoder rẻ

```mermaid
flowchart TD
    A["50 ứng viên của 1 query"] --> B["bge-reranker-v2-m3<br/>chấm (query, content) từng cặp"]
    B --> C["50 ứng viên có t1_score liên tục"]
    C --> D["xếp hạng theo t1_score giảm dần"]
```

Model: `BAAI/bge-reranker-v2-m3` (cùng backbone với reranker đang fine-tune,
nhưng dùng checkpoint gốc chưa train — chấm thô để lọc, không phải model
cuối cùng). Chạy trên GPU local, không cần vLLM, rẻ: batch cả 16 cặp trong
một lần forward.

**Tại sao cần tầng 1 trước khi gọi LLM:** LLM 8B là phần đắt nhất của toàn bộ
pipeline sinh nhãn. Tầng 1 lọc bớt các ứng viên rõ ràng không liên quan (điểm
CE rất thấp) để tầng 2 chỉ phải xử lý phần mơ hồ - đúng/sai chưa rõ.

### 5b - Tầng 2: LLM, chỉ 15/50

```mermaid
flowchart TD
    A["50 ứng viên đã xếp hạng theo t1_score"] --> B["chọn 10 điểm CAO nhất<br/>+ 5 quanh NGƯỠNG MƠ HỒ"]
    B --> C["15 ứng viên cần LLM chấm"]
    C --> D["build 1 prompt:<br/>instruction + query<br/>+ 15 tài liệu đánh số"]
    D --> E["gọi Qwen3-8B<br/>1 lần cho cả 15"]
    E --> F{"parse JSON<br/>15 điểm 0-3"}
    F -- hỏng --> G["retry, tối đa 3 lần"]
    F -- được --> H["gán t2_score cho 15 cái"]
    I["35 ứng viên còn lại<br/>t1_score quá thấp"] --> J["suy ra t2_score = 0<br/>KHÔNG gọi LLM"]
    H --> K["gộp lại đủ 50, ghi file"]
    J --> K
```

**"10 cao nhất + 5 quanh ngưỡng mơ hồ" nghĩa là gì cụ thể:** sau khi tầng 1 xếp
hạng 50 ứng viên theo `t1_score`, chọn vị trí 1-10 (chắc chắn liên quan nhất)
cộng 5 ứng viên nằm quanh điểm median của toàn bộ 50 — đó là vùng CE "lưỡng
lự", không rõ liên quan hay không, cần LLM phân xử. 35 ứng viên còn lại (điểm
thấp nhất) **không gọi LLM**, gán thẳng `t2_score = 0`. Số 10+5 là lựa chọn
cố định, đơn giản hơn ngưỡng tương đối theo phân phối điểm — cần soát ở cổng
chặn việc 6 (`TODO.md`) xem có hay bỏ lỡ ứng viên cạnh tranh ngay ngoài biên,
hoặc có đang lố so với cần thiết.

**Vì sao gộp cả 15 tài liệu vào MỘT lời gọi, không gọi 15 lần riêng:** phần
`instruction + query` giống hệt nhau cho cả 15 tài liệu của cùng 1 query. Gộp
vào 1 prompt (tài liệu đánh số 1-15, model trả JSON 15 điểm) tận dụng được
prefix caching của vLLM — phần đầu prompt chỉ nạp 1 lần thay vì 15 lần. Đây là
cơ chế duy nhất khiến ngân sách bước này khả thi (xem mục 6 của
`docs/pipeline-sinh-nhan-reranker.md`, đã cập nhật lại số giờ theo quy mô mới).

**Thang điểm, nhắc lại từ spec:**

| Điểm | Nghĩa | Thành |
|---|---|---|
| 3 | Trả lời trực tiếp và đầy đủ | positive, trọng số 1,0 |
| 2 | Có bằng chứng hữu ích nhưng chưa đủ | positive, trọng số 0,6 |
| 1 | Cùng chủ đề, không trả lời được | **bỏ khỏi tập train**, không phải negative |
| 0 | Không liên quan | negative |

**Điểm 1 không dùng làm negative** vì corpus 1,22 triệu tài liệu có hàng chục
bản ghi đúng cho một câu hỏi — ép điểm 1 thành negative là tự tạo negative giả.

**Loại cả query:** nếu không ứng viên nào trong 16 đạt điểm ≥2 (kể cả sau tầng
2), bỏ hẳn query đó khỏi tập train — không có bằng chứng nào để học từ nó.

Một dòng output:

```json
{"qid": "g000123_1", "query": "...",
 "candidates": [
   {"chunk_id": "...", "language": "en", "t1_score": 0.91, "t2_score": 3, "weight": 1.0, "is_src": false},
   {"chunk_id": "...", "language": "vi", "t1_score": 0.44, "t2_score": 1, "weight": null, "is_src": false},
   {"chunk_id": "...", "language": "en", "t1_score": 0.08, "t2_score": 0, "weight": 0.0, "is_src": true}
 ]}
```

---

## MLflow ghi lại gì

Một lần chạy = một run, experiment gợi ý `labeling_judging`.

**Params:** `LabelingConfig`, `prompt_judge_tier2_uri` + version.

**Metrics:**

| Bước | Chỉ số |
|---|---|
| 4 | `b4_n_query`, `b4_src_missing_from_pool` (tần suất phải chèn thủ công, theo từng nhóm ngôn ngữ) |
| 5a | `b5a_giay`, phân phối `t1_score` (p10/p50/p90) |
| 5b | `b5b_n_llm_call`, `b5b_json_fail`, phân phối điểm theo mức 0/1/2/3 |
| cuối | `ty_le_query_bi_loai` (không có ứng viên ≥2), `ty_le_src_thang` (chunk gốc đạt ≥2, dùng để audit rò rỉ xuất xứ), `t2_positive_rate_by_language` (tỷ lệ ứng viên đạt ≥2 trong số ứng viên đã chấm, tách riêng en/vi/zh) |

`ty_le_src_thang` là chỉ số đáng theo dõi nhất: nếu chunk gốc thắng (đạt điểm
cao) gần 100% số query, nghĩa là cơ chế trộn ẩn danh không thực sự phá được
lợi thế trùng từ vựng — cần xem lại prompt bước 2 có đang để lộ thuật ngữ quá
rõ không.

`t2_positive_rate_by_language` là chỉ số thứ hai cần theo dõi: nếu tỷ lệ của
zh gần 0 qua hàng nghìn mẫu dù corpus đã lên ~30%, đó là dấu hiệu reranker có
thể đang học đường tắt "thấy tiếng Trung → chấm thấp" thay vì đọc nội dung,
và cần xem lại hạn ngạch `candidate_lang_quota`.

**Artifacts:** mẫu 30-50 bản ghi `judged.jsonl` để soát tay — đây là cổng chặn
số 2 trong `TODO.md`.

---

## Việc cần quyết trước khi code

1. **Prompt tầng 2** — còn mở, cùng nguyên tắc như bước 2: bạn viết ở
   `prompts/judge_tier2.txt`, tôi chỉ load và validate biến. Biến dự kiến:
   `{{query}}`, `{{documents}}` (15 tài liệu đã đánh số, format JSON hoặc
   markdown list - cần bạn chọn), yêu cầu output JSON `{"scores": [3,0,2,...]}`
   theo đúng thứ tự đánh số.

**Đã chốt (trước đây là việc 2 và 3, nay đã có số):**

- `candidates_per_query = 50`, chia theo `candidate_lang_quota = {en:0.50, vi:0.30, zh:0.20}` → EN 25 / VI 15 / ZH 10.
- Chọn LLM chấm theo rank cố định: `tier2_top_n = 10` + `tier2_median_n = 5` = 15/50. Không dùng ngưỡng tương đối theo phân phối điểm (đã cân nhắc, chọn đơn giản hơn) — nếu sau pilot thấy 15 lố hoặc thiếu, xem metric gợi ý ở mục trên trước khi đổi số.
- `judge_temperature = 0.0`, tách khỏi `llm_temperature = 0.7` dùng cho sinh query — tầng 2 cần nhất quán, không cần đa dạng.

Cả 3 đã có trong `src/rag/labeling/config.py`.
