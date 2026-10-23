# Kế hoạch Phase 1 và Phase 2

Lập ngày 2026-10-23, sau khi Phase 0 kết thúc ở FINAL 0.1112.
Hạn nộp 31/10 và 04/11, tức còn khoảng 8 ngày.

## 1. Mốc đo hiện tại

Mọi con số dưới đây là ĐO, không phải ước lượng. Nguồn: `res/*/scores.txt`, xem `ket_qua_thi_nghiem.txt` vòng 3.

| | Giá trị | Ý nghĩa |
|---|---:|---|
| FINAL tốt nhất | 0.1112 | N3_d35_c44 |
| h_doc | 6,93 | tìm đúng gần 7 trong ~51 gold doc mỗi câu |
| Doc recall | 13,7% | |
| h_chunk / h_doc | 0,79 | ổn định qua MỌI cấu hình đã thử |
| m_doc tối ưu | ~35 | đỉnh đo được, hai bên đều thấp hơn |
| m_chunk tối ưu | ~43 (hết pool) | biên 4,21% còn trên ngưỡng 2,15% |
| Trần cửa sổ chunk | 8.000 ký tự | 2.500 thấp hơn 2%, 20.000 tụt lại |

### Trần của pool hiện tại đã được đo, không còn là giả định

Bài E1 nộp TOÀN BỘ 44 doc có trong pool và chỉ trúng 7,06.
Nghĩa là pool chứa đúng khoảng 7 gold doc; 44 trong 51 gold doc mỗi câu nằm HOÀN TOÀN ngoài pool.
Mọi cách rerank, xếp lại hay chỉnh số lượng nộp bên trong pool này đều bị chặn cứng ở recall 13,9%.

Đây là lý do Phase 1 tồn tại, và nó là lý do đã được chứng minh chứ không phải suy đoán.

### Tầng chunk đã hết là vấn đề riêng

`h_chunk / h_doc = 0,79` giữ nguyên ở cả E2, E3, E4, N3, N5 - tức ở mọi độ dài cửa sổ và mọi số doc.
Nó đã trở thành một hệ số nhân cố định trên kết quả tầng doc.
Phase 1 và 2 vì thế chỉ cần đuổi theo MỘT đại lượng duy nhất là `h_doc`.

## 2. Mô hình lợi ích

Giữ `m_doc = 35`, `m_chunk = 43`, `h_c/h_d = 0,79`, `g_doc = 51`, `g_chunk = 53`:

| h_doc | Doc recall | F2_doc | F2_chunk | FINAL |
|---:|---:|---:|---:|---:|
| 7,1 | 13,9% | 0.148 | 0.109 | **0.128** |
| 10 | 19,6% | 0.210 | 0.154 | **0.182** |
| 15 | 29,5% | 0.314 | 0.231 | **0.273** |
| 20 | 39,3% | 0.419 | 0.308 | **0.364** |
| 25 | 49,1% | 0.524 | 0.385 | **0.455** |

Mỗi gold doc tìm thêm được đáng khoảng **+0.018 FINAL**.
Đó là thước đo duy nhất cần nhìn từ giờ.

Lưu ý: `m_doc = 35` là tối ưu cho `h = 7`. Khi `h` tăng thì `m*` cũng tăng, nên phải chỉnh lại số doc nộp sau mỗi lần `h` nhảy bậc.

## 3. Phase 1 - mở phễu

### 1.0 Thăm dò ngôn ngữ (0 GPU, chạy song song)

Phân bố ngôn ngữ trong pool hiện tại: vi 48% / en 34% / zh 17%.
Sau rerank thành vi 59% / en 22% / zh 18%, tức reranker đang đẩy vi lên và dìm en xuống.
Chưa biết 7 gold doc tìm được thuộc ngôn ngữ nào.

Cách đo: lắp offline từ dump, nộp CHỈ các doc en và zh của pool (khoảng 23 doc).
`h_enzh` đọc được trực tiếp; `h_vi` xấp xỉ `7,06 - h_enzh`.

Quyết định điều gì: có đáng đầu tư dịch query sang en/zh không, và có đáng crawl thêm tiếng Trung không.
Tốn 1 lượt nộp, không tốn GPU, chạy được trong lúc GPU bận việc 1.1.

### 1.1 Truy xuất sâu - THÍ NGHIỆM QUYẾT ĐỊNH

Đây là việc quan trọng nhất trong cả kế hoạch. Mọi thứ sau nó phụ thuộc vào kết quả của nó.

Thay đổi so với hiện tại:

| | Hiện tại | Sau |
|---|---|---|
| Lọc ngôn ngữ | vi/en/zh riêng, quota cố định | bỏ hẳn, một nhánh phẳng |
| Chiều sâu | 65 chunk | 3.000 chunk |
| Gộp về doc | không có | max điểm chunk mỗi doc |
| Rerank | 65 chunk | chunk tốt nhất của 400 doc |
| Nộp | 35 doc + 43 cửa sổ | giữ nguyên |

Ba chi tiết triển khai dễ làm hỏng nếu bỏ qua:

1. **`with_payload=["doc_id"]`** ở lượt tìm sâu. Lấy đủ payload cho 3.000 chunk là khoảng 2,7 MB mỗi câu, nhân 1.200 câu thành 3,2 GB truyền qua mạng. Chỉ lấy `doc_id` + điểm, gộp về doc, rồi mới `retrieve()` payload cho đúng 400 doc giữ lại.
2. **Nâng `hnsw_ef`** trong `search_params`. HNSW với `limit=3.000` mà `ef` mặc định thì recall ở đuôi rất tệ, lấy sâu mà không nâng `ef` là lấy sâu giả.
3. **`max_length=512`** cho lượt rerank đầu. Giảm một nửa chi phí; chunk tốt nhất của mỗi doc vốn đã ngắn hơn 1024 token.

Chi phí ước tính: truy xuất 20-40 phút, rerank 400 x 1.200 = 480k cặp khoảng 2 giờ.

**Cách đọc kết quả:**

- `h_doc` > 7,06 → pool đúng là nút thắt. Đi tiếp 1.2, 1.3, Phase 2.
- `h_doc` ≈ 7,06 → nút thắt là CÁCH XẾP HẠNG, không phải độ rộng. Bỏ Phase 2 như đang thiết kế, chuyển sang 1.4.

### 1.2 Hybrid ở chiều sâu

Sparse vector BGE-M3 đã nằm sẵn trong Qdrant với `Modifier.IDF`, hàm `hybrid_search()` đã viết xong, chỉ chưa bật.
Prefetch 3.000 dense + 3.000 sparse, RRF phía server.

Kết luận "hybrid không đổi gì" ở vòng 1 KHÔNG có giá trị: nó đo trên query đã augment, K=20, pool 65.
Trong chế độ đó không gì có thể hiện ra.

Lý do kỳ vọng hybrid ăn ở đây: query y khoa tiếng Việt chứa tên bệnh và tên thuốc chính xác, đúng chỗ lexical thắng dense.

### 1.3 Đa truy vấn kèm bản dịch

Ba query độc lập (vi gốc, en, zh), mỗi query một lượt dense và một lượt sparse, RRF sáu danh sách.
Chi tiết phần tri thức dịch thuật ở mục 5.

Ràng buộc học được bằng giá đắt ở vòng 1: **tuyệt đối không nối thuật ngữ và bản dịch vào cùng một chuỗi query**.
B, B' và mọi biến thể mix đều thấp hơn A. Một vector dense bị pha loãng khi nhồi nhiều thứ vào.
Mỗi bản dịch phải là một lượt truy xuất riêng, gộp bằng RRF.

### 1.4 Đường lui nếu 1.1 cho kết quả âm

Nếu mở phễu mà `h_doc` không nhúc nhích thì vấn đề nằm ở biểu diễn và xếp hạng. Theo thứ tự chi phí tăng dần:

1. Doc-level index: `title + heading_path + phần mở đầu`, 1,0M doc x 1024-d fp16 = 2,1 GB, encode lại khoảng 1 giờ GPU. Biểu diễn khác thì thứ hạng khác.
2. Reranker mạnh hơn (Qwen3-Reranker-0.6B) chạy trên pool sâu.
3. Đổi encoder.

## 4. Phase 2 - mở rộng theo láng giềng

Lý do: gold khoảng 51 doc mỗi câu là một CỤM CHỦ ĐỀ, không phải vài bài trả lời đúng câu hỏi.
Một ranker query-document không bao giờ xếp hạng cao các thành viên còn lại của cụm, vì chúng không trực tiếp trả lời câu hỏi.
Mở rộng theo láng giềng đi tới được chỗ đó.

Thiết kế rẻ nhất, không cần embedding mới:

1. Sau rerank của 1.1, lấy 30 doc đứng đầu.
2. Lấy `point_id` của chunk tốt nhất mỗi doc.
3. Gọi Qdrant `recommend(positive=[các id đó], limit=...)` - nó dùng vector đã lưu, không phải encode lại.
4. Gộp chunk mới về doc, loại doc đã có, rerank phần mới.
5. Trộn vào danh sách cũ theo điểm rerank.

Chi phí: thêm một lời gọi Qdrant mỗi câu, cộng rerank khoảng 300 ứng viên mới.

Đo: `h_doc` trước và sau khi mở rộng, cùng `m_doc`.

Phase 2b nếu còn thời gian và 2a có hiệu quả: dựng doc-level index để mở rộng trong không gian document thay vì không gian chunk. Sạch hơn nhưng tốn 1 giờ GPU encode.

## 5. Tri thức dịch thuật vi / en / zh

### 5.1 Những gì đã có trong repo

| Nguồn | Quy mô | Ngôn ngữ |
|---|---:|---|
| `data/vocab/icd_bilingual.tsv` | 10.754 dòng → 17.293 bề mặt tiếng Việt | vi ↔ en, cầu nối là mã ICD |
| `data/knowledge/rxnorm_concepts_final.csv` | 229.135 khái niệm thuốc | en |
| `data/knowledge/icd10_vn_aliases_v1.csv` | 12.922 | vi |
| `data/knowledge/icd10cm_2027.csv` | 74.880 | en |

Tiếng Trung: KHÔNG CÓ GÌ.

### 5.2 Độ phủ đã đo

477 trong 1.200 query thật (39,8%) khớp ít nhất một bề mặt tiếng Việt của bảng ICD.
Phân bố: 723 query không khớp gì, 385 khớp 1 thuật ngữ, 70 khớp 2, 22 khớp từ 3 trở lên.

### 5.3 Vấn đề chất lượng - phải sửa trước khi dùng

Khớp chuỗi con cho ra thuật ngữ sai. Ví dụ đo được:

```
"xung huyết"  ->  "Congestion and hemorrhage of prostate"
```

Một bề mặt tiếng Việt chung chung bị ánh xạ sang thuật ngữ tiếng Anh gắn chặt với một cơ quan cụ thể.
Nhét thứ này vào prompt dịch là chủ động làm hỏng query, đúng cảnh báo ở slide 13 của mentor.

Wikidata có đúng loại nhiễu đó. Thử 7 mã ICD:

```
E11 -> 2型糖尿病        đúng
N20 -> 腎結石           đúng
A09 -> 腹瀉 / 旅行者腹瀉 / 小腸炎   ba mục, mục đầu đúng
K29 -> 胰酶 (pancreatin)           SAI, K29 là viêm dạ dày
```

Một mã ICD ứng với nhiều mục Wikidata, có mục gán sai.

Bốn bước lọc bắt buộc nếu dùng glossary:

1. Khớp theo RANH GIỚI TỪ, không phải chuỗi con.
2. Bỏ bề mặt tiếng Việt ngắn (dưới ~8 ký tự) mà ánh xạ sang thuật ngữ tiếng Anh có thêm định tố cơ quan không xuất hiện ở phía tiếng Việt.
3. Khớp dài thắng, không chồng lấn.
4. Giữ một cột độ tin cậy, chỉ đưa thuật ngữ tin cậy cao vào prompt.

### 5.4 Thứ tự làm, đã đổi so với dự định ban đầu

Vì độ phủ chỉ 39,8% và nhiễu ánh xạ là có thật, glossary KHÔNG nên là bước đầu.

**Bước 1 - dịch trực tiếp bằng Qwen3-8B.**
Độ phủ 100%, không có lỗi ánh xạ, chính là Baseline ở slide 15 của mentor.
1.200 query x 2 ngôn ngữ, chạy một lần, cache ra file.

**Bước 2 - glossary làm RÀNG BUỘC, chỉ cho 39,8% query khớp được.**
Đưa cặp đã xác minh vào prompt: "các thuật ngữ sau đã kiểm chứng, giữ nguyên: viêm dạ dày = Gastritis".
Đây là glossary-augmented prompting ở slide 14 mục 2, không phải nối thuật ngữ vào query.

**Bước 3 - tiếng Trung qua cầu mã ICD, chỉ làm nếu 1.0 cho thấy zh đáng giá.**
SPARQL Wikidata trên `P494`, lọc theo bốn bước ở 5.3, đối chiếu chéo với nhãn tiếng Anh đã có để loại mục sai như `K29 -> 胰酶`.
Phần không phủ được thì để Qwen3-8B sinh, có ràng buộc bằng thuật ngữ tiếng Anh đã xác minh.

**Thuốc:** RxNorm chỉ có tiếng Anh. Tên thuốc tiếng Việt phần lớn là phiên âm INN (paracetamol, amoxicillin) nên phía tiếng Anh gần như miễn phí. Tiếng Trung khác hẳn về cấu tạo (对乙酰氨基酚); chỉ làm nếu 1.0 cho thấy zh đáng giá.

### 5.5 Lưu ý hạ tầng

Máy này có SSL cert store hết hạn: mọi lời gọi HTTPS bằng Python đều báo `CERTIFICATE_VERIFY_FAILED`.
Wikidata chạy được khi tắt verify, nhưng nên sửa gốc (`pip install -U certifi`) thay vì tắt verify trong code chính thức.

## 6. Lịch

| Ngày | Việc | Đầu ra |
|---|---|---|
| D1 (23/10) | Dựng Qdrant lại, nộp thăm dò 1.0, chạy 1.1 | `h_doc` ở pool 3.000 |
| D2 | Đọc 1.1. Nếu dương thì bật hybrid 1.2, chỉnh lại `m_doc` | |
| D3 | Dịch trực tiếp bằng Qwen3-8B, đa truy vấn 1.3 | |
| D4-D5 | Phase 2 mở rộng láng giềng | |
| D6 | Chỉnh lại `m_doc`, `m_chunk`, trần cửa sổ ở mức `h` mới | |
| D7-D8 | Chốt cấu hình tốt nhất, dự phòng | |

## 7. Rủi ro

- **Docker và Qdrant đang tắt.** Phải dựng lại trước mọi việc. Collection `medical_documents` nằm trong volume `qdrant_data`, chưa kiểm tra còn nguyên không.
- **Hạn mức nộp bài mỗi ngày chưa biết.** Cần nắm trước khi lên lịch đo.
- **1.1 cho kết quả âm sẽ vô hiệu hoá Phase 2 như đang thiết kế.** Đường lui ở 1.4.
- **Chưa có dev set.** `data/labeling/dev_set.jsonl` có 100 query x ~30 ứng viên nhưng `human_score` toàn None. Leaderboard vẫn là công cụ đo duy nhất, nên mỗi thay đổi phải cô lập đúng một biến.
