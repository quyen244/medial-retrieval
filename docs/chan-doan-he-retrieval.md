# Chẩn đoán hệ retrieval ViBioMIR

Cập nhật: 2026-10-23.
Bản Claude Doc tương ứng: https://claude.ai/artifact/PaBPrmdaBTCw4GxbR3DqHJ

## Tóm tắt

Nút thắt không nằm ở embedding model hay reranker.
Candidate pool hiện tại là 65 ứng viên mỗi câu, trong khi ground truth có khoảng 51 document mỗi câu.
Hệ thống đang cố tìm 51 tài liệu đúng bằng cách nhìn 65 ứng viên.

Điều đó giải thích vì sao 21 thí nghiệm ở hai vòng (augment, phân rã, giữ chỗ, quota, hybrid, fine-tune reranker) đều cho delta gần bằng 0: tất cả đều chạy bên trong cùng một cổ chai.
Reranker chỉ sắp xếp lại những gì đã có trong pool, nó không tạo ra recall.

Hai khoản lợi lớn nhất không cần GPU và không phụ thuộc gì cả: tách `relevant_docs` khỏi `relevant_chunks`, và nộp chunk dài hơn.
Ước lượng cộng lại đưa FINAL từ 0.0813 lên khoảng 0.12.
Sau đó mới đến việc mở rộng phễu, nơi trần lý thuyết nằm quanh 0.40.

## Số liệu thật

Leaderboard của D_ft cho phép dựng ngược gần như toàn bộ ground truth.
Trước hết, cách gộp điểm là trung bình đơn giản hai tầng: `(0.1293 + 0.0333) / 2 = 0.0813`, khớp tuyệt đối với FINAL SCORE.

Với `F2 = 5PR/(4P+R)`, `P = h/m`, `R = h/g`, rút gọn được dạng đóng:

```
F2 = 5h / (4g + m)
```

Trong đó `h` là số trúng, `g` là số gold, `m` là số nộp.
Số nộp `m` đo trực tiếp trên `data/tmp/sub1200/D_ft.json`; `P` và `R` lấy từ leaderboard; `h` và `g` suy ra.

| Tầng | Nộp m | Precision | Recall | Trúng h | Gold g | F2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Document | 35.1 | 0.1998 | 0.1368 | **7.0** | **51.3** | 0.1293 |
| Chunk | 40.0 | 0.0500 | 0.0367 | **2.0** | **54.5** | 0.0333 |

Ba hệ quả làm đổi toàn bộ chiến lược.

**Gold có khoảng 51 document mỗi câu, không phải 2-3.**
Đây không phải bài toán precision mà là bài toán vét recall trên một cụm tài liệu cùng chủ đề.
Tỷ lệ `g_chunk / g_doc = 1.06` cho thấy mỗi doc gold chỉ có đúng một chunk gold.

**Candidate pool nhỏ hơn gold set.**
Router lấy vi 35 / en 20 / zh 10 = 65 ứng viên rồi giữ 50.
Về mặt toán học không thể tìm 51 tài liệu đúng khi chỉ nhìn 65 ứng viên.

**Tầng chunk hỏng nặng hơn tầng doc.**
Khi đã trúng doc, chỉ 29% chunk vượt ngưỡng `LCS >= 0.40` (2.0 trên 7.0).
Chunk 600-1000 ký tự không đủ phủ chunk tham chiếu.

Chẩn đoán ban đầu đúng về hướng (top_k quá nhỏ, augment quá dài, filter ngôn ngữ không tốt) nhưng sai về thứ tự ưu tiên.
Embedding model và reranker không phải nút thắt, pool size và định dạng nộp mới là.

## Hai lỗ hổng chỉ tốn CPU

### `relevant_docs` đang bị khóa cứng vào `relevant_chunks`

`src/rag/pipeline/submission.py` sinh danh sách doc bằng cách lấy doc_id của các chunk đã nộp.
Hai danh sách được chấm bằng hai công thức độc lập nhưng đang bị tối ưu chung một tham số K.

Quy tắc biên rút từ dạng đóng: thêm doc thứ `m+1` có lợi khi `P(trúng) > h/(4g+m)`, tức `7/(205+35) = 2.9%`.
Ngưỡng rất thấp.
Điểm tối ưu `m* = 4gb/(1-b)` với `h(m) ~ m^b`:

| Độ dốc h(m) | m* tối ưu |
| --- | ---: |
| b = 0.25 | 68 doc |
| b = 0.30 | 88 doc |
| b = 0.40 | 137 doc |

Hiện đang nộp 35.
Lưu ý trung thực: nới `m` khi pool vẫn 50 thì gần như không lên, vì dump `D_ft_top50.jsonl` chỉ có 44.4 doc distinct mỗi câu.
Nới `m` chỉ có giá trị sau khi pool lớn.

### LCS là subsequence, mẫu số là độ dài tham chiếu

Đây là lỗ hổng lớn nhất hiện có.
Nếu chuỗi token của reference nằm trong chuỗi nộp theo đúng thứ tự (không cần liên tiếp) thì `LCS = len(ref)`, tỉ lệ bằng 1.0.
Token thừa không bị phạt, README đã ghi rõ điều này.

Chunk hiện tại có median 877 ký tự, pass rate có điều kiện 29%.
Đo trên 1.003.740 tài liệu đã crawl: tài liệu được retrieve có median 5.250 ký tự, p90 8.354.
Một chunk 877 ký tự vì thế chỉ phủ khoảng 17% tài liệu, hẹp hơn nhiều so với ước lượng 3.400 ký tự ban đầu.
Cách sửa: mỗi doc được chọn, nộp trọn markdown gốc (trần 8.000 ký tự) thay cho một chunk.
Vì `g_chunk / g_doc = 1.06`, nộp đúng một cửa sổ mỗi doc là cỡ chuẩn, và khi đó `relevant_docs` trùng luôn tập doc của chunk nên không có rủi ro nhất quán.

### Ước lượng tác động

Giả định containment hoạt động, `h_c` tăng từ 2.0 lên 7.0 bằng `h_d`, với `m_c = 44`.

| | Hiện tại | Sau khi sửa cả hai |
| --- | ---: | ---: |
| DOCS F2 | 0.1293 | ~0.130 |
| CHUNKS F2 | 0.0333 | ~0.115 |
| **FINAL** | **0.0813** | **~0.122** |

Nếu containment chỉ đạt một nửa (`h_c = 4`) thì FINAL khoảng 0.098.
Hướng vẫn dương trong mọi kịch bản.

## Bảy câu hỏi đối chiếu với tài liệu mentor

| # | Câu hỏi | Phán quyết | Lý do ngắn |
| --- | --- | --- | --- |
| 1 | Document store riêng | **Bắt buộc** | Đơn vị chấm điểm là document, hệ hiện tại không có biểu diễn doc nào |
| 1b | Phân cụm (k-means) | **Có, dưới dạng IVF + kNN expansion** | Gold là lân cận chủ đề; expansion là động cơ recall |
| 2 | Late chunking | **Hoãn** | Giải sai vấn đề, tốn nhiều ngày GPU re-encode 3.93M chunk |
| 3 | Dịch query + few-shot | **Có, nhưng tách lượt** | Đã triển khai sai dạng (concat thay vì multi-retrieval) |
| 4 | Reranker Qwen3 2B | **Đúng hướng, sai thời điểm** | Pool phải lớn trước; 0.6B trên pool lớn hơn 4B trên pool nhỏ |
| 5 | Metadata filtering | **Đang có hại, phải đổi cách dùng** | Filter ngôn ngữ áp prior chưa đo; title/heading chưa dùng |
| 6 | BM25 narrow down | **Có, hạ tầng đã sẵn nhưng chưa bật** | Kết luận "hybrid không đổi gì" vòng 1 không có giá trị |
| 7 | Không SFT, tập trung retrieval | **Đồng ý** | Toàn bộ vòng 2 chạy sau một cổ chai 65 ứng viên |

### 1. Document store và phân cụm

Hệ hiện tại không có biểu diễn document nào: một doc chỉ tồn tại như chunk nào tình cờ nổi lên.
Đúng như mentor slide 7, doc vector nên là `title + heading_path + phần mở đầu`.
Chi phí khoảng 1.05M doc x 1024-d fp16 = 2.1 GB, nằm gọn trong RAM, encode lại chỉ 1.05M đoạn ngắn chứ không phải 3.93M chunk.

Về phân cụm, đừng viết k-means riêng rồi coi nó là topic model.
`IndexIVFFlat` của Faiss với nlist khoảng 4000 vừa là ANN vừa là phân cụm.
Cái thực sự sinh lợi là **kNN expansion**: lấy 30 doc tin cậy nhất, kéo 20 láng giềng của từng doc trong không gian doc-embedding, chấm lại.
Một ranker thuần không làm được việc này vì nó chỉ xếp lại những gì đã có trong pool.

### 2. Late chunking

Late chunking giải quyết chuyện chunk mất ngữ cảnh tài liệu (đại từ, "thuốc này").
Lỗi hiện tại không phải vậy: pool 65 và chunk nộp quá ngắn.
Chỉ quay lại khi đã đo được rằng nút thắt nằm ở phân biệt chunk bên trong doc đã đúng.

### 3. Dịch query

Mentor đúng, thí nghiệm cũng đúng, hai bên không mâu thuẫn.
`multiquery.py` dùng `mode="concat"`: nhồi câu + thuật ngữ + bản dịch vào một chuỗi, làm một vector dense bị pha loãng.
B, B' và mix đều tệ hơn A, nhất quán.

Mentor nói là mỗi bản dịch một lượt truy xuất riêng, gộp bằng RRF.
Code đã có (`mode="lines"`) nhưng bị đóng băng vì "quá tốn".
Trong bài toán recall-dominated với g = 51, đó là thứ không được phép đóng băng: thêm một query chỉ là thêm một ANN search, rẻ hơn rerank rất nhiều.
Glossary dùng để **ràng buộc bản dịch**, tuyệt đối không nối thuật ngữ vào chuỗi query.

### 4. Reranker

Một rủi ro cần biết: fine-tune hiện tại dùng nhãn *utility* (chunk này trả lời được câu hỏi không, thang 0-3), trong khi gold trông giống *topical relevance* trên 51 doc.
Mục tiêu train có thể lệch với mục tiêu chấm.
Đây là giả thuyết, chưa verify, nhưng nó giải thích vì sao fine-tune không ăn.

Ngân sách: 1200 câu x 400 cặp = 480k cặp.
Trên 5060 Ti 16GB, Qwen3-Reranker-4B không khả thi trong thời gian nộp bài.
Chọn 0.6B hoặc giữ bge-reranker-v2-m3 nhưng chạy trên pool gấp 10 lần.

### 5. Metadata

Cách dùng hiện tại (hard filter `language` + quota cố định vi 26 / en 16 / zh 8) đang có hại.
Nó áp một prior chưa bao giờ được đo, và nó sống tới tận bước rerank nên không sửa được ở hạ nguồn.
Câu nào bằng chứng toàn tiếng Việt vẫn bị ép nhận 8 chunk tiếng Trung.

Cách dùng đúng: `title` và `heading_path` trở thành nội dung của doc vector (đã lưu sẵn, chưa dùng); `domain` là feature xếp hạng chứ không phải filter; `language` bỏ filter, chỉ dùng như ràng buộc đa dạng sau khi đo thấy một ngôn ngữ nuốt hết top-k.

### 6. BM25 và hybrid

Sparse vector BGE-M3 đã nằm trong Qdrant với `Modifier.IDF`, `hybrid_search()` đã viết xong, nhưng default vẫn là dense-only.

Kết luận "hybrid không đổi gì" ở vòng 1 không có giá trị vì nó được đo trên query B' đã augment, K = 20, pool 65.
Trong chế độ đó không gì có thể hiện ra.
Lý do mentor nêu là tốc độ; lý do ở đây phải là recall, vì query y khoa tiếng Việt chứa tên bệnh và tên thuốc chính xác, đúng chỗ lexical thắng dense.

## Thiết kế lại phễu

Phễu hiện tại và phễu mentor lệch nhau một đến hai bậc độ lớn.

```
HIỆN TẠI :   65 ứng viên ->   50 giữ    ->  35 doc nộp
MENTOR   : 2000 doc      ->  400 doc    -> 3200 passage -> rerank 400
ĐỀ XUẤT  : 6000 doc      ->  800 doc    -> kNN expansion -> rerank 800 -> ~100 nộp

                 gold = 51 doc mỗi câu
```

Toàn bộ chặng đường của pipeline hiện tại nằm trong dải 35-65, vắt qua đường gold chứ không bao giờ vượt xa nó.
Hai phễu còn lại bắt đầu ở phía bên phải của gold một đến hai bậc.

**Phase 0 - khai thác metric, không dùng GPU, 1 ngày.**
Tách `relevant_docs` khỏi `relevant_chunks`; nộp cửa sổ gộp 2500-4000 ký tự thay cho chunk 877 ký tự.
Lắp lại offline từ `D_ft_top50.jsonl` cộng việc đọc chunk liền kề từ Qdrant.
Làm trước tiên vì nó không phụ thuộc gì cả.

**Phase 1 - document store và phễu rộng, 2-3 ngày.**
Doc index 1.05M đoạn `title + heading_path + lead`.
Bỏ filter ngôn ngữ.
Ba query variant nhân hai nhánh dense và sparse, RRF sáu danh sách, ra 2000-6000 doc, chấm exact dot-product bằng NumPy, còn 800 doc.

**Phase 2 - kNN expansion, 1-2 ngày.**
Đây là nguồn recall thật.
Top 30 doc kéo 20 láng giềng mỗi doc trong doc-space, hợp vào pool, chấm lại.
Nhắm đưa `h` từ 7 lên 25.

**Phase 3 - reranker.**
Chỉ sau Phase 1 và 2.
Lúc đó một model 0.6B trên 800 ứng viên sẽ ăn, vì đã có cái để xếp.

Trần lý thuyết: nếu `h` đạt 25 trên 51 ở `m = 100` thì `F2_doc = 125/305 = 0.41` và `F2_chunk` khoảng 0.39, cho FINAL khoảng **0.40**, tức gấp năm lần hiện tại.

## Việc cần quyết và giả định

- [ ] Xác nhận từ thể lệ: có giới hạn độ dài `chunk_text` hoặc số phần tử `relevant_docs` không.
      Toàn bộ Phase 0 dựa trên giả định không có.
      Phương án an toàn nếu chưa rõ: cửa sổ 2500 ký tự và 80 doc.
- [ ] Quyết định thứ tự: chạy Phase 0 trước để lấy một số đo leaderboard mới, hay làm song song với Phase 1.

Các con số cần đọc đúng mức độ chắc chắn.

| Loại | Nội dung |
| --- | --- |
| Đo trực tiếp | P, R, F2 từ leaderboard; `m` = 35.1 doc và 40 chunk mỗi câu, đếm trên `D_ft.json`; chunk median 877 ký tự; 44.4 doc distinct trong dump top-50; markdown gốc của doc được retrieve median 5.250 ký tự; 100% doc_id trong dump có bản gốc; chunk căn chỉnh được vào bản gốc 97,5% |
| Suy ra | `h` = 7.0 doc và 2.0 chunk; `g` = 51.3 doc và 54.5 chunk. Suy từ `P = h/m` và `R = h/g`, có sai số nhỏ vì leaderboard là macro-average |
| Giả định chưa verify | Containment của cửa sổ dài sẽ đẩy `h_c` lên bằng `h_d`; gold là topical relevance chứ không phải answer-bearing; độ dốc `h(m) ~ m^0.3` |
| Số đề xuất | 6.000 / 800 / 100 ở Phase 1-2 là mục tiêu thiết kế, chưa đo |
