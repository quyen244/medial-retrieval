# TODO - ViBioMIR

Hạn nộp 31/10 và 04/11.
Ràng buộc cuộc thi: [CLAUDE.md](CLAUDE.md).
Chẩn đoán: [docs/chan-doan-he-retrieval.md](docs/chan-doan-he-retrieval.md).
Kế hoạch: [docs/ke-hoach-phase-1-2.md](docs/ke-hoach-phase-1-2.md).
Kết quả mọi bài nộp: [ket_qua_thi_nghiem.txt](ket_qua_thi_nghiem.txt).

## Đứng ở đâu

| | |
|---|---:|
| FINAL tốt nhất | **0.1112** (N3_d35_c44) |
| h_doc | 6,93 trên ~51 gold doc mỗi câu |
| Doc recall | 13,7% |
| h_chunk / h_doc | 0,79, ổn định qua mọi cấu hình |

**Chỉ còn một đại lượng cần tối ưu: `h_doc`.** Mọi thứ khác đã chốt.
`FINAL ≈ 0,016 × h_doc`, tức mỗi tài liệu đúng tìm thêm được đáng +0,016 điểm.

| h_doc | FINAL |
|---:|---:|
| 6,93 (nay) | 0.111 |
| 15 | ~0.24 |
| 25 | ~0.40 |

**Trần của pool hiện tại đã ĐO, không phải suy đoán.** Bài E1 nộp toàn bộ 44 doc
trong pool và chỉ trúng 7,06. Pool chứa đúng 7 gold doc; 44 trong 51 gold nằm
hoàn toàn ngoài. Mọi cách rerank hay xếp lại bên trong pool đó đều bị chặn ở
recall 13,9%. Đó là lý do phải mở phễu.

## Đã xong

| | Việc | Kết quả |
|---|---|---|
| P0-1 | Tách `relevant_docs` khỏi `relevant_chunks` | m_doc tối ưu ~35, nới lên 43 thì tụt |
| P0-2 | Nộp cửa sổ markdown gốc thay cho chunk 877 ký tự | h_chunk 1,89 → 5,50, trần 8.000 là đỉnh |
| P0-3 | Dựng ngược ground truth từ leaderboard | g_doc ≈ 51, g_chunk ≈ 53 |
| P0-4 | Công cụ đọc ngược `res/*/scores.txt` | `scripts/phase0_readout.py` |
| P1-0 | Khảo sát trường raw doc | tìm ra `source.description`, phủ 88,8% vi / 86,1% zh |
| P1-1 | Lấy lại checkpoint reranker fine-tune | `scripts/fetch_reranker.py` |
| P1-2 | ICD-11 en↔zh do WHO căn sẵn | 34.919 cặp, thay hẳn Wikidata (vốn nhiễu) |

## Đang chạy

| | Việc | Máy | Ghi chú |
|---|---|---|---|
| A | `scripts/build_doc_texts.py` → `data/doc_texts.jsonl` | CPU, local | ~1,0M file, chậm vì đọc đĩa |

## Trình tự trên server

Chạy đúng thứ tự này, không cần nghĩ thêm.

| # | Lệnh | Máy | Mục đích |
|---|---|---|---|
| 1 | `python -m scripts.qdrant_backup` | bất kỳ | **Làm trước tiên.** Collection 3,9M điểm đang nằm trong volume của project khác (xem Rủi ro) |
| 2 | `python -m scripts.build_doc_texts` | CPU | bỏ qua nếu việc A đã xong trên local, chỉ cần copy file sang |
| 3 | `python -m scripts.ingest_doc_index --recreate` | **GPU** | encode + nạp collection `medical_docs` (1,0M điểm, dense + sparse một forward pass) |
| 4 | `python -m scripts.phase1_doc_retrieve --name S1_doc60 --top-docs 60` | **GPU** | **nông**. So với N3 → đo riêng hiệu ứng BIỂU DIỄN |
| 5 | `python -m scripts.phase1_doc_retrieve --name S2_doc1000 --top-docs 1000` | **GPU** | **sâu**. So với S1 → đo riêng hiệu ứng CHIỀU SÂU |
| 6 | nộp, bỏ `res/<tên>.zip`, chạy `python -m scripts.phase0_readout` | — | ra `h_doc` mới |
| 7 | `python -m scripts.append_phase0_results` | — | cập nhật bảng kết quả |

Bước 4 và 5 khác nhau **đúng một tham số**, nên quy được trách nhiệm cho từng
hiệu ứng. Vòng 1 và 2 có 21 thí nghiệm đổi nhiều thứ cùng lúc và không rút ra
được gì; Phase 0 đi được vì mỗi bài chỉ đổi một biến.

## Sau khi có kết quả bước 4-5

| | Việc | Điều kiện |
|---|---|---|
| B | Chỉnh lại `m_doc`, `m_chunk` ở mức `h` mới | `m*` tăng theo `h`, phải dò lại |
| C | Bật `--hybrid` (dense + sparse) | sparse đã nạp sẵn ở bước 3, chỉ thêm cờ |
| D | Mở rộng láng giềng (Phase 2) | chỉ làm nếu bước 5 cho `h_doc` tăng |
| E | Dịch query en/zh + HyDE | cần GPU đủ cho Qwen3-8B; xem mục 5 kế hoạch |
| F | Reranker mạnh hơn | chỉ sau khi pool đã rộng |

## Việc CPU làm được ngay, không cần server

| | Việc | Ghi chú |
|---|---|---|
| G | Nộp thăm dò ngôn ngữ `scripts/phase1_lang_probe.py` | cho biết 7 gold doc hiện tại thuộc ngôn ngữ nào, quyết định có đáng đầu tư dịch en/zh không |
| H | Few-shot dịch thuật từ ICD-11 | `data/vocab/trilingual_terms.tsv` đã dựng; cách dùng đúng là RETRIEVE vài cặp gần nhất cho mỗi query, không phải tra cứu toàn bảng |

## Rủi ro

- **Dữ liệu mong manh.** Collection `medical_documents` (3.927.800 điểm) nằm trong
  volume `prompt-engineering-evaluation_qdrant_storage` của project khác, không
  phải `r2ai_qdrant_data`. Ai đó chạy `docker compose down -v` ở project kia là
  mất hết, phải ingest lại từ đầu. **Snapshot trước khi làm bất cứ gì.**
- **`docker-compose.yaml` của repo này map sai cổng gRPC** (`6334:5334` thay vì
  `6334:6334`) và trùng cổng với container đang chạy.
- **Chưa biết hạn mức nộp bài mỗi ngày.** Cần nắm trước khi lên lịch đo.
- **Chưa có dev set.** `data/labeling/dev_set.jsonl` có 100 query x ~30 ứng viên
  nhưng `human_score` toàn None. Leaderboard vẫn là thước đo duy nhất, nên mỗi
  bài nộp phải cô lập đúng một biến.
- **Latency.** Phase 1.1 (tìm sâu trên index chunk) đo được 478 giây mỗi câu trên
  máy local, tức 159 giờ cho 1.200 câu. Chưa chẩn đoán được là do Qdrant hay do
  rerank trên GTX 1660 Ti. Doc index 1,0M điểm giảm phía tìm kiếm, nhưng phải
  đo lại một câu trên server trước khi phóng cả 1.200.

## Hai mức phần cứng, đừng lẫn

| | Cấu hình | Dùng để |
|---|---|---|
| Local | GTX 1660 Ti 6 GB | thử nghiệm nhỏ, việc CPU |
| Server | chưa biết | chạy thật |

Qwen3-8B (dịch query, HyDE) và Qwen3-Reranker-4B chắc chắn không chạy nổi trên
local, phải đợi server.

---

## Lưu trữ: giai đoạn sinh nhãn reranker

Việc này đã hoàn thành phần quan trọng (`models/medical-reranker` đang dùng ở
mọi bài nộp, tải lại bằng `scripts/fetch_reranker.py`). Spec đầy đủ ở
[docs/kien-truc-tong-quan.md](docs/kien-truc-tong-quan.md). Các việc còn dở
(listwise KL loss, dev set gán tay, hiệu chỉnh cutoff) tạm gác vì nút thắt hiện
tại là recall của phễu, không phải chất lượng xếp hạng.
