# TODO - sinh nhãn train reranker

Spec đầy đủ: [docs/pipeline-sinh-nhan-reranker.md](docs/pipeline-sinh-nhan-reranker.md)
Deadline 31/10. Cập nhật 2026-10-19.

## Đã chốt

| | Quyết định |
|---|---|
| Teacher | Qwen3-8B-AWQ (điều lệ cho tới 15B, chọn 8B cho nhẹ máy) |
| Quy mô | 5.000 chunk nguồn → ~5.100 nhóm listwise → 81.600 cặp chấm |
| Ngân sách | ~5 giờ GPU, xong trong ngày |
| Dev set | 100 query gán tay, bắt buộc |

## Bảng việc

| # | Việc | GPU | Trạng thái |
|---|---|---|---|
| 0 | Dựng bảng ICD song ngữ vi-en | không | **xong** - `data/vocab/icd_bilingual.tsv`, 10.752 dòng |
| 1 | Sửa backbone `ms-marco-MiniLM` → `bge-reranker-v2-m3` | không | **xong** - `sft-reranker/main.py`, smoke test vi→en xếp đúng |
| 2 | Script host vLLM | sau | **xong** - `scripts/serve_vllm.sh`, lên server chỉ việc chạy |
| A | **Ingest corpus** - nút thắt thật, xem mục CHẶN | có | **chưa làm**, 0,23% |
| 3 | Viết bước 1-2-3: chọn chunk, sinh query, lọc round-trip | không | **code + prompt xong**. Chờ việc A (ingest) để chạy pilot thật |
| 3b | Audit độ phủ `audit_coverage.py` | có | **xong**, phát hiện việc A |
| 4 | Pilot 300 chunk → **CỔNG CHẶN: soát tay 30 query sinh ra** | có | |
| 5 | Viết bước 4-5: lấy ứng viên, chấm utility hai tầng | không | |
| 6 | Pilot chấm → **CỔNG CHẶN: soát tay 50 cặp đã chấm** | có | |
| 7 | Chạy full 5.000 chunk | có | |
| 8 | Đổi loss sang listwise KL (cần dữ liệu từ việc 7) | không | |
| 9 | Hiệu chỉnh cutoff cho F2 trên dev set | có | |
| — | Xen kẽ: gán tay 100 query dev set | không | |

## CHẶN: corpus chưa được index

Đo lúc 2026-10-19, trên Qdrant đang chạy:

| | |
|---|---|
| Tài liệu trong corpus | 1.220.598 |
| Đã ingest | **2.854 (0,23%)** |
| Ngôn ngữ đã ingest | **chỉ tiếng Trung**, 19 domain, 4.105 chunk |
| Chunk tiếng Anh | **0** |
| Chunk tiếng Việt | **0** |

Audit độ phủ xác nhận: với 200 query thật, nhánh `en` và `vi` trả về điểm 0,0 cho **mọi** query.
Không có gì để truy xuất.

Bước 1 của pipeline sinh nhãn cần chọn 5.000 chunk với tỷ lệ EN 60% / VI 30%.
Hiện không có chunk EN hay VI nào, nên **không chạy được, kể cả pilot**.

Tốc độ ingest đo từ ledger: **1,6 doc/s** → ngoại suy 1,22 M tài liệu = **218 giờ**.
Deadline còn 12 ngày. Đây là nút thắt lớn nhất của dự án, lớn hơn mọi thứ khác cộng lại.

Xem mục "Việc cần quyết về ingest" bên dưới.

## Đã có

Prompt đã viết xong ở `prompts/gen_query_short.txt` và `prompts/gen_query_narrative.txt`.
Qua validator, đã đăng ký MLflow registry (version 1 cả hai).
Doc giải thích từng bước kèm sơ đồ: [docs/buoc-1-2-3-sinh-query.md](docs/buoc-1-2-3-sinh-query.md)

Viết xong thì chạy pilot:

```bash
python -m src.rag.labeling.run_generate --pilot
```

## Hai cổng chặn

**Việc 4** - soát tay 30 query sinh ra. Kiểm:
- Phân phối dài/ngắn có khớp 66/34 không?
- Có bê nguyên thuật ngữ lâm sàng từ chunk sang query không?
- Tỷ lệ round-trip giữ lại bao nhiêu? **Dưới 40% thì sửa prompt, chạy lại.**

**Việc 6** - soát tay 50 cặp đã chấm. Kiểm teacher chấm có đúng không.
**Không được chạy full nếu cổng này chưa qua.** Sai ở đây nghĩa là 5.000 query nhãn rác.

## Để dành, không làm lúc này

- Query router. 1.200 query là tập submission đã biết trước nên không có bài toán latency. Chi tiết ở mục 10 của spec.
- Từ điển dân gian ↔ lâm sàng. Bảng ICD đã phủ 38,9% rồi.
- Thuật ngữ tiếng Trung. Corpus zh chỉ phủ 4,1%, không có tài liệu để khớp vào.
- Hard negative cùng chương ICD. Bật sau khi pilot chạy ổn.

## Việc cần quyết về ingest

1. **Chạy ingest trên GPU thuê, không chạy local.** Con số 1,6 doc/s đo trên GTX 1660 Ti,
   card không có Tensor Core. Chính comment trong `ingest_documents.py` đã ghi: fp16 chậm
   hơn fp32 2-4 lần trên card này, và dặn *"re-measure on different hardware"*.
   5060 Ti có Tensor Core và 16 GB, nên fp16 + batch lớn sẽ đổi hẳn con số. Giả định
   "fp32 nhanh hơn" là đặc thù của 1660 Ti, **không được mang sang máy mới mà không đo lại**.

2. **Ưu tiên thứ tự domain.** Nếu hết thời gian thì phần đã ingest phải là phần đáng giá nhất.
   Query là bệnh nhân hỏi bác sĩ, nên ưu tiên: vinmec, medlatec, suckhoedoisong,
   nhathuoclongchau, bachmai trước; báo tổng hợp (thanhnien, laodong) sau cùng.

3. **PubMed 240k tài liệu tiếng Anh là bắt buộc**, vì đó là đường truy xuất chính vi→en.
   Không được để nó xuống cuối hàng đợi.
