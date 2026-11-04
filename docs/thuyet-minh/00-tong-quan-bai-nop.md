# Thuyết minh 00 - Tổng quan 5 bài nộp private test

Đội nộp 5 bài cho vòng private của ViBioMIR (ROAD TO AI 2026 Stage 3).
Mỗi bài là một cặp (danh sách doc, danh sách đoạn): hai tầng chấm độc lập, nên mỗi tầng được chọn riêng.
Chi tiết hệ thống: [01-he-thong-sota.md](01-he-thong-sota.md) (index, truy hồi, rerank bge, tầng chunk).
Chi tiết reranker fine-tune: [02-sft-reranker.md](02-sft-reranker.md) (sửa chính tả, gán nhãn, SFT, chấm pool, gộp ens).

## Ký hiệu

| Mã | Nghĩa |
|---|---|
| `d160` | nộp 160 doc mỗi câu |
| `c100`, `c120` | nộp 100 / 120 đoạn mỗi câu |
| `cap2700`, `cap2050` | mỗi đoạn là 2.700 / 2.050 ký tự đầu markdown gốc, đã bóc cú pháp |
| `top300` | pool 300 doc tìm dense bằng `BAAI/bge-m3` trên index cấp tài liệu `medical_docs` |
| `ctop60+top300` | thứ tự đoạn: hết 60 doc của pool top60 rồi tiếp phần còn lại của pool top300 |
| `bge`, `orig` (ở bài 0) | doc xếp bằng `BAAI/bge-reranker-v2-m3`, câu hỏi gốc, len 1024 |
| `ftrr`, `ftrr_qc` | doc xếp bằng reranker fine-tune `quyenpro/re-ranker-vibio`, câu sửa chính tả, text 8k, len 2048 |
| `ens_ftrrw0.6_bge` | RRF có trọng số: `0,6/(60 + hạng ftrr) + 0,4/(60 + hạng bge)` trên cùng 300 ứng viên |
| `priv` | bài dựng cho vòng private |

## Bảng 5 bài

| # | Tên bài | Tầng doc lấy từ | Tầng chunk lấy từ | Script | Điểm |
|---|---|---|---|---|---|
| 0 | `fuse_d160_c100_top300_orig_ctop60+top300_cap2700` (id 7910, file nộp `fuse_d160_c100_top300_orig_ctop60t.zip`) | 160 doc đầu pool top300 xếp bge câu gốc | `ctop60+top300`, 100 đoạn, trần 2.700 | `scripts/fuse_submission.py` | public FINAL 0.2562 (`DOCS_F2` 0.2849, `CHUNKS_F2` 0.2275) |
| 1 | `fuse_d160_c100_ftrr_qc_orig_ctop60+top300_cap2700` | 160 doc đầu pool top300 xếp ftrr câu sửa | y hệt bài 0 | `scripts/fuse_submission.py` (hoặc `scripts/build_private_submissions.py`) | chưa có |
| 2 | `priv_d160_bge_c120_ctop60+top300_cap2050` | y hệt bài 0 | `ctop60+top300`, 120 đoạn, trần 2.050 | `scripts/build_private_submissions.py` | chưa có |
| 3 | `priv_d160_ftrr_c120_ctop60+top300_cap2050` | y hệt bài 1 | y hệt bài 2 | `scripts/build_private_submissions.py` | chưa có |
| 4 | `priv_d160_ens_ftrrw0.6_bge_c120_ctop60+top300_cap2050` | RRF 0,6 ftrr + 0,4 bge, 160 doc đầu | y hệt bài 2 | `scripts/build_private_submissions.py` | chưa có |

Vòng private không hiện điểm, nên bài 1-4 chưa có số đo nào.

## Lệnh dựng (PowerShell, thư mục gốc repo)

Điều kiện: đã có tài liệu gốc `data\raw_documents\` + `data\raw_documents_index.json`, các pool trong `data\tmp\phase1\` và `data\tmp\phase1\rescore_bge.jsonl` (xem danh sách link ở cuối).

Bài 0:

```powershell
python -m scripts.fuse_submission --doc-pool top300 --methods orig --m-doc 160 --m-chunk 100 --cap 2700 `
    --out-dir data --chunk-pool top60 top300
```

Ra `data\fuse_d160_c100_top300_orig_ctop60+top300_cap2700.zip`.

Bài 1:

```powershell
python -m scripts.fuse_submission --doc-pool ftrr_qc --base-pool top300 --chunk-pool top60 top300 `
    --methods orig --cap 2700 --m-chunk 100 --m-doc 160
```

Ra `data\tmp\phase1\fuse_d160_c100_ftrr_qc_orig_ctop60+top300_cap2700.zip`.

Bài 1, 2, 3, 4 trong một lệnh (cần zip bài 0 ở `data\`):

```powershell
python -m scripts.build_private_submissions --cap 2050
```

Mặc định của script là trần 2.050; lần dựng đầu ở trần 2.250 cho đoạn trung bình 1.897 ký tự và mỗi zip c120 nặng 106,1 MB, vượt giới hạn 100 MB.
Khi dựng, script in: doc bge trùng SOTA 1200/1200 câu; ftrr trùng bge 124,1/160 doc mỗi câu; ens trùng bge 138,1 và trùng ftrr 145,0 doc mỗi câu.
Ra 4 zip trong `data\tmp\phase1\`; bài 1 của lệnh này ghi đè bản của `fuse_submission` (cùng `relevant_docs`, `relevant_chunks` chép thẳng từ zip bài 0).
Kiểm định dạng mỗi zip theo [01-he-thong-sota.md](01-he-thong-sota.md) mục 5.5.

## Dữ liệu cần upload và link cần điền

Thay mỗi `<LINK_DRIVE: ...>` trong ba tài liệu bằng link thật sau khi upload.
Dung lượng đo bằng `ls -la` trên máy local ngày 2026-11-04, trừ chỗ ghi `TODO(kiểm)`.

| # | File / thư mục | Dung lượng | Dùng cho | Link |
|---|---|---:|---|---|
| 1 | Mã nguồn repo (nhánh `kaggle-rerank-jobs`) | - | mọi bài | `<REPO_URL>` hoặc `<LINK_DRIVE: mã nguồn repo>` |
| 2 | `data/queries/real_queries.jsonl` | 0,23 MB | mọi bài | `<LINK_DRIVE: data/queries/real_queries.jsonl>` |
| 3 | `zip_folder/` (85 zip tài liệu gốc theo domain) hoặc `data/raw_documents/` | `TODO(kiểm)` | tầng chunk mọi bài | `<LINK_DRIVE: zip_folder/>`; có sẵn ở HF dataset `quyenpro/vibiomir-raw-documents` (`TODO(kiểm):` public?) |
| 4 | `data/raw_documents_index.json` | 30,3 MB | tầng chunk mọi bài | `<LINK_DRIVE: data/raw_documents_index.json>` |
| 5 | `data/backups/medical_docs-20261024-211206/` (snapshot Qdrant + manifest) | 7,0 GB | truy hồi lại từ đầu | `<LINK_DRIVE: data/backups/medical_docs-20261024-211206/>`; có sẵn ở HF dataset `quyenpro/vibiomir-medical-docs` (`TODO(kiểm):` public?) |
| 6 | `data/doc_texts.jsonl` | 2,14 GB | dựng lại index | `<LINK_DRIVE: data/doc_texts.jsonl>` |
| 7 | `data/doc_texts_8k.jsonl` | 4,08 GB | chấm ftrr, gán nhãn | `<LINK_DRIVE: data/doc_texts_8k.jsonl>` |
| 8 | `data/spelling/corrected_queries.jsonl` | 1,1 MB | ftrr | `<LINK_DRIVE: data/spelling/corrected_queries.jsonl>` |
| 9 | `data/tmp/phase1/doc_top300_bge_pool.jsonl` | 73,7 MB | bài 0-4 | `<LINK_DRIVE: data/tmp/phase1/doc_top300_bge_pool.jsonl>` |
| 10 | `data/tmp/phase1/doc_d35_c44_top60_bge_pool.jsonl` | 13,6 MB | tầng chunk bài 0-4 | `<LINK_DRIVE: data/tmp/phase1/doc_d35_c44_top60_bge_pool.jsonl>` |
| 11 | `data/tmp/phase1/rescore_bge.jsonl` | 8,9 MB | `fuse_submission.py` luôn mở | `<LINK_DRIVE: data/tmp/phase1/rescore_bge.jsonl>` |
| 12 | `data/tmp/phase1/doc_top300_ftrr_qcorr_full_pool.jsonl` | 86,3 MB | bài 1, 3, 4 | `<LINK_DRIVE: data/tmp/phase1/doc_top300_ftrr_qcorr_full_pool.jsonl>` |
| 13 | `data/tmp/phase1/doc_top300_ftrr_qcorr_pool.jsonl` (406 câu server) | khoảng 30 MB | tái hiện merge | `<LINK_DRIVE: data/tmp/phase1/doc_top300_ftrr_qcorr_pool.jsonl>` |
| 14 | `data/kaggle/out/ftrr_rest/` (điểm 794 câu Kaggle + log) | khoảng 4,8 MB | tái hiện merge | `<LINK_DRIVE: data/kaggle/out/ftrr_rest/>` |
| 15 | `data/labeling_doc_qcorr/` (`candidates.jsonl`, `texts.jsonl`) | 316 MB | gán nhãn LLM | `<LINK_DRIVE: data/labeling_doc_qcorr/>` |
| 16 | `data/labeling_doc_qcorr_fit/train.jsonl`, `val.jsonl` (tập SFT thật, khớp model card; `out_labels/` của server đã mất) | 93,0 MB + 6,9 MB | SFT | `<LINK_DRIVE: data/labeling_doc_qcorr_fit/>` |
| 17 | `data/T28_rrf_bge_q3e/w0.6_qc_vi90_bge_q3etext2048_full_pool.jsonl` | 88,8 MB | pool gán nhãn | `<LINK_DRIVE: data/T28_rrf_bge_q3e/w0.6_qc_vi90_bge_q3etext2048_full_pool.jsonl>` |
| 18 | `artifact/labeling.zip` (trang gán nhãn) | 5,7 MB | minh hoạ nhãn tay | `<LINK_DRIVE: artifact/labeling.zip>` |
| 19 | Bài 0 `data/fuse_d160_c100_top300_orig_ctop60+top300_cap2700.zip` | 98,9 MB | nộp, đầu vào bài 1-4 | `<LINK_DRIVE: bài 0>` |
| 20 | Bài 1-4 trong `data/tmp/phase1/` | 98,9 MB (bài 1), 99,5 MB mỗi bài 2-4 (94,9 MiB) | nộp | `<LINK_DRIVE: bài 1>`, `<LINK_DRIVE: bài 2>`, `<LINK_DRIVE: bài 3>`, `<LINK_DRIVE: bài 4>` |

Checkpoint (không cần upload Drive, tải thẳng từ HuggingFace):

| Model | Định danh | Ghi chú |
|---|---|---|
| Embedding | `BAAI/bge-m3` | public |
| Reranker bge | `BAAI/bge-reranker-v2-m3` | public |
| Reranker fine-tune | `quyenpro/re-ranker-vibio` | public theo `docs/ban-giao-ftrr-submission.md`; `TODO(kiểm):` link model card và commit hash của checkpoint đã chấm |
| LLM sửa chính tả + chấm nhãn | `Qwen/Qwen3-8B-AWQ` | public |
