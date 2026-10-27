# Chạy rerank trên Kaggle (T12, T14)

Chia việc:

- **Máy local** (nhẹ): tìm ứng viên trên Qdrant, lấy text doc từ payload `text` của Qdrant `medical_docs`, đóng thành job `.zip`; sau đó gộp điểm và lắp bài nộp.
- **Kaggle** (nặng GPU): chỉ rerank các cặp (câu, doc) trong job.

Vì vậy KHÔNG phải đưa lên Kaggle:

| Thứ | Cần trên Kaggle? | Vì sao |
|---|---|---|
| `data/raw_documents/` (1M file) | Không | Chỉ dùng để cắt `chunk_text` khi lắp bài nộp, việc này làm ở local (`fuse_submission.py`). |
| Snapshot `medical_docs` 1,0M (7,5 GB) | Không | Tìm kiếm chạy ở local; Kaggle chỉ cần text của các doc ứng viên. |
| Snapshot `medical_documents` 3,9M (27 GB) | Không | Chỉ C0 (index chunk) cần; C0 cũng sẽ đi theo cùng kiểu job này. |

Job chứa đúng text trong payload Qdrant, chính chuỗi phase1 đã đưa vào reranker, nên điểm Kaggle cùng thang với điểm cache.
Không dùng `data/doc_texts.jsonl`: cùng nội dung nhưng nối các phần bằng ` | `, điểm bge lệch tới 0,24 (đo trên Kaggle 2026-10-26).
Vì vậy cả T12 lẫn T14 đều cần Qdrant local đang chạy khi dựng job.
Job T12 còn kèm `check.jsonl`: 400 cặp đã chấm ở local; kernel chấm lại và DỪNG nếu lệch quá 0,02, để không gộp hai thang điểm khác nhau.

## Một lần: cài Kaggle CLI

```bash
pip install kaggle
# Kaggle > Settings > API > Create New Token, lưu vào ~/.kaggle/kaggle.json (Windows: C:\Users\<bạn>\.kaggle\kaggle.json)
```

Tài khoản phải xác minh số điện thoại thì kernel mới bật được GPU và internet (cần internet để tải model từ HuggingFace).

## T14: Qwen3-Reranker-4B chấm lại pool top300

```bash
python -m scripts.kaggle_jobs t14 --owner <kaggle-user>
kaggle datasets create -p data/kaggle/dataset              # lần đầu; lần sau: kaggle datasets version -p data/kaggle/dataset -m "t14"
kaggle kernels push -p data/kaggle/kernel_t14_qwen3r4b_smoke  # chạy thử 20 câu
kaggle kernels status <kaggle-user>/vibiomir-rerank-t14-qwen3r4b-smoke
kaggle kernels output <kaggle-user>/vibiomir-rerank-t14-qwen3r4b-smoke -p data/kaggle/out/smoke_t14
```

Đọc dòng `cặp/s` và `còn ~... giờ` trong `log_*.txt`.
Dataset mới tạo có thể cần vài phút mới gắn được vào kernel; push lỗi "dataset not found" thì đợi rồi push lại.

**Thời gian ước lượng (SUY LUẬN, chưa đo):** 360.000 cặp, Qwen3-4B fp16 trên 1 T4 có thể mất trên 12 giờ.
Nếu bản chạy thử báo vượt 11 giờ, có hai lựa chọn:

- chạy nhiều lượt (xem "Chạy tiếp" bên dưới), tốn khoảng nửa quota GPU tuần;
- hoặc dựng lại job chỉ chấm 100 doc đầu theo bge mỗi câu: `python -m scripts.kaggle_jobs t14 --owner <user> --top-k 100 --name t14_qwen3r4b_k100 --out-pool data/tmp/phase1/doc_top300_qwen3r4b_k100_pool.jsonl`. Khi đó T14 đo "Qwen xếp lại top 100", không phải cả pool.

Chạy thật:

```bash
kaggle kernels push -p data/kaggle/kernel_t14_qwen3r4b
kaggle kernels output <kaggle-user>/vibiomir-rerank-t14-qwen3r4b -p data/kaggle/out/t14_qwen3r4b/run1
python -m scripts.kaggle_jobs merge --name t14_qwen3r4b
python -m scripts.fuse_submission --doc-pool top300q4b --methods orig --m-doc 160 --m-chunk 100 --cap 2700 --out-dir data --chunk-pool top60 top300
```

T4 không có bf16 nên model chạy fp16; điểm NaN (tràn số) được ghi là `null` và log báo số lượng.

## T12: pool top1000

Cần Qdrant local có collection `medical_docs` (bật Docker Desktop; nếu mất collection thì `bash scripts/restore_medical_docs.sh`).
Bước tìm có encode 1.200 câu bằng BGE-M3, chạy ở local.

```bash
docker compose up -d
python -m scripts.kaggle_jobs t12 --owner <kaggle-user>
kaggle datasets version -p data/kaggle/dataset -m "t12"
kaggle kernels push -p data/kaggle/kernel_t12_top1000_bge_smoke
kaggle kernels push -p data/kaggle/kernel_t12_top1000_bge
kaggle kernels output <kaggle-user>/vibiomir-rerank-t12-top1000-bge -p data/kaggle/out/t12_top1000_bge/run1
python -m scripts.kaggle_jobs merge --name t12_top1000_bge
python -m scripts.fuse_submission --doc-pool top1000 --methods orig --m-doc 160 --m-chunk 100 --cap 2700 --out-dir data --chunk-pool top60 top300
```

Chỉ các cặp chưa có điểm bge trong 5 pool phase1 mới lên Kaggle; cặp đã có điểm dùng lại khi gộp.
`hnsw_ef` mặc định 1000 (bằng độ sâu), nên 300 doc đầu có thể lệch nhẹ so với pool top300 cũ (ef 256); doc nào chưa có điểm thì job chấm.

## Chạy tiếp khi một lượt không đủ 12 giờ

Kernel dừng gọn sau 11 giờ (`--max-hours`), điểm đã chấm vẫn nằm trong output.

1. Tải output lượt 1 về `data/kaggle/out/<job>/run1`.
2. Chép file `scores_<job>.jsonl` của lượt 1 vào `data/kaggle/dataset/`, rồi `kaggle datasets version -p data/kaggle/dataset -m "<job> phần 1"`.
3. Push lại kernel: kernel thấy `scores_<job>*.jsonl` trong `/kaggle/input` thì bỏ qua các câu đã chấm.
4. Tải output lượt 2 về `run2`; `merge` đọc mọi `run*/` một lần.

## File

- `kaggle/rerank_job.py`: script chạy trên Kaggle, tự chứa (không import repo). Đọc job dạng `.zip` hoặc thư mục đã giải nén.
  Nó được nhúng thẳng vào `run.py` của kernel, không nằm trong dataset: sửa code thì sinh lại kernel và push, không cần `datasets version`.
- Kernel chạy 1 GPU, `--dtype fp16` cố định (cache local chấm fp16; bf16 trên T4 lệch tới 0,24).
- `scripts/kaggle_jobs.py`: dựng job (`t12`, `t14`), sinh thư mục dataset và kernel, gộp điểm (`merge`).
- `data/kaggle/`: đầu ra của các lệnh trên (không commit).
