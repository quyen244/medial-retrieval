# Bàn giao: chấm nốt pool bằng reranker fine-tune trên Kaggle, dựng 2 bài nộp d160 và d100

Viết 2026-11-04. Người nhận: agent chạy ở **máy local** (có `data/tmp/phase1/`, `data/doc_texts_8k.jsonl`,
`data/raw_documents/`, kaggle CLI). Đọc `CLAUDE.md` trước; mọi mã bài viết kèm nghĩa (xem `docs/ky-hieu-bai-nop.md`).

## Mục tiêu

Hai file zip nộp bài, chỉ khác nhau ở `m_doc`:

| Tên bài | `relevant_docs` | `relevant_chunks` |
|---|---|---|
| `fuse_d160_c100_ftrr_qc_orig_ctop60+top300_cap2700` | 160 doc, pool top300 chấm lại bằng reranker fine-tune | y hệt SOTA |
| `fuse_d100_c100_ftrr_qc_orig_ctop60+top300_cap2700` | 100 doc, cùng pool | y hệt SOTA |

Đối chứng: SOTA `fuse_d160_c100_top300_orig_ctop60+top300_cap2700` (FINAL 0.2562, `DOCS_F2` 0.2849, `CHUNKS_F2` 0.2275).
Tầng chunk giữ nguyên SOTA nên `CHUNKS_F2` của cả hai bài PHẢI ra đúng 0.2275; lệch là bài dựng sai.
So `DOCS_F2` với 0.2849, không so FINAL.

Biến số đổi so với SOTA ở tầng doc (cùng 300 doc ứng viên mỗi câu, chỉ thứ tự đổi):
1. Reranker: `BAAI/bge-reranker-v2-m3` -> `quyenpro/re-ranker-vibio` (SFT trên nhãn LLM, xem dưới).
2. Câu chấm: câu gốc -> câu sửa (`corrected_query`), theo quyết định 2026-10-27 trong `CLAUDE.md`.
3. Text doc: payload Qdrant -> `data/doc_texts_8k.jsonl`, len 1024 -> 2048 (khớp điều kiện lúc train).
Ba thứ đi chung vì model được train đúng trên điều kiện 2-3; không tách được bằng hai bài này.

Vì sao d100: reranker tốt hơn dồn gold lên đầu, nên đỉnh `m_doc` dịch xuống. Đường biên đã đo với bge cũ:
81-110 là 12,9%, 111-160 là 10,0%, 161-220 là 6,8% (`docs/phat-hien-tang-diem.md` mục 19, 20, 24).
d160 đo hiệu ứng reranker sạch (cùng `m_doc` với SOTA); d100 đo xem đỉnh có dịch không.

## Trạng thái khi bàn giao

- Model: https://huggingface.co/quyenpro/re-ranker-vibio (public). SFT 1 epoch, 15.870 cặp / 263 câu, len 2048,
  batch hiệu dụng 32. Val pairwise (18 câu, nhãn chủ yếu là LLM): mọi cặp 0,7903 -> 0,8248; 1,0 so với 0,4: 0,6985 -> 0,7331.
  Nhãn: Qwen3-8B-AWQ, prompt `71b3b840a577`, precision mức 2 so với người = 0,60 (dễ dãi).
- Server RTX 5060 Ti chấm được một phần pool bằng `scripts/rerank_pool_ft.py` (fp16, ~35 cặp/s), tự dừng sau 90 phút
  (~10:11 ngày 2026-11-04), ra `data/doc_top300_ftrr_qcorr_pool.jsonl` trên server, khoảng 600+ câu.
  Câu 1 chấm bf16 (lệch fp16 tối đa 0,008), còn lại fp16.

## Các bước

### 0. Code

```bash
git checkout kaggle-rerank-jobs && git pull
```

### 1. Lấy phần đã chấm từ server

Chép `data/doc_top300_ftrr_qcorr_pool.jsonl` của server về **`data/tmp/phase1/doc_top300_ftrr_qcorr_pool.jsonl`**.
Kiểm:

```bash
python -c "
import json
rows=[json.loads(l) for l in open('data/tmp/phase1/doc_top300_ftrr_qcorr_pool.jsonl', encoding='utf-8')]
print(len(rows), 'câu', {(r['rr_model'], r['rr_len']) for r in rows}, {len(r['candidates']) for r in rows})"
```
Phải là `{('quyenpro/re-ranker-vibio', 2048)}` và `{300}`. Dòng cuối ghi dở (nếu có) được các script bỏ qua.

### 2. Dựng job Kaggle cho phần còn thiếu

```bash
python -m scripts.kaggle_rerank_ft build --owner <kaggle-user>
```
Ra `data/kaggle/rerank_ft_dataset/ftrr_rest.zip` và kernel `data/kaggle/kernel_ftrr_rest_ce/`.
Kernel chạy `kaggle/rerank_job.py` (CrossEncoder fp16, batch 16), **mỗi GPU T4 một tiến trình, chia câu theo `id % 2`**,
tự dừng sau 11 giờ. Shard 0 chấm lại 32 cặp đã chấm trên server trước (`check`, ngưỡng 0,05); lệch quá là dừng.
Thử trên server: lệch 0,0002.

Rồi làm theo lệnh in ra (dataset phải `ready` trước khi push kernel):
```bash
kaggle datasets create -p data/kaggle/rerank_ft_dataset     # lần sau: kaggle datasets version -p ... -m "..."
kaggle datasets status <kaggle-user>/vibiomir-rerank-ft
kaggle kernels push -p data/kaggle/kernel_ftrr_rest_ce
kaggle kernels status <kaggle-user>/vibiomir-ftrr-rest-ce
kaggle kernels output <kaggle-user>/vibiomir-ftrr-rest-ce -p data/kaggle/out/ftrr_rest --file-pattern "scores_.*jsonl|log_.*txt"
```
Bỏ qua dòng `python -m scripts.kaggle_rerank_vllm merge ...` mà lệnh build in ra (template dùng chung); dùng merge ở bước 3.

Tốc độ T4 cho len 2048 CHƯA đo với model này. Đọc `log_ftrr_rest.txt` (`[gpu0] [job] N/M câu | x cặp/s | còn ~h`) sau
vài phút để ước lượng; đừng đoán.

Nếu 11 giờ không đủ: tải điểm về, gộp phần đã có vào file DONE, rồi build lại cho phần còn thiếu:
```bash
python -m scripts.kaggle_rerank_ft merge --allow-partial --out data/tmp/phase1/doc_top300_ftrr_qcorr_pool.jsonl
python -m scripts.kaggle_rerank_ft build --owner <kaggle-user>     # rồi datasets version + kernels push
```

### 3. Gộp đủ 1.200 câu

```bash
python -m scripts.kaggle_rerank_ft merge
```
Ra `data/tmp/phase1/doc_top300_ftrr_qcorr_full_pool.jsonl`, tức `POOLS["ftrr_qc"]` trong `scripts/fuse_submission.py`.
Phải in `thiếu 0`. Câu có ở cả server và Kaggle thì lấy điểm server.

### 4. Dựng hai bài

```bash
for M in 160 100; do
  python -m scripts.fuse_submission --doc-pool ftrr_qc --base-pool top300 --chunk-pool top60 top300 \
    --methods orig --cap 2700 --m-chunk 100 --m-doc $M
done
```
Ra `data/tmp/phase1/fuse_d{160,100}_c100_ftrr_qc_orig_ctop60+top300_cap2700.zip`.
Cột "trùng orig" in ra là số doc/câu trùng top `m_doc` của SOTA: ghi lại, đó là mức thay đổi thực của reranker.

### 5. Kiểm trước khi nộp (bắt buộc)

- Zip < 100 MB, đúng một file `.json`, không thư mục con.
- Đủ 1.200 id; `doc_id` là số nguyên.
- `len(relevant_docs)` = 160 (hoặc 100) ở mọi câu; `len(relevant_chunks)` = 100 trừ câu thiếu doc.
- `relevant_chunks` TRÙNG từng câu với bài SOTA `fuse_d160_c100_top300_orig_ctop60+top300_cap2700` (nếu còn file).
  Không trùng là sai tầng chunk.
- 100 doc đầu của bài d160 trùng `relevant_docs` của bài d100 ở cả 1.200 câu.

### 6. Sau khi có điểm

- Đổi tên `scoring_result.zip` thành tên bài ngay khi tải (mục 22 của `docs/phat-hien-tang-diem.md`), cất vào `res/`.
- Tính `h = P × m` từ `scores.txt`, không suy từ dạng đóng.
- Ghi một mục mới vào `docs/phat-hien-tang-diem.md` và bảng trong `CLAUDE.md`. `CHUNKS_F2` khác 0.2275 là bài hỏng, đừng ghi.
- Đọc kết quả: d160 so `DOCS_F2` với 0.2849. d100 so với d160 để biết biên hạng 101-160 với reranker mới.

## File liên quan

| File | Vai trò |
|---|---|
| `scripts/rerank_pool_ft.py` | chấm pool trên 1 GPU (server); `--dtype fp16 --max-minutes N`, chạy lại là làm tiếp |
| `scripts/kaggle_rerank_ft.py` | build job Kaggle cho câu còn thiếu, merge về pool đủ |
| `kaggle/rerank_job.py` | code chạy trên Kaggle (nhúng vào kernel), shard theo GPU |
| `scripts/kaggle_rerank_vllm.py` | chứa `KERNEL_CE` và `kernels()` mà builder dùng lại |
| `scripts/fuse_submission.py` | lắp bài; pool `ftrr_qc` đã đăng ký trong `POOLS` |
| `sft-reranker/main.py`, `out_labels/after_labels_train.py` (server, không commit) | train reranker |
