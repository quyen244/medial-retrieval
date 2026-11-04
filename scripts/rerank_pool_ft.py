"""Chấm lại pool doc bằng reranker đã fine-tune, câu SỬA, text data/doc_texts_8k.jsonl, len 2048.

Khác scripts/rerank_pool.py (câu gốc, text payload Qdrant): ở đây khớp đúng điều kiện lúc train
(src/rag/labeling/run_doc_labels.py: câu sửa, text 8k cắt phần reranker len 2048 thấy).
Cùng tập ứng viên với pool vào, chỉ thay `rerank_score` (cũ giữ ở `rerank_score_prev`) và xếp lại.
Chạy lại cùng lệnh là làm tiếp (bỏ câu đã có trong --out).

    python scripts/rerank_pool_ft.py --limit 3          # đo tốc độ
    python scripts/rerank_pool_ft.py
"""
import argparse
import json
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/doc_top300_bge_pool.jsonl")
    ap.add_argument("--out", default="data/doc_top300_ftrr_qcorr_pool.jsonl")
    ap.add_argument("--model", default="models/medical-reranker")
    ap.add_argument("--model-name", default="quyenpro/re-ranker-vibio")
    ap.add_argument("--texts", default="data/doc_texts_8k.jsonl")
    ap.add_argument("--corrected", default="data/spelling/corrected_queries.jsonl")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16", help="T4 trên Kaggle không có bf16")
    ap.add_argument("--max-minutes", type=float, default=0, help="quá hạn thì dừng sau câu đang chấm; 0 = không giới hạn")
    a = ap.parse_args()

    pool = list(read_jsonl(a.pool))
    if a.limit:
        pool = pool[: a.limit]
    done = {r["id"] for r in read_jsonl(a.out)} if os.path.exists(a.out) else set()
    todo = [r for r in pool if r["id"] not in done]
    print(f"[pool] {len(pool)} câu, đã có {len(done)}, còn {len(todo)}", flush=True)
    if not todo:
        return 0

    qc = {int(r["id"]): r["corrected_query"] for r in read_jsonl(a.corrected)}
    need = {str(c["doc_id"]) for r in todo for c in r["candidates"]}
    t0 = time.time()
    texts = {}
    for r in read_jsonl(a.texts):
        d = str(r["doc_id"])
        if d in need:
            texts[d] = r["text"]
    miss = need - texts.keys()
    print(f"[text] {len(texts):,}/{len(need):,} doc có text, thiếu {len(miss)} | {time.time() - t0:.0f}s", flush=True)

    import torch
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(a.model, num_labels=1, max_length=a.max_length,
                         model_kwargs={"torch_dtype": torch.float16 if a.dtype == "fp16" else torch.bfloat16})

    n_pairs, t_start = 0, time.time()
    with open(a.out, "a", encoding="utf-8") as fh:
        for i, row in enumerate(todo, 1):
            if a.max_minutes and time.time() - t0 > a.max_minutes * 60:
                print(f"[dừng] hết {a.max_minutes:.0f} phút: xong {i - 1}/{len(todo)} câu, chạy lại cùng lệnh để làm tiếp",
                      flush=True)
                break
            q = qc[int(row["id"])]
            cands = row["candidates"]
            scores = model.predict([(q, texts.get(str(c["doc_id"]), c.get("title", ""))) for c in cands],
                                   batch_size=a.batch_size, show_progress_bar=False)
            new = []
            for c, s in zip(cands, scores):
                new.append({**c, "rerank_score_prev": c.get("rerank_score"), "rerank_score": float(s)})
            new.sort(key=lambda c: -c["rerank_score"])
            fh.write(json.dumps({**row, "query": q, "query_orig": row["query"], "rr_model": a.model_name,
                                 "rr_len": a.max_length, "candidates": new}, ensure_ascii=False) + "\n")
            fh.flush()
            n_pairs += len(cands)
            el = time.time() - t_start
            if i == 1 or i % 10 == 0 or i == len(todo):
                print(f"[rerank] {i}/{len(todo)} câu | {n_pairs / el:.1f} cặp/s | "
                      f"còn ~{(len(todo) - i) * el / i / 3600:.2f}h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
