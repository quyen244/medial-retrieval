"""Chấm nốt pool top300 bằng reranker fine-tune (quyenpro/re-ranker-vibio) trên Kaggle 2 x T4, rồi gộp lại.

Một phần pool đã chấm trên server RTX 5060 Ti bằng scripts/rerank_pool_ft.py (câu SỬA, text 8k, len 2048, fp16)
ra data/doc_top300_ftrr_qcorr_pool.jsonl; chép nó về
data/tmp/phase1/ ở máy local trước khi build. Lệnh build lấy các câu CÒN THIẾU từ pool gốc, đóng job cho
kaggle/rerank_job.py (CrossEncoder fp16, mỗi GPU một shard câu id % 2, dùng lại KERNEL_CE của
scripts/kaggle_rerank_vllm.py), kèm check.jsonl là vài cặp đã chấm trên server để kiểm cùng thang điểm.

    python -m scripts.kaggle_rerank_ft build --owner <kaggle-user>
    # làm theo lệnh kaggle in ra; tải scores_*.jsonl về data/kaggle/out/ftrr_rest
    python -m scripts.kaggle_rerank_ft merge
    # -> data/tmp/phase1/doc_top300_ftrr_qcorr_full_pool.jsonl (đủ 1.200 câu), POOLS["ftrr_qc"] của fuse_submission
    # python -m scripts.fuse_submission --doc-pool ftrr_qc --chunk-pool top60 top300 --methods orig --cap 2700 \
    #     --m-chunk 100 --m-doc 160

Model tải từ HF trên Kaggle (kernel bật internet). Repo HF phải để public, hoặc thêm HF_TOKEN vào Kaggle Secrets.
"""
import argparse
import glob
import json
import os
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import kaggle_rerank_vllm as kv
from scripts.kaggle_jobs import write_job

MODEL = "quyenpro/re-ranker-vibio"
POOL = "data/tmp/phase1/doc_top300_bge_pool.jsonl"
DONE = "data/tmp/phase1/doc_top300_ftrr_qcorr_pool.jsonl"
FULL = "data/tmp/phase1/doc_top300_ftrr_qcorr_full_pool.jsonl"
TEXTS = "data/doc_texts_8k.jsonl"
CORRECTED = "data/spelling/corrected_queries.jsonl"
DATASET_DIR = "data/kaggle/rerank_ft_dataset"
OUT = "data/kaggle/out"
NAME = "ftrr_rest"
MAX_LEN = 2048


def jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError:
                    continue                # dòng cuối ghi dở


def load_done(path):
    if not os.path.exists(path):
        return {}
    done = {int(r["id"]): r for r in jsonl(path)}
    bad = {(r.get("rr_model"), r.get("rr_len")) for r in done.values()} - {(MODEL, MAX_LEN)}
    if bad:
        raise SystemExit(f"{path}: có câu không phải {MODEL} len {MAX_LEN}: {bad}")
    return done


def build(args):
    done = load_done(args.done)
    base = list(jsonl(args.pool))
    qc = {int(r["id"]): r["corrected_query"] for r in jsonl(args.corrected)}
    todo = [r for r in base if int(r["id"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        raise SystemExit("pool đã chấm đủ, chạy merge")
    queries = {int(r["id"]): qc[int(r["id"])] for r in todo}
    pairs = {int(r["id"]): [str(c["doc_id"]) for c in r["candidates"]] for r in todo}

    # check: câu đã chấm trên server (bỏ câu 1, chấm bf16 trước khi đổi sang fp16), mẫu rải đều thứ hạng
    rng = random.Random(42)
    pool_ids = sorted(q for q in done if q != 1)
    check_q = sorted(rng.sample(pool_ids, min(args.check_queries, len(pool_ids))))
    check = []
    for q in check_q:
        cs = done[q]["candidates"]
        queries[q] = done[q]["query"]
        if queries[q] != qc[q]:
            raise SystemExit(f"câu {q}: câu trong pool đã chấm khác corrected_queries")
        for c in cs[:: max(1, len(cs) // args.check_per_query)][: args.check_per_query]:
            check.append({"id": q, "doc_id": str(c["doc_id"]), "expected": c["rerank_score"]})

    need = {d for ds in pairs.values() for d in ds} | {c["doc_id"] for c in check}
    print(f"[build] đã chấm {len(done)} câu | còn {len(todo)} câu, {sum(map(len, pairs.values())):,} cặp, "
          f"{len(need):,} doc | check {len(check)} cặp từ {len(check_q)} câu", flush=True)
    texts = kv.read_texts(args.texts, need)
    meta = {"model": MODEL, "kind": "cross-encoder", "max_length": MAX_LEN, "query_source": "corrected",
            "rr_text": "doc_texts_8k", "source_pool": args.pool, "done_pool": args.done}
    zpath = write_job(args.name, meta, queries, pairs, texts, check, dataset_dir=DATASET_DIR)
    print(f"[build] {zpath} ({os.path.getsize(zpath) / 1e6:.0f} MB)")
    kernels(args, dataset_create=True)


def kernels(args, dataset_create=False):
    """Dùng lại KERNEL_CE (mỗi GPU T4 một tiến trình rerank_job.py) với dataset riêng của job này."""
    kv.DATASET_DIR = DATASET_DIR
    kv.kernels(SimpleNamespace(name=args.name, owner=args.owner, dataset_slug=args.dataset_slug, backend="ce",
                               job_arg=args.job_arg, vllm_spec="-"), dataset_create=dataset_create)
    print(f"Sau khi tải điểm về {OUT}/{args.name}:  python -m scripts.kaggle_rerank_ft merge")


def merge(args):
    done = load_done(args.done)
    scores = {}
    for path in sorted(glob.glob(os.path.join(args.scores, "**", "scores_*.jsonl"), recursive=True)):
        for r in jsonl(path):
            scores[int(r["id"])] = r["scores"]
    qc = {int(r["id"]): r["corrected_query"] for r in jsonl(args.corrected)}
    rows, miss, n_k = [], [], 0
    for r in jsonl(args.pool):
        q = int(r["id"])
        if q in done:
            rows.append(done[q])
            continue
        s = scores.get(q)
        if s is None or any(s.get(str(c["doc_id"])) is None for c in r["candidates"]):
            miss.append(q)
            continue
        cands = [{**c, "rerank_score_prev": c.get("rerank_score"), "rerank_score": s[str(c["doc_id"])]}
                 for c in r["candidates"]]
        cands.sort(key=lambda c: c["rerank_score"], reverse=True)
        rows.append({**r, "query": qc[q], "query_orig": r["query"], "rr_model": MODEL, "rr_len": MAX_LEN,
                     "candidates": cands})
        n_k += 1
    if miss and not args.allow_partial:
        raise SystemExit(f"{len(miss)} câu chưa có điểm đủ (vd {miss[:10]}); chạy tiếp kernel hoặc --allow-partial")
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[merge] {len(rows)} câu ({len(done)} từ server + {n_k} từ Kaggle), thiếu {len(miss)} -> {args.out}")


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pool", default=POOL)
    common.add_argument("--done", default=DONE)
    common.add_argument("--corrected", default=CORRECTED)
    common.add_argument("--name", default=NAME)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", parents=[common])
    p.add_argument("--owner", default=None, help="kaggle username (lần đầu bắt buộc)")
    p.add_argument("--dataset-slug", default="vibiomir-rerank-ft")
    p.add_argument("--texts", default=TEXTS)
    p.add_argument("--limit", type=int, default=0, help="chỉ N câu đầu còn thiếu (chạy thử)")
    p.add_argument("--check-queries", type=int, default=4)
    p.add_argument("--check-per-query", type=int, default=8)
    p.add_argument("--job-arg", action="append", default=[], help="tham số thêm cho rerank_job.py")
    p.set_defaults(fn=build)
    p = sub.add_parser("kernels", parents=[common], help="sinh lại kernel mà không dựng lại zip")
    p.add_argument("--owner", default=None)
    p.add_argument("--dataset-slug", default="vibiomir-rerank-ft")
    p.add_argument("--job-arg", action="append", default=[])
    p.set_defaults(fn=kernels)
    p = sub.add_parser("merge", parents=[common])
    p.add_argument("--scores", default=f"{OUT}/{NAME}")
    p.add_argument("--out", default=FULL)
    p.add_argument("--allow-partial", action="store_true")
    p.set_defaults(fn=merge)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
