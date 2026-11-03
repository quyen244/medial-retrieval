"""Chấm lại một pool doc bằng reranker khác, giữ nguyên tập ứng viên (T14, T05).

Pool vào là dump của scripts/phase1_doc_retrieve.py. Pool ra cùng định dạng, cùng
doc từng câu, chỉ thay `rerank_score` (điểm cũ giữ ở `rerank_score_prev`) và xếp
lại theo điểm mới. Biến duy nhất so với pool vào là reranker, nên bài dựng từ pool
ra (scripts/fuse_submission.py --doc-pool top300q4b) đo đúng hiệu ứng reranker.

Chấm bằng câu GỐC trên text doc trong payload `medical_docs`, như phase1_doc_retrieve.
Chạy dở thì chạy lại cùng lệnh: câu đã ghi ra --out được bỏ qua.
Cuối lượt in trung vị và p90 điểm theo ngôn ngữ doc (chẩn đoán T05).

    python -m scripts.rerank_pool --limit 5                      # đo tốc độ
    python -m scripts.rerank_pool
    python -m scripts.rerank_pool --set rerank.model=Qwen/Qwen3-Reranker-8B --out ...
"""
import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient

from scripts.ingest_doc_index import point_id
from src.rag.config import add_config_args, config_from_args

D = "data/tmp/phase1"


def language_report(path):
    """Trung vị / p90 điểm rerank theo ngôn ngữ doc, trên mọi cặp câu-doc của pool."""
    by = defaultdict(list)
    for r in map(json.loads, open(path, encoding="utf-8")):
        for c in r["candidates"]:
            by[c.get("language") or "vi"].append(c["rerank_score"])
    for lg in ("vi", "en", "zh"):
        v = sorted(by.get(lg, []))
        if v:
            print(f"[rerank_pool] {lg}  n={len(v):,}  median {statistics.median(v):.4f}  "
                  f"p90 {v[int(0.9 * (len(v) - 1))]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=f"{D}/doc_top300_bge_pool.jsonl")
    ap.add_argument("--out", default=f"{D}/doc_top300_qwen3r4b_pool.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="chỉ N câu đầu, để đo tốc độ")
    add_config_args(ap)
    args = ap.parse_args(["--set", "rerank.model=Qwen/Qwen3-Reranker-4B"] + sys.argv[1:])

    cfg = config_from_args(args)
    coll = cfg.get("qdrant.collection_docs", "medical_docs")
    rr_model = cfg.get("rerank.model")
    rr_len = cfg.get("rerank.max_length", 1024)

    rows = [json.loads(l) for l in open(args.pool, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    done = set()
    if os.path.exists(args.out):
        for r in map(json.loads, open(args.out, encoding="utf-8")):
            if (r["rr_model"], r["rr_len"]) != (rr_model, rr_len):
                raise SystemExit(f"{args.out} đã có điểm của {r['rr_model']} len={r['rr_len']}; ghi ra --out khác")
            done.add(int(r["id"]))
    todo = [r for r in rows if int(r["id"]) not in done]
    n_pairs = sum(len(r["candidates"]) for r in todo)
    print(f"[rerank_pool] {args.pool} -> {args.out} | {rr_model} len={rr_len} "
          f"| {len(todo)}/{len(rows)} câu, {n_pairs:,} cặp")

    if todo:
        from src.rag.pipeline.components.reranking import make_reranker
        reranker = make_reranker(model_name=rr_model, max_length=rr_len)
        client = QdrantClient(url=cfg.get("qdrant.url", "http://localhost:6333"),
                              timeout=cfg.get("qdrant.timeout", 600))
        t0, scored = time.time(), 0
        with open(args.out, "a", encoding="utf-8") as fh:
            for i, row in enumerate(todo, 1):
                cands = row["candidates"]
                text_of = {str((p.payload or {}).get("doc_id")): (p.payload or {}).get("text") or ""
                           for p in client.retrieve(collection_name=coll,
                                                    ids=[point_id(c["doc_id"]) for c in cands],
                                                    with_payload=["doc_id", "text"])}
                missing = [c["doc_id"] for c in cands if not text_of.get(c["doc_id"])]
                if missing:
                    raise SystemExit(f"câu {row['id']}: {len(missing)} doc không có text trong {coll}: {missing[:5]}")
                scores = reranker.score_pairs([(row["query"], text_of[c["doc_id"]]) for c in cands])
                new = [{**c, "rerank_score_prev": c["rerank_score"], "rerank_score": s}
                       for c, s in zip(cands, scores)]
                new.sort(key=lambda c: c["rerank_score"], reverse=True)
                fh.write(json.dumps({**row, "rr_model": rr_model, "rr_len": rr_len, "candidates": new},
                                    ensure_ascii=False) + "\n")
                fh.flush()
                scored += len(cands)
                if i % 20 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"[rerank_pool] {i}/{len(todo)} câu | {scored / el:.0f} cặp/s "
                          f"| còn ~{(n_pairs - scored) / max(scored / el, 1e-9) / 60:.0f} phút", flush=True)
    language_report(args.out)


if __name__ == "__main__":
    main()
