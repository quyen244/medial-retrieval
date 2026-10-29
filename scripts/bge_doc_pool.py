"""Pool top1000 bge-m3 (dense) trên `medical_docs` bằng câu SỬA - bản qcorr của doc_top1000_bge_pool.jsonl.

Cùng cách tìm với pool cũ (phase1_doc_retrieve: dense, không hybrid, hnsw_ef >= top), chỉ đổi câu hỏi.
Vector câu hỏi lấy lại từ scripts/c0_chunk_pool.py encode (bge-m3, câu sửa): data/C0_chunk_pool/q_dense.npy.
Không rerank: rrf_score = 1/(60 + hạng) như pool cũ, nên fuse_retrieval / kaggle_embed --pool dùng thẳng được.

    python -m scripts.bge_doc_pool --limit 20        # smoke
    python -m scripts.bge_doc_pool

Kiểm tra cuối lượt: với câu sửa TRÙNG câu gốc, top1000 mới phải trùng gần hết pool cũ.
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from qdrant_client import QdrantClient, models

D = "data/tmp/phase1"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", default="data/C0_chunk_pool", help="q_dense.npy + q_ids.json (bge-m3, câu sửa)")
    ap.add_argument("--collection", default="medical_docs")
    ap.add_argument("--top", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16, help="số câu mỗi query_batch_points")
    ap.add_argument("--old", default=f"{D}/doc_top1000_bge_pool.jsonl", help="pool câu gốc, để kiểm tra")
    ap.add_argument("--queries", default="data/spelling/corrected_queries.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    vecs = np.load(f"{args.qdir}/q_dense.npy")
    qids = json.load(open(f"{args.qdir}/q_ids.json"))
    qrow = {int(r["id"]): r for r in map(json.loads, open(args.queries, encoding="utf-8"))}
    order = list(range(len(qids)))[: args.limit or None]
    out = args.out or f"{D}/doc_top{args.top}_bge_retr_qcorr_pool.jsonl" + ("" if not args.limit else f".q{args.limit}")
    client = QdrantClient(url="http://localhost:6333", timeout=600)
    params = models.SearchParams(hnsw_ef=max(256, args.top))

    t0, last, res = time.time(), time.time(), {}
    for i in range(0, len(order), args.batch):
        idx = order[i: i + args.batch]
        reqs = [models.QueryRequest(query=vecs[j].tolist(), using="dense", limit=args.top, params=params,
                                    with_payload=["doc_id", "language", "title"]) for j in idx]
        for j, r in zip(idx, client.query_batch_points(args.collection, requests=reqs, timeout=600)):
            res[int(qids[j])] = r.points
        done = min(i + args.batch, len(order))
        if time.time() - last >= 60 or done == len(order):
            last = time.time()
            rate = done / (last - t0)
            print(f"[bge-pool] {done}/{len(order)} câu, {rate:.1f} câu/s, còn ~{(len(order) - done) / rate / 60:.0f} phút",
                  flush=True)

    with open(out, "w", encoding="utf-8") as fh:
        for q in sorted(res):
            cs = [{"doc_id": str(p.payload["doc_id"]), "language": p.payload.get("language"),
                   "title": p.payload.get("title"), "rrf_score": round(1.0 / (60 + k), 5),
                   "found_by": ["corrected"], "rerank_score": round(float(p.score), 6)}
                  for k, p in enumerate(res[q], 1)]
            fh.write(json.dumps({"id": q, "query": qrow[q].get("corrected_query") or qrow[q]["query"],
                                 "rr_model": "retrieval:bge-m3_dense_qcorr", "rr_len": 0, "candidates": cs},
                                ensure_ascii=False) + "\n")

    if os.path.exists(args.old):                 # câu sửa = câu gốc -> pool phải gần như trùng pool cũ
        old = {int(r["id"]): {c["doc_id"] for c in r["candidates"]}
               for r in map(json.loads, open(args.old, encoding="utf-8")) if int(r["id"]) in res}
        same, diff = [], []
        for q, pts in res.items():
            ov = len({str(p.payload["doc_id"]) for p in pts} & old.get(q, set())) / max(1, len(pts))
            (same if (qrow[q].get("corrected_query") or qrow[q]["query"]) == qrow[q]["query"] else diff).append(ov)
        print(f"[bge-pool] trùng pool cũ: câu không đổi {statistics.mean(same) if same else 0:.1%} (n={len(same)}) | "
              f"câu đã sửa {statistics.mean(diff) if diff else 0:.1%} (n={len(diff)})")
    print(f"[bge-pool] {len(res)} câu, top {args.top} -> {out}")


if __name__ == "__main__":
    main()
