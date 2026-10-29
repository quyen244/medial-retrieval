"""Xuất 1M vector dense bge-m3 của `medical_docs` ra .npy (fp16) rồi tìm top-K CHÍNH XÁC bằng numpy.

Thay cho HNSW khi K lớn (top5000): HNSW phải đọc ngẫu nhiên rất nhiều vector, máy thiếu RAM thì nghẽn đĩa;
scroll đọc tuần tự một lần (~4 GB) rồi nhân ma trận trên CPU. Cache vector dùng lại được (mở rộng cụm sau này).

    python -m scripts.dense_dump dump                      # -> data/T31_pool_expand/docs_dense_f16.npy + docs_meta.jsonl
    python -m scripts.dense_dump search --top 5000         # -> doc_top5000_bge_retr_qcorr_pool.jsonl (định dạng như bge_doc_pool)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

D = "data/T31_pool_expand"


def cmd_dump(args):
    from qdrant_client import QdrantClient
    client = QdrantClient(url="http://localhost:6333", timeout=600)
    n = client.count(args.collection, exact=True).count
    vecs = np.zeros((n, 1024), dtype=np.float16)
    meta, off, i, t0, last = [], None, 0, time.time(), time.time()
    while True:
        pts, off = client.scroll(args.collection, limit=args.page, offset=off, with_vectors=["dense"],
                                 with_payload=["doc_id", "language"])
        for p in pts:
            vecs[i] = np.asarray(p.vector["dense"], dtype=np.float32)
            meta.append((str(p.payload["doc_id"]), p.payload.get("language")))
            i += 1
        if time.time() - last >= 60 or off is None:
            last = time.time()
            rate = i / (last - t0)
            print(f"[dump] {i:,}/{n:,} vector, {rate:,.0f}/s, còn ~{(n - i) / rate / 60:.0f} phút", flush=True)
        if off is None:
            break
    np.save(f"{D}/docs_dense_f16.npy", vecs[:i])
    with open(f"{D}/docs_meta.jsonl", "w", encoding="utf-8") as fh:
        for d, lg in meta:
            fh.write(json.dumps({"doc_id": d, "language": lg}) + "\n")
    print(f"[dump] {i:,} vector -> {D}/docs_dense_f16.npy")


def cmd_search(args):
    vecs = np.load(f"{D}/docs_dense_f16.npy", mmap_mode="r")
    meta = [json.loads(l) for l in open(f"{D}/docs_meta.jsonl", encoding="utf-8")]
    q = np.load("data/C0_chunk_pool/q_dense.npy").astype(np.float32)
    qids = json.load(open("data/C0_chunk_pool/q_ids.json"))
    qrow = {int(r["id"]): r for r in map(json.loads, open(args.queries, encoding="utf-8"))}
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    scores = np.zeros((len(q), len(meta)), dtype=np.float16)
    t0 = time.time()
    for a in range(0, len(meta), args.block):      # khối doc: fp16 -> fp32, chuẩn hoá (Cosine), nhân với mọi câu
        blk = np.asarray(vecs[a: a + args.block], dtype=np.float32)
        blk /= np.maximum(np.linalg.norm(blk, axis=1, keepdims=True), 1e-9)
        scores[:, a: a + len(blk)] = q @ blk.T
    print(f"[search] cosine {len(q)} x {len(meta):,} trong {time.time() - t0:.0f}s", flush=True)
    out = args.out or f"{D}/doc_top{args.top}_bge_retr_qcorr_pool.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for j, qid in enumerate(qids):
            s = scores[j].astype(np.float32)
            top = np.argpartition(-s, args.top)[: args.top]
            top = top[np.argsort(-s[top])]
            cs = [{"doc_id": meta[k]["doc_id"], "language": meta[k]["language"], "rrf_score": round(1.0 / (60 + r), 5),
                   "found_by": ["corrected"], "rerank_score": round(float(s[k]), 6)} for r, k in enumerate(top, 1)]
            fh.write(json.dumps({"id": int(qid), "query": qrow[int(qid)].get("corrected_query") or qrow[int(qid)]["query"],
                                 "rr_model": "retrieval:bge-m3_dense_qcorr_exact", "rr_len": 0, "candidates": cs},
                                ensure_ascii=False) + "\n")
    old = {int(r["id"]): {c["doc_id"] for c in r["candidates"]}
           for r in map(json.loads, open(args.check, encoding="utf-8"))}
    new = {int(r["id"]): [c["doc_id"] for c in r["candidates"]][:1000] for r in map(json.loads, open(out, encoding="utf-8"))}
    ov = np.mean([len(set(new[q]) & old[q]) / 1000 for q in new if q in old])
    print(f"[search] top1000 exact trùng top1000 HNSW ({args.check}): {ov:.1%} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dump")
    p.add_argument("--collection", default="medical_docs")
    p.add_argument("--page", type=int, default=2000)
    p = sub.add_parser("search")
    p.add_argument("--top", type=int, default=5000)
    p.add_argument("--block", type=int, default=100000)
    p.add_argument("--queries", default="data/spelling/corrected_queries.jsonl")
    p.add_argument("--check", default="data/tmp/phase1/doc_top1000_bge_retr_qcorr_pool.jsonl")
    p.add_argument("--out", default=None)
    args = ap.parse_args()
    os.makedirs(D, exist_ok=True)
    {"dump": cmd_dump, "search": cmd_search}[args.cmd](args)


if __name__ == "__main__":
    main()
