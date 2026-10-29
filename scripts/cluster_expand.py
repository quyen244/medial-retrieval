"""Mở rộng cụm: thêm vào pool các doc GẦN CÁC DOC TỐT NHẤT (seed) thay vì gần câu hỏi.

Gold ~51 doc/câu là một cụm cùng chủ đề; doc gold ít giống câu hỏi nhưng giống các doc gold khác
thì nằm ngoài pool. Seed = top-k của một pool ĐÃ rerank (mặc định T29). Vector: bge-m3 dense của đủ 1M doc
(data/T31_pool_expand/docs_dense_f16.npy, scripts/dense_dump.py dump), không cần q3e, chạy CPU.

  a       cách A: tâm cụm = trung bình vector k seed (chuẩn hoá) -> top doc gần tâm, bỏ doc đã có trong pool,
          giữ --add doc mới (điểm 1/(60+hạng) x trọng số ngôn ngữ). Ra pool = pool cũ + doc mới (rerank_score None),
          đưa thẳng vào rerank_pool --reuse <pool cũ> để chỉ chấm doc mới.
  b       cách B: kNN từng seed, doc được nhiều seed chọn thì điểm cao (số phiếu x trọng số ngôn ngữ); ra pool như a.
  bench-b đo thời gian cách B trên mẫu seed rồi ngoại suy.

    python -m scripts.cluster_expand a --k 20 --add 50
    python -m scripts.cluster_expand b --k 20 --knn 100 --add 50
    python -m scripts.cluster_expand bench-b --k 20 --sample 240
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

D = "data/T31_pool_expand"
POOL = "data/T28_rrf_bge_q3e/w0.6_qc_vi90_bge_pool.jsonl"


def load_vecs():
    vecs = np.load(f"{D}/docs_dense_f16.npy", mmap_mode="r")
    meta = [json.loads(l) for l in open(f"{D}/docs_meta.jsonl", encoding="utf-8")]
    return vecs, meta


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def sims(q, vecs, block, label):
    """q (n, 1024) đã chuẩn hoá -> cosine (n, N) fp16 với mọi doc, theo khối doc như dense_dump.search."""
    out = np.zeros((len(q), len(vecs)), dtype=np.float16)
    t0, last = time.time(), time.time()
    for a in range(0, len(vecs), block):
        out[:, a: a + block] = q @ unit(vecs[a: a + block]).T
        if time.time() - last >= 60:
            last = time.time()
            done = min(a + block, len(vecs))
            print(f"[{label}] {done:,}/{len(vecs):,} doc, còn ~{(len(vecs) - done) / (done / (last - t0)) / 60:.0f} phút",
                  flush=True)
    return out, time.time() - t0


def seeds_of(rows, k):
    return {int(r["id"]): [c["doc_id"] for c in r["candidates"][:k]] for r in rows}


def cmd_a(args):
    rows = [json.loads(l) for l in open(args.pool, encoding="utf-8")]
    vecs, meta = load_vecs()
    row_of = {m["doc_id"]: i for i, m in enumerate(meta)}
    lw = {"vi": 1.0, **{k: float(v) for k, v in (s.split("=") for s in args.lang_w)}}
    seeds = seeds_of(rows, args.k)
    cent = np.stack([unit(vecs[[row_of[d] for d in seeds[int(r["id"])] if d in row_of]]).mean(0) for r in rows])
    s, sec = sims(unit(cent), vecs, args.block, "cụm A")
    print(f"[cụm A] cosine {len(rows)} tâm x {len(meta):,} doc trong {sec:.0f}s", flush=True)

    bge_rank = {}
    if os.path.exists(args.bge_pool):
        bge_rank = {int(r["id"]): {c["doc_id"]: i for i, c in enumerate(r["candidates"], 1)}
                    for r in map(json.loads, open(args.bge_pool, encoding="utf-8"))}
    out = args.out or f"{D}/cluster_a_k{args.k}_add{args.add}_pool.jsonl"
    n_new, deep, langs, cos_new = [], [], {}, []
    with open(out, "w", encoding="utf-8") as fh:
        for j, r in enumerate(rows):
            q = int(r["id"])
            have = {c["doc_id"] for c in r["candidates"]}
            sj = s[j].astype(np.float32)
            top = np.argpartition(-sj, args.search)[: args.search]
            top = top[np.argsort(-sj[top])]
            new = []
            for rank, i in enumerate(top, 1):
                d, lg = meta[i]["doc_id"], meta[i]["language"] or "vi"
                if d in have:
                    continue
                new.append({"doc_id": d, "language": lg, "rrf_score": round(lw.get(lg, 1.0) / (60 + rank), 6),
                            "found_by": ["cluster_a"], "cluster_cos": round(float(sj[i]), 4), "rerank_score": None})
            new.sort(key=lambda c: -c["rrf_score"])
            new = new[: args.add]
            n_new.append(len(new))
            deep.append(sum(1 for c in new if bge_rank.get(q, {}).get(c["doc_id"], 10 ** 9) > 1000))
            for c in new:
                langs[c["language"]] = langs.get(c["language"], 0) + 1
                cos_new.append(c["cluster_cos"])
            fh.write(json.dumps({**r, "candidates": r["candidates"] + new}, ensure_ascii=False) + "\n")
    tot = sum(langs.values())
    print(f"[cụm A] {statistics.mean(n_new):.1f} doc mới/câu | ngoài bge top1000: {statistics.mean(deep):.1f}/câu "
          f"(theo {args.bge_pool}) | ngôn ngữ " + " ".join(f"{k} {v / tot:.0%}" for k, v in sorted(langs.items()))
          + f" | cosine với tâm median {statistics.median(cos_new):.3f} -> {out}")


def cmd_b(args):
    """Cách B: mỗi seed lấy --knn hàng xóm gần nhất; điểm doc = số seed chọn nó (hoà thì tổng cosine),
    nhân trọng số ngôn ngữ. Seed chia lô --lot để ma trận điểm vừa RAM (lô x 1M x 2 byte)."""
    rows = [json.loads(l) for l in open(args.pool, encoding="utf-8")]
    vecs, meta = load_vecs()
    row_of = {m["doc_id"]: i for i, m in enumerate(meta)}
    lw = {"vi": 1.0, **{k: float(v) for k, v in (s.split("=") for s in args.lang_w)}}
    seeds = seeds_of(rows, args.k)
    flat = [(int(r["id"]), row_of[d]) for r in rows for d in seeds[int(r["id"])] if d in row_of]
    votes = {}                                   # qid -> {doc_row: [số phiếu, tổng cosine]}
    t0, last = time.time(), time.time()
    for a in range(0, len(flat), args.lot):
        lot = flat[a: a + args.lot]
        s, _ = sims(unit(vecs[[i for _, i in lot]]), vecs, args.block, "cụm B")
        for (q, _), x in zip(lot, s):
            x = x.astype(np.float32)
            nb = np.argpartition(-x, args.knn)[: args.knn]
            v = votes.setdefault(q, {})
            for i in nb:
                e = v.setdefault(int(i), [0, 0.0])
                e[0] += 1
                e[1] += float(x[i])
        if time.time() - last >= 60 or a + args.lot >= len(flat):
            last = time.time()
            done = min(a + args.lot, len(flat))
            rate = done / (last - t0)
            print(f"[cụm B] {done:,}/{len(flat):,} seed, {rate:.0f} seed/s, còn ~{(len(flat) - done) / rate / 60:.0f} phút",
                  flush=True)

    bge_rank = {}
    if os.path.exists(args.bge_pool):
        bge_rank = {int(r["id"]): {c["doc_id"]: i for i, c in enumerate(r["candidates"], 1)}
                    for r in map(json.loads, open(args.bge_pool, encoding="utf-8"))}
    out = args.out or f"{D}/cluster_b_k{args.k}_knn{args.knn}_add{args.add}_pool.jsonl"
    n_new, deep, langs, nvote = [], [], {}, []
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            q = int(r["id"])
            have = {c["doc_id"] for c in r["candidates"]}
            cand = []
            for i, (n, cs) in votes.get(q, {}).items():
                d, lg = meta[i]["doc_id"], meta[i]["language"] or "vi"
                if d not in have:
                    cand.append((n * lw.get(lg, 1.0), cs, d, lg, n))
            cand.sort(reverse=True)
            new = [{"doc_id": d, "language": lg, "rrf_score": round(sc / args.k, 6), "found_by": ["cluster_b"],
                    "cluster_votes": n, "rerank_score": None} for sc, _, d, lg, n in cand[: args.add]]
            n_new.append(len(new))
            deep.append(sum(1 for c in new if bge_rank.get(q, {}).get(c["doc_id"], 10 ** 9) > 1000))
            for c in new:
                langs[c["language"]] = langs.get(c["language"], 0) + 1
                nvote.append(c["cluster_votes"])
            fh.write(json.dumps({**r, "candidates": r["candidates"] + new}, ensure_ascii=False) + "\n")
    tot = sum(langs.values())
    print(f"[cụm B] {statistics.mean(n_new):.1f} doc mới/câu | ngoài bge top1000: {statistics.mean(deep):.1f}/câu | "
          f"ngôn ngữ " + " ".join(f"{k} {v / tot:.0%}" for k, v in sorted(langs.items()))
          + f" | số phiếu median {statistics.median(nvote)}/{args.k} | {time.time() - t0:.0f}s -> {out}")


def cmd_bench_b(args):
    rows = [json.loads(l) for l in open(args.pool, encoding="utf-8")]
    vecs, meta = load_vecs()
    row_of = {m["doc_id"]: i for i, m in enumerate(meta)}
    seeds = [d for ds in seeds_of(rows, args.k).values() for d in ds if d in row_of]
    sample = unit(vecs[[row_of[d] for d in seeds[: args.sample]]])
    s, sec = sims(sample, vecs, args.block, "cụm B")
    t = time.time()
    for x in s:                                  # kNN mỗi seed: phần chọn top-knn cũng phải tính
        np.argpartition(-x.astype(np.float32), args.knn)[: args.knn]
    sec_top = time.time() - t
    per = (sec + sec_top) / len(sample)
    total = per * len(seeds)
    print(f"[cụm B] {len(sample)} seed: cosine {sec:.0f}s + top{args.knn} {sec_top:.0f}s = {per:.2f}s/seed | "
          f"{len(seeds):,} seed ({len(rows)} câu x k={args.k}) -> ~{total / 60:.0f} phút CPU "
          f"(RAM điểm: {len(sample) * len(meta) * 2 / 1e9:.1f} GB cho {len(sample)} seed/lô)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("a", "b", "bench-b"):
        p = sub.add_parser(name)
        p.add_argument("--pool", default=POOL, help="pool ĐÃ rerank (xếp theo rerank_score): seed = top-k")
        p.add_argument("--k", type=int, default=20)
        p.add_argument("--block", type=int, default=100000)
    for name in ("a", "b"):
        p = sub.choices[name]
        p.add_argument("--add", type=int, default=50, help="số doc mới tối đa mỗi câu")
        p.add_argument("--lang-w", nargs="*", default=["en=0.4", "zh=0.4"])
        p.add_argument("--bge-pool", default=f"{D}/doc_top5000_bge_retr_qcorr_pool.jsonl",
                       help="để đếm doc mới ở hạng bge >1000")
        p.add_argument("--out", default=None)
    sub.choices["a"].add_argument("--search", type=int, default=1000, help="số doc gần tâm xét trước khi bỏ doc đã có")
    sub.choices["b"].add_argument("--knn", type=int, default=100, help="số hàng xóm mỗi seed")
    sub.choices["b"].add_argument("--lot", type=int, default=1200, help="số seed mỗi lô (RAM ~ lô x 2 MB)")
    p = sub.choices["bench-b"]
    p.add_argument("--sample", type=int, default=240)
    p.add_argument("--knn", type=int, default=100)
    args = ap.parse_args()
    {"a": cmd_a, "b": cmd_b, "bench-b": cmd_bench_b}[args.cmd](args)


if __name__ == "__main__":
    main()
