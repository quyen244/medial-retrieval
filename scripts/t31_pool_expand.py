"""T31 - cache offline để chọn độ sâu pool và nguồn ứng viên mới (không nộp, không GPU).

Đầu vào chính: pool bge-m3 dense top5000 câu sửa (scripts/bge_doc_pool --top 5000).

    python -m scripts.t31_pool_expand coverage   # (2)+(3) doc pool 5000 đã có vector q3e chưa, danh sách ưu tiên embed
    python -m scripts.t31_pool_expand rrf        # (4) RRF bge + q3e trên pool 5000: bao nhiêu doc hạng bge > 1000 lọt top300
    python -m scripts.t31_pool_expand sparse     # (5) list sparse bge-m3 top1000 trên medical_docs (câu sửa)
    python -m scripts.t31_pool_expand summary    # (7)+(8) gộp nguồn: dense 1000/3000/5000, sparse, C0 chunk -> doc mới
    python -m scripts.t31_pool_expand t32        # T32a (dense 3000 + q3e) / T32b (+ sparse): pool top300, đếm doc mới cần rerank

Mọi file ra ở data/T31_pool_expand/.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

D = "data/T31_pool_expand"
POOL = f"{D}/doc_top5000_bge_retr_qcorr_pool.jsonl"
Q3E_DIRS = ["data/kaggle/out/emb_full_q3e4b_p0of8/q3e4b", "data/kaggle/out/emb_full_q3e4b_p2of8/q3e4b",
            "data/kaggle/out/emb_full_q3e4b_server/q3e4b"]
Q3E_CANDS = f"{D}/cands_top5000_q3e4b_p1_own5000bgeqc_qcorr.jsonl"
SPARSE = f"{D}/doc_top1000_bgesparse_retr_qcorr.jsonl"
C0 = "data/C0_chunk_pool/search_top1000.jsonl"
TIERS = [(1, 1000), (1001, 3000), (3001, 5000)]
TOK_PER_DOC, TOK_PER_S = 1200, 5700          # đo ở job embed q3e4b vLLM (TB ~1.000-1.400 token/doc, 5.700 token/s)


def jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:            # search_top1000.jsonl có một dòng cụt ở cuối (lần ghi cũ)
                    print(f"[t31] bỏ dòng hỏng trong {path}: {line[:40]!r}")


def bge_ranks():
    """id -> [(doc_id, language)] theo hạng bge dense."""
    return {int(r["id"]): [(c["doc_id"], c.get("language")) for c in r["candidates"]] for r in jsonl(POOL)}


def embedded_ids():
    ids = set()
    for d in Q3E_DIRS:
        for f in glob.glob(f"{d}/docs_*.ids.npy"):
            ids.update(str(x) for x in np.load(f, allow_pickle=True).tolist())
    return ids


def tier_of(rank):
    return next(f"{a}-{b}" for a, b in TIERS if a <= rank <= b)


def cmd_coverage(args):
    pool, emb = bge_ranks(), embedded_ids()
    print(f"[t31] {len(emb):,} doc đã có vector q3e")
    for a, b in TIERS:
        docs = {d for cs in pool.values() for d, _ in cs[a - 1: b]}
        per_q = statistics.mean(sum(d in emb for d, _ in cs[a - 1: b]) / max(1, len(cs[a - 1: b])) for cs in pool.values())
        print(f"[t31] hạng {a}-{b}: {len(docs):,} doc khác nhau, có vector {sum(d in emb for d in docs) / len(docs):.1%} "
              f"(TB mỗi câu {per_q:.1%})")
    # ưu tiên embed: doc thiếu vector, xếp theo hạng tốt nhất của nó trên mọi câu (hạng nông trước)
    best = {}
    for cs in pool.values():
        for i, (d, _) in enumerate(cs, 1):
            if d not in emb and i < best.get(d, 10 ** 9):
                best[d] = i
    order = sorted(best, key=best.get)
    with open(f"{D}/embed_priority.txt", "w") as fh:
        fh.write("\n".join(order) + "\n")
    for cut in (1000, 3000, 5000):
        n = sum(1 for d in order if best[d] <= cut)
        print(f"[t31] cần embed thêm để phủ đủ top{cut}: {n:,} doc ~ {n * TOK_PER_DOC / TOK_PER_S / 3600:.1f} giờ GPU")
    print(f"[t31] -> {D}/embed_priority.txt ({len(order):,} doc, hạng nông trước)")


def cmd_rrf(args):
    if not os.path.exists(Q3E_CANDS):
        import subprocess
        subprocess.run([sys.executable, "-m", "scripts.kaggle_embed", "search", "--vecs", *Q3E_DIRS, "--instruction", "p1",
                        "--retriever", "q3e4b_p1_own5000bgeqc", "--top-k", "5000", "--pool", POOL, "--pool-mode", "own",
                        "--out", Q3E_CANDS], check=True)
    pool = bge_ranks()
    q3e = {int(r["id"]): r["doc_ids"] for r in jsonl(Q3E_CANDS)}
    for depth in (1000, 3000, 5000):
        for lw in (1.0, 0.4):
            deep, nonvi, keep = [], [], []
            for q, cs in pool.items():
                cs = cs[:depth]
                lang = dict(cs)
                rb = {d: i for i, (d, _) in enumerate(cs, 1)}
                rq = {d: i for i, d in enumerate((d for d in q3e[q] if d in rb), 1)}
                worst = len(cs) + 1
                sc = {d: (0.6 / (60 + rq.get(d, worst)) + 0.4 / (60 + rb[d])) * (lw if lang[d] in ("en", "zh") else 1.0)
                      for d in rb}
                top = sorted(sc, key=lambda d: -sc[d])[:300]
                deep.append(sum(rb[d] > 1000 for d in top))
                nonvi.append(sum(lang[d] in ("en", "zh") for d in top))
                keep.append(len(set(top) & {d for d, _ in pool[q][:1000]}))
            print(f"[t31] pool {depth}, en/zh x{lw:g}: top300 có {statistics.mean(deep):.1f} doc hạng bge > 1000 "
                  f"(max {max(deep)}), trùng top300 của pool 1000 TB {statistics.mean(keep):.1f}, en/zh {statistics.mean(nonvi):.1f}")
    print("[t31] lưu ý: doc chưa có vector q3e bị tính hạng q3e cuối, nên số doc sâu lọt top300 là CẬN DƯỚI")


def cmd_sparse(args):
    from qdrant_client import QdrantClient, models
    qs = {int(r["id"]): r["w"] for r in jsonl("data/C0_chunk_pool/q_sparse.jsonl")}
    client = QdrantClient(url="http://localhost:6333", timeout=600)
    t0, last = time.time(), time.time()
    with open(SPARSE, "w", encoding="utf-8") as fh:
        for i, (q, w) in enumerate(sorted(qs.items()), 1):
            vec = models.SparseVector(indices=[int(k) for k in w], values=[float(v) for v in w.values()])
            pts = client.query_points("medical_docs", query=vec, using="sparse", limit=args.top,
                                      with_payload=["doc_id", "language"], timeout=600).points
            fh.write(json.dumps({"id": q, "candidates": [{"doc_id": str(p.payload["doc_id"]), "language": p.payload.get("language"),
                                                          "score": round(float(p.score), 5)} for p in pts]}) + "\n")
            if time.time() - last >= 60 or i == len(qs):
                last = time.time()
                rate = i / (last - t0)
                print(f"[t31] sparse {i}/{len(qs)} câu, {rate:.1f} câu/s, còn ~{(len(qs) - i) / rate / 60:.0f} phút", flush=True)
    print(f"[t31] -> {SPARSE}")


def cmd_summary(args):
    pool = bge_ranks()
    rank = {q: {d: i for i, (d, _) in enumerate(cs, 1)} for q, cs in pool.items()}
    lang = {d: lg for cs in pool.values() for d, lg in cs}
    sources = {}
    if os.path.exists(SPARSE):
        for r in jsonl(SPARSE):
            sources.setdefault("sparse top1000", {})[int(r["id"])] = [c["doc_id"] for c in r["candidates"]]
            lang.update({c["doc_id"]: c["language"] for c in r["candidates"]})
    if os.path.exists(C0):
        for r in jsonl(C0):
            seen = []
            for c in r["chunks"]:
                if c["doc_id"] not in seen:
                    seen.append(c["doc_id"])
                lang.setdefault(c["doc_id"], c.get("language"))
            sources.setdefault("C0 chunk top1000 -> doc", {})[int(r["id"])] = seen
    rows = []
    for name, by_q in sources.items():
        n, new1k, new5k, in_deep, nonvi_new = [], [], [], [], []
        for q, docs in by_q.items():
            rk = rank.get(q, {})
            n.append(len(docs))
            out1k = [d for d in docs if rk.get(d, 10 ** 9) > 1000]
            new1k.append(len(out1k))
            in_deep.append(sum(1 for d in out1k if d in rk))
            new5k.append(sum(1 for d in docs if d not in rk))
            nonvi_new.append(sum(1 for d in out1k if lang.get(d) in ("en", "zh")))
        rows.append((name, statistics.mean(n), statistics.mean(new1k), statistics.mean(in_deep), statistics.mean(new5k),
                     statistics.mean(nonvi_new)))
    print("| nguồn | doc/câu | ngoài dense top1000 | trong đó nằm ở dense 1001-5000 | ngoài cả dense top5000 | en/zh trong phần ngoài top1000 |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r[0]} | {r[1]:.0f} | {r[2]:.1f} | {r[3]:.1f} | {r[4]:.1f} | {r[5]:.1f} |")
    # hợp mọi nguồn ngoài dense top1000
    union = []
    for q in pool:
        extra = set()
        for by_q in sources.values():
            extra |= {d for d in by_q.get(q, []) if rank[q].get(d, 10 ** 9) > 1000}
        union.append(len(extra))
    print(f"[t31] hợp sparse + C0 ngoài dense top1000: TB {statistics.mean(union):.1f} doc/câu "
          f"(rerank bge số cặp thêm ~{sum(union):,})")


T29_POOL = "data/T28_rrf_bge_q3e/w0.6_qc_vi90_pool.jsonl"
SCORED = ["data/T28_rrf_bge_q3e/w0.6_qc_bge_partial.jsonl", "data/T28_rrf_bge_q3e/w0.6_qc_vi90_bge_pool.jsonl"]
UNION_POOL = f"{D}/doc_union_dense3000_sparse1000_qcorr_pool.jsonl"
Q3E_UNION = f"{D}/cands_q3e4b_p1_own_union3000s1000_qcorr.jsonl"


def rrf_top(lists, weights, lang, top=300, k=60, lw=0.4):
    """lists: {tên: [doc theo hạng]}; doc vắng ở một list nhận hạng cuối của list đó (len + 1)."""
    pool = set().union(*map(set, lists.values()))
    rank = {n: {d: i for i, d in enumerate(ds, 1)} for n, ds in lists.items()}
    sc = {d: sum(w / (k + rank[n].get(d, len(lists[n]) + 1)) for n, w in weights.items())
             * (lw if lang.get(d) in ("en", "zh") else 1.0) for d in pool}
    return sorted(sc, key=lambda d: -sc[d])[:top], sc


def cmd_t32(args):
    import subprocess
    dense = {q: cs[: args.depth] for q, cs in bge_ranks().items()}
    lang = {d: lg for cs in dense.values() for d, lg in cs}
    sparse = {}
    for r in jsonl(SPARSE):
        sparse[int(r["id"])] = [c["doc_id"] for c in r["candidates"]]
        lang.update({c["doc_id"]: c["language"] for c in r["candidates"]})
    if not os.path.exists(UNION_POOL):          # pool hợp dense + sparse để q3e xếp lại (chế độ own)
        with open(UNION_POOL, "w", encoding="utf-8") as fh:
            for q, cs in dense.items():
                ds = [d for d, _ in cs] + [d for d in sparse[q] if d not in {d for d, _ in cs}]
                fh.write(json.dumps({"id": q, "candidates": [{"doc_id": d} for d in ds]}) + "\n")
    if not os.path.exists(Q3E_UNION):
        subprocess.run([sys.executable, "-m", "scripts.kaggle_embed", "search", "--vecs", *Q3E_DIRS, "--instruction", "p1",
                        "--retriever", "q3e4b_p1_own_union3000s1000", "--top-k", "4000", "--pool", UNION_POOL,
                        "--pool-mode", "own", "--out", Q3E_UNION], check=True)
    q3e = {int(r["id"]): r["doc_ids"] for r in jsonl(Q3E_UNION)}
    t29 = {int(r["id"]): [c["doc_id"] for c in r["candidates"]] for r in jsonl(T29_POOL)}
    scored = {}
    for path in SCORED:
        if os.path.exists(path):
            for r in jsonl(path):
                scored.setdefault(int(r["id"]), set()).update(c["doc_id"] for c in r["candidates"])
    for q in t29:                                # T29 top300 sẽ được chấm hết khi job rerank xong
        scored.setdefault(q, set()).update(t29[q])
    variants = {"T32a": {"q3e": 0.6, "dense": 0.4}}
    for wq, wd, ws in args.weights:
        variants[f"T32b_q{wq:g}_d{wd:g}_s{ws:g}"] = {"q3e": wq, "dense": wd, "sparse": ws}
    print(f"| biến thể | trùng top300 T29 | doc hạng dense > 1000 | doc chỉ sparse thấy (ngoài dense {args.depth}) | en/zh | cặp phải rerank mới |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, w in variants.items():
        ov, deep, sp_only, nonvi, new, rows = [], [], [], [], 0, []
        for q in sorted(dense):
            dl = [d for d, _ in dense[q]]
            lists = {"dense": dl, "q3e": [d for d in q3e[q] if name != "T32a" or d in set(dl)]}
            if "sparse" in w:
                lists["sparse"] = sparse[q]
            top, sc = rrf_top(lists, w, lang)
            rd = {d: i for i, d in enumerate(dl, 1)}
            ov.append(len(set(top) & set(t29[q])))
            deep.append(sum(1000 < rd.get(d, 10 ** 9) <= args.depth for d in top))
            sp_only.append(sum(d not in rd for d in top))
            nonvi.append(sum(lang.get(d) in ("en", "zh") for d in top))
            new += sum(d not in scored.get(q, set()) for d in top)
            rows.append({"id": q, "query": "", "rr_model": f"retrieval:{name}", "rr_len": 0,
                         "candidates": [{"doc_id": d, "language": lang.get(d), "rrf_score": round(1.0 / (60 + i), 5),
                                         "found_by": [name], "rerank_score": round(sc[d], 7)} for i, d in enumerate(top, 1)]})
        with open(f"{D}/{name}_pool.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"| {name} | {statistics.mean(ov):.1f} | {statistics.mean(deep):.1f} | {statistics.mean(sp_only):.1f} | "
              f"{statistics.mean(nonvi):.1f} | {new:,} (~{new / 55 / 60:.0f} phút bge) |")
    print(f"[t32] pool top300 ghi ở {D}/<biến thể>_pool.jsonl; q3e thiếu vector -> hạng cuối (cận dưới)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("coverage")
    sub.add_parser("rrf")
    p = sub.add_parser("sparse")
    p.add_argument("--top", type=int, default=1000)
    sub.add_parser("summary")
    p = sub.add_parser("t32")
    p.add_argument("--depth", type=int, default=3000)
    p.add_argument("--weights", type=lambda x: tuple(map(float, x.split(","))), nargs="*",
                   default=[(0.5, 0.3, 0.2), (0.5, 0.35, 0.15), (0.45, 0.3, 0.25)], help="bộ q3e,dense,sparse cho T32b")
    args = ap.parse_args()
    os.makedirs(D, exist_ok=True)
    {"coverage": cmd_coverage, "rrf": cmd_rrf, "sparse": cmd_sparse, "summary": cmd_summary, "t32": cmd_t32}[args.cmd](args)


if __name__ == "__main__":
    main()
