"""Gộp HẠNG TÌM của bge-m3 và Qwen3-Embedding-4B trong pool top1000 bge của từng câu (T28), không rerank.

    python -m scripts.fuse_retrieval --w-q3e 0.5 --top-k 300
    python -m scripts.fuse_retrieval --w-q3e 0.5 --lang-w en=0.8 zh=0.8           # hạ điểm doc en/zh
    python -m scripts.fuse_retrieval --w-q3e 1.0 --max-nonvi 0.15 --top-k 300   # chỉ q3e, trần doc en/zh

điểm = w/(k + hạng_q3e) + (1 - w)/(k + hạng_bge). w = 1 là q3e thuần (T26a), w = 0 là bge thuần (R300_top300).
Hạng bge suy từ rrf_score = 1/(60 + hạng) của pool top1000 (làm tròn 5 số lẻ: sai vài hạng ở đuôi ~1000).
--lang-w en=a zh=b: nhân điểm gộp của doc theo ngôn ngữ (vi giữ 1); a < 1 là hạ doc en.
--max-nonvi f: top-k giữ tối đa f*top-k doc en/zh, chỗ trống lấp bằng doc vi kế tiếp.

Ghi pool dạng retr-pool (rerank_score = điểm gộp) vào data/T28_rrf_bge_q3e/<biến>_pool.jsonl, nộp bằng
    python -m scripts.fuse_submission --doc-pool data/T28_rrf_bge_q3e/<biến>_pool.jsonl --name <biến> \
        --out-dir data/T28_rrf_bge_q3e --methods orig --m-doc 300 ...
"""
import argparse
import json
import os
import statistics
from collections import Counter

D = "data/tmp/phase1"
OUT_DIR = "data/T28_rrf_bge_q3e"


def jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bge-pool", default=f"{D}/doc_top1000_bge_pool.jsonl")
    ap.add_argument("--q3e-cands", default=f"{D}/cands_top1000_q3e4b_p1_own1000_qcorr.jsonl")
    ap.add_argument("--w-q3e", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--top-k", type=int, default=300)
    ap.add_argument("--lang-w", nargs="*", default=[], help="trọng số ngôn ngữ, vd. en=0.8 zh=0.8")
    ap.add_argument("--max-nonvi", type=float, default=None, help="tỉ lệ tối đa doc en/zh trong top-k")
    ap.add_argument("--rerank", action="store_true",
                    help="rerank_score = điểm bge-reranker CÓ SẴN trong --bge-pool (câu gốc) thay cho điểm gộp; "
                         "fuse_submission orig khi đó xếp top-k theo reranker như SOTA")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    w = args.w_q3e
    lang_w = {k: float(v) for k, v in (x.split("=") for x in args.lang_w)}
    # tên ngắn chỉ ghi biến: w (trọng số q3e), k nếu khác 60, trọng số ngôn ngữ, trần en/zh, top-k nếu khác 300
    tag = (f"w{w:g}" + (f"_k{args.k}" if args.k != 60 else "")
           + "".join(f"_{k}{v:g}" for k, v in sorted(lang_w.items()))
           + (f"_nv{args.max_nonvi:g}" if args.max_nonvi is not None else "")
           + (f"_top{args.top_k}" if args.top_k != 300 else "") + ("_rr" if args.rerank else ""))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = args.out or f"{OUT_DIR}/{tag}_pool.jsonl"

    bge = {int(r["id"]): r for r in jsonl(args.bge_pool)}
    q3e = {int(r["id"]): r["doc_ids"] for r in jsonl(args.q3e_cands)}
    cap = None if args.max_nonvi is None else int(args.max_nonvi * args.top_k)
    n_nonvi, ov_bge, ov_q3e, missing = [], [], [], 0
    with open(out, "w", encoding="utf-8") as fh:
        for qid in sorted(bge):
            cands = {c["doc_id"]: c for c in bge[qid]["candidates"]}
            r_b = {d: 1.0 / c["rrf_score"] - 60 for d, c in cands.items()}
            r_q = {d: i for i, d in enumerate(q3e[qid], 1)}
            missing += sum(1 for d in r_q if d not in cands)
            worst = len(cands) + 1                         # doc một bên không xếp: hạng sau cùng
            score = {d: w / (args.k + r_q.get(d, worst)) + (1 - w) / (args.k + r_b.get(d, worst))
                     for d in set(r_b) | set(r_q) if d in cands}
            for d in score:
                score[d] *= lang_w.get(cands[d]["language"], 1.0)
            ranked = sorted(score, key=lambda d: -score[d])
            top, nv = [], 0
            for d in ranked:
                is_nonvi = cands[d]["language"] in ("en", "zh")
                if cap is not None and is_nonvi and nv >= cap:
                    continue
                top.append(d)
                nv += is_nonvi
                if len(top) == args.top_k:
                    break
            n_nonvi.append(nv)
            ov_bge.append(len(set(top) & set(sorted(r_b, key=r_b.get)[: args.top_k])))
            ov_q3e.append(len(set(top) & set(q3e[qid][: args.top_k])))
            cs = [{"doc_id": d, "language": cands[d]["language"], "title": cands[d].get("title"),
                   "rrf_score": round(1.0 / (60 + i), 5), "found_by": [f"rrf_{tag}"], "rerank_score": cands[d]["rerank_score"] if args.rerank else round(score[d], 7)}
                  for i, d in enumerate(top, 1)]
            fh.write(json.dumps({"id": qid, "query": bge[qid]["query"], "rr_model": f"retrieval:rrf_bge_q3e4b_{tag}", "rr_len": 0,
                                 "candidates": cs}, ensure_ascii=False) + "\n")
    lang = Counter(c["language"] for r in jsonl(out) for c in r["candidates"])
    n = sum(lang.values())
    print(f"[fuse-retr] {tag}: {len(bge)} câu, top {args.top_k} | trùng top{args.top_k} bge {statistics.mean(ov_bge):.1f}, "
          f"q3e {statistics.mean(ov_q3e):.1f} | en/zh TB {statistics.mean(n_nonvi):.1f}/câu "
          f"(vi {lang['vi'] / n:.0%} en {lang['en'] / n:.0%} zh {lang['zh'] / n:.0%})"
          + (f" | {missing} doc q3e ngoài pool bge" if missing else "") + f" -> {out}")


if __name__ == "__main__":
    main()
