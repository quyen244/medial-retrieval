"""Gộp điểm rerank theo nhiều ngôn ngữ rồi lắp bài nộp, không cần GPU.

Đầu vào là pool của scripts/phase1_doc_retrieve.py (điểm rerank theo câu gốc
tiếng Việt) và cache của scripts/rescore_pools.py (điểm theo câu đã sửa chính
tả, en, zh). Mỗi doc d của câu q có:
  s_o  điểm với câu gốc
  s_m  điểm với câu CÙNG ngôn ngữ doc (vi <- câu đã sửa, en <- en, zh <- zh);
       câu đã sửa trùng chữ câu gốc thì s_m = s_o
Cách xếp hạng relevant_docs (--methods, nhiều cách một lần):
  orig        s_o                         (= bài đã nộp, làm đối chứng)
  matched     s_m                         (cách A)
  max         max(s_o, s_m)
  mix0.5      0.5*s_m + 0.5*s_o           (mixλ: λ là trọng số của s_m)
  rrf         RRF thứ hạng theo s_o và theo s_m
  maxall      max mọi điểm đã chấm của doc (cần --scope all, cách B)
relevant_chunks lấy từ --chunk-pool theo thứ tự rerank của chính pool đó
(pool top60 cho h_chunk cao nhất đã đo, mục 15-17 docs/phat-hien-tang-diem.md).
Nhiều pool (--chunk-pool top60 top300) thì nối tiếp, bỏ trùng: hết 60 doc của
top60 mới lấy tiếp theo thứ tự top300.

    python -m scripts.fuse_submission --dry                       # chỉ in chẩn đoán
    python -m scripts.fuse_submission --methods matched max mix0.5 rrf
"""
import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.assemble_doc_submission import write_submission
from scripts.rescore_pools import MATCH

D = "data/tmp/phase1"
POOLS = {"top60": f"{D}/doc_d35_c44_top60_bge_pool.jsonl", "top300": f"{D}/doc_top300_bge_pool.jsonl",
         "B": f"{D}/doc_top300_bge_vcorr_enzh_pool.jsonl", "C": f"{D}/doc_top300_bge_vcorr_enzh_asp_pool.jsonl",
         "Bhyde": f"{D}/doc_top300_bge_vcorr_enzh_hyde_pool.jsonl"}


def load_pool(path):
    return {int(r["id"]): r["candidates"] for r in map(json.loads, open(path, encoding="utf-8"))}


def rank_docs(cands, scores, qid, method):
    """cands: [{doc_id, language, rerank_score}] -> doc_id xếp theo method."""
    s_o = {c["doc_id"]: c["rerank_score"] for c in cands}

    def s_m(c):
        return scores[(qid, MATCH.get(c["language"] or "vi", "corrected"))].get(c["doc_id"], s_o[c["doc_id"]])

    if method == "orig":
        key = {d: s for d, s in s_o.items()}
    elif method == "matched":
        key = {c["doc_id"]: s_m(c) for c in cands}
    elif method == "max":
        key = {c["doc_id"]: max(s_o[c["doc_id"]], s_m(c)) for c in cands}
    elif method.startswith("mix"):
        lam = float(method[3:])
        key = {c["doc_id"]: lam * s_m(c) + (1 - lam) * s_o[c["doc_id"]] for c in cands}
    elif method == "rrf":
        by_o = sorted(s_o, key=lambda d: -s_o[d])
        m = {c["doc_id"]: s_m(c) for c in cands}
        by_m = sorted(m, key=lambda d: -m[d])
        key = defaultdict(float)
        for order in (by_o, by_m):
            for r, d in enumerate(order, 1):
                key[d] += 1.0 / (60 + r)
    elif method == "maxall":
        key = {d: max([s] + [scores[(qid, k)][d] for k in ("corrected", "en", "zh") if d in scores[(qid, k)]])
               for d, s in s_o.items()}
    else:
        raise SystemExit(f"cách gộp không biết: {method}")
    return sorted(key, key=lambda d: -key[d])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-pool", default="B", choices=sorted(POOLS), help="pool cung cấp ứng viên cho relevant_docs")
    ap.add_argument("--chunk-pool", nargs="+", default=["top60"], choices=sorted(POOLS))
    ap.add_argument("--rescore", default=f"{D}/rescore_bge.jsonl")
    ap.add_argument("--methods", nargs="+", default=["orig", "matched", "max", "mix0.5", "rrf"])
    ap.add_argument("--m-doc", type=int, default=80)
    ap.add_argument("--m-chunk", type=int, default=55)
    ap.add_argument("--cap", type=int, default=8000)
    ap.add_argument("--window", choices=["head", "middle"], default="head",
                    help="chunk_text lấy cap ký tự đầu bài hay ở giữa bài")
    ap.add_argument("--dry", action="store_true", help="chỉ in chẩn đoán, không ghi bài nộp")
    ap.add_argument("--out-dir", default=D)
    args = ap.parse_args()

    scores = defaultdict(dict)
    for r in map(json.loads, open(args.rescore, encoding="utf-8")):
        scores[(int(r["id"]), r["kind"])].update(r["scores"])
    doc_pool = load_pool(POOLS[args.doc_pool])
    lang = {(q, c["doc_id"]): c["language"] for q, cs in doc_pool.items() for c in cs}
    miss = sum(1 for (q, d), lg in lang.items() if lg in ("en", "zh") and d not in scores[(q, MATCH[lg])])
    n_enzh = sum(1 for lg in lang.values() if lg in ("en", "zh"))
    print(f"[fuse] doc pool {args.doc_pool} | chunk pool {'+'.join(args.chunk_pool)} | doc en/zh chưa có điểm "
          f"cùng ngôn ngữ: {miss:,}/{n_enzh:,}")

    base = {q: rank_docs(cs, scores, q, "orig")[: args.m_doc] for q, cs in doc_pool.items()}
    print(f"{'cách':10}{'trùng orig':>12}{'vi':>7}{'en':>7}{'zh':>7}   (top {args.m_doc} doc)")
    ranked = {}
    for method in args.methods:
        ranked[method] = {q: rank_docs(cs, scores, q, method) for q, cs in doc_pool.items()}
        top = {q: r[: args.m_doc] for q, r in ranked[method].items()}
        ov = statistics.mean(len(set(top[q]) & set(base[q])) for q in top)
        lg = Counter(lang[(q, d)] for q in top for d in top[q])
        n = sum(lg.values())
        print(f"{method:10}{ov:12.1f}{lg['vi'] / n:7.0%}{lg['en'] / n:7.0%}{lg['zh'] / n:7.0%}")
    if args.dry:
        return

    chunk_rank = defaultdict(list)
    for label in args.chunk_pool:
        for q, cs in load_pool(POOLS[label]).items():
            seen = set(chunk_rank[q])
            chunk_rank[q] += [c["doc_id"] for c in cs if c["doc_id"] not in seen]
    cap_tag = ("" if args.cap == 8000 else f"_cap{args.cap}") + ("_mid" if args.window == "middle" else "")
    for method in args.methods:
        name = (f"fuse_d{args.m_doc}_c{args.m_chunk}_{args.doc_pool}_{method}"
                f"_c{'+'.join(args.chunk_pool)}{cap_tag}")
        rows = [(q, ranked[method][q], chunk_rank[q]) for q in sorted(doc_pool)]
        write_submission(name, rows, args.m_doc, args.m_chunk, args.cap, True, args.out_dir, args.window)


if __name__ == "__main__":
    main()
