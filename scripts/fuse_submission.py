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
  langmm      s_o chuẩn hoá min-max về [-1, 1] RIÊNG trong từng nhóm ngôn ngữ của câu   (T03)
  langz       z-score của logit(s_o) RIÊNG trong từng nhóm ngôn ngữ của câu            (T04)
  chunkrrf    RRF của thứ hạng theo s_o và thứ hạng doc từ nhánh chunk (--chunk-evidence)  (T06)
Thí nghiệm T03, T04, T06: docs/theo-doi-thi-nghiem.md.
relevant_chunks lấy từ --chunk-pool theo thứ tự rerank của chính pool đó
(pool top60 cho h_chunk cao nhất đã đo, mục 15-17 docs/phat-hien-tang-diem.md).
Nhiều pool (--chunk-pool top60 top300) thì nối tiếp, bỏ trùng: hết 60 doc của
top60 mới lấy tiếp theo thứ tự top300.

    python -m scripts.fuse_submission --dry                       # chỉ in chẩn đoán
    python -m scripts.fuse_submission --methods matched max mix0.5 rrf
"""
import argparse
import json
import math
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
# Dump top-50 chunk mỗi câu từ index chunk `medical_documents` (Phase 0).
CHUNK_EVIDENCE = "data/tmp/sub1200/D_ft_top50.jsonl"
RRF_K = 60


def load_pool(path):
    return {int(r["id"]): r["candidates"] for r in map(json.loads, open(path, encoding="utf-8"))}


def load_chunk_evidence(path):
    """Dump chunk -> ({qid: [doc_id xếp theo bằng chứng chunk]}, {(qid, doc_id): language}).

    `retrieval_rank` đánh RIÊNG trong từng branch (vi/en/zh), nên không so thẳng được giữa
    các branch. Mỗi doc lấy hạng chunk tốt nhất của nó trong từng branch (MaxP), rồi gộp các
    branch bằng RRF. Không dùng `rerank_score`: file do models/medical-reranker chấm, model
    đã bị loại."""
    order, lang = {}, {}
    for r in map(json.loads, open(path, encoding="utf-8")):
        q = int(r["id"])
        best = {}
        for c in r["candidates"]:
            d, k = str(c["doc_id"]), (str(c["doc_id"]), c["branch"])
            best[k] = min(best.get(k, math.inf), int(c["retrieval_rank"]))
            lang.setdefault((q, d), c["language"])
        s = defaultdict(float)
        for (d, _), rank in best.items():
            s[d] += 1.0 / (RRF_K + rank)
        order[q] = sorted(s, key=lambda d: (-s[d], d))
    return order, lang


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def per_language(cands, method):
    """Chuẩn hoá điểm câu gốc RIÊNG trong từng nhóm ngôn ngữ của một câu, rồi mới xếp chung.

    langmm: min-max về [-1, 1]. langz: z-score trên logit - điểm rerank là xác suất sau
    sigmoid, logit trả về thang trước sigmoid nơi các điểm thấp còn phân biệt được.
    Nhóm không chuẩn hoá được (một doc, hoặc mọi doc bằng điểm) nhận 0, tức giữa thang."""
    groups = defaultdict(list)
    for c in cands:
        groups[c["language"] or "vi"].append(c)
    key = {}
    for cs in groups.values():
        if method == "langmm":
            v = [c["rerank_score"] for c in cs]
            lo, hi = min(v), max(v)
            norm = [-1 + 2 * (x - lo) / (hi - lo) if hi > lo else 0.0 for x in v]
        else:
            v = [logit(c["rerank_score"]) for c in cs]
            mu, sd = statistics.mean(v), statistics.pstdev(v)
            norm = [(x - mu) / sd if sd > 0 else 0.0 for x in v]
        key.update((c["doc_id"], x) for c, x in zip(cs, norm))
    return key


def rrf(orders):
    key = defaultdict(float)
    for order in orders:
        for r, d in enumerate(order, 1):
            key[d] += 1.0 / (RRF_K + r)
    return key


def rank_docs(cands, scores, qid, method, chunk_order=None):
    """cands: [{doc_id, language, rerank_score}] -> doc_id xếp theo method.
    chunk_order: doc_id xếp theo nhánh chunk, chỉ `chunkrrf` dùng; có thể chứa doc
    không có trong cands."""
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
        key = rrf([by_o, sorted(m, key=lambda d: -m[d])])
    elif method in ("langmm", "langz"):
        key = per_language(cands, method)
    elif method == "chunkrrf":
        if chunk_order is None:
            raise SystemExit("chunkrrf cần bằng chứng chunk (--chunk-evidence)")
        key = rrf([sorted(s_o, key=lambda d: -s_o[d]), chunk_order])
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
    ap.add_argument("--chunk-evidence", default=CHUNK_EVIDENCE, help="dump chunk cho method chunkrrf")
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
    chunk_order = {}
    if "chunkrrf" in args.methods:
        chunk_order, chunk_lang = load_chunk_evidence(args.chunk_evidence)
        outside = statistics.mean(sum(1 for d in chunk_order.get(q, []) if (q, d) not in lang) for q in doc_pool)
        for k, v in chunk_lang.items():   # doc chỉ nhánh chunk thấy, cho cột ngôn ngữ của chẩn đoán
            lang.setdefault(k, v)
        print(f"[fuse] bằng chứng chunk {args.chunk_evidence}: {len(chunk_order)} câu, "
              f"{statistics.mean(map(len, chunk_order.values())):.1f} doc/câu, {outside:.1f} doc ngoài doc pool")

    base = {q: rank_docs(cs, scores, q, "orig")[: args.m_doc] for q, cs in doc_pool.items()}
    print(f"{'cách':10}{'trùng orig':>12}{'vi':>7}{'en':>7}{'zh':>7}   (top {args.m_doc} doc)")
    ranked = {}
    for method in args.methods:
        ranked[method] = {q: rank_docs(cs, scores, q, method, chunk_order.get(q, [])) for q, cs in doc_pool.items()}
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
