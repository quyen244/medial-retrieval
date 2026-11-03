"""Chấm lại mọi doc trong các pool bằng câu hỏi ở NHIỀU ngôn ngữ, cache ra jsonl.

Lý do (docs/phat-hien-tang-diem.md mục 16): pool B có 63% doc en/zh nhưng
reranker chấm bằng câu tiếng Việt dồn chúng xuống, top 80 chỉ còn 24%. Script
này chỉ SINH ĐIỂM, không chọn cách gộp: mọi cách gộp (rerank theo đúng ngôn
ngữ doc, max / RRF qua ba ngôn ngữ, nới m_doc, lấy chunk từ pool khác) làm
offline sau bằng điểm đã cache, không cần GPU.

Mỗi câu: hợp mọi doc của các --pools. Điểm với câu gốc (orig) đã có trong pool;
script chấm thêm các loại câu:
  corrected  câu đã sửa chính tả (chỉ 131 câu khác câu gốc ở chữ; còn lại bỏ
             qua, offline dùng điểm orig)
  en, zh     bản dịch nguyên câu (data/translate/query_translation_v3.jsonl)

--scope matched : mỗi doc chỉ chấm với câu CÙNG ngôn ngữ (doc vi <- corrected,
                  doc en <- en, doc zh <- zh). Rẻ nhất, = cách A.
--scope all     : mọi doc x mọi loại câu, = cách B. Chạy sau matched thì dùng lại
                  các cặp đã chấm.
--only-top N    : (chỉ với all, cần matched đã xong) giới hạn cách B vào doc nằm
                  trong top N theo điểm câu gốc HOẶC theo điểm cùng ngôn ngữ của
                  cách A. Doc dưới hạng N ở cả hai gần như không lọt vào phần nộp.

Cache: một dòng mỗi (câu, loại câu) đã chấm xong, {"id", "kind", "scores":
{doc_id: điểm}}. Khoá KHÔNG gồm văn bản câu: đổi file dịch hay file sửa chính tả
thì phải ghi ra --out mới.

    python -m scripts.rescore_pools --scope matched --limit 20      # đo tốc độ
    python -m scripts.rescore_pools --scope matched
    python -m scripts.rescore_pools --scope all --only-top 200
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient

from scripts.ingest_doc_index import point_id
from src.rag.config import add_config_args, config_from_args

D = "data/tmp/phase1"
POOLS = [f"{D}/doc_d35_c44_top60_bge_pool.jsonl", f"{D}/doc_top300_bge_pool.jsonl",
         f"{D}/doc_top300_bge_vcorr_enzh_pool.jsonl", f"{D}/doc_top300_bge_vcorr_enzh_asp_pool.jsonl",
         f"{D}/doc_top300_bge_vcorr_enzh_hyde_pool.jsonl"]
KINDS = ("corrected", "en", "zh")
MATCH = {"vi": "corrected", "en": "en", "zh": "zh"}


def letters(text):
    return re.sub(r"[\W_]+", "", (text or "").lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", default=POOLS, help="pool thiếu file thì bỏ qua, có báo")
    ap.add_argument("--scope", choices=["matched", "all"], required=True)
    ap.add_argument("--out", default=f"{D}/rescore_bge.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="chỉ N câu đầu, để đo tốc độ")
    ap.add_argument("--only-top", type=int, default=0,
                    help="scope all: chỉ doc trong top N theo câu gốc hoặc theo cách A")
    add_config_args(ap)
    args = ap.parse_args()

    cfg = config_from_args(args)
    coll = cfg.get("qdrant.collection_docs", "medical_docs")
    rr_model = cfg.get("rerank.model")
    rr_len = cfg.get("rerank.max_length", 1024)

    # Hợp các pool: id -> {doc_id: ngôn ngữ}, và câu gốc của từng id.
    docs, orig, s_orig = defaultdict(dict), {}, defaultdict(dict)
    for path in args.pools:
        if not os.path.exists(path):
            print(f"[rescore] BỎ QUA, chưa có {path}")
            continue
        for row in map(json.loads, open(path, encoding="utf-8")):
            qid = int(row["id"])
            orig[qid] = row["query"]
            for c in row["candidates"]:
                docs[qid].setdefault(c["doc_id"], c.get("language") or "vi")
                s_orig[qid][c["doc_id"]] = c["rerank_score"]
        print(f"[rescore] đọc {path}")

    corrected = {int(r["id"]): r["corrected_query"] for r in map(json.loads, open(
        cfg.get("retrieval.doc.corrected_file", "data/spelling/corrected_queries.jsonl"), encoding="utf-8"))}
    tr = {int(r["id"]): r for r in map(json.loads, open(
        cfg.get("retrieval.doc.translation_file", "data/translate/query_translation_v3.jsonl"), encoding="utf-8"))}

    def query_text(qid, kind):
        if kind == "corrected":
            t = corrected.get(qid)
            return t if t and letters(t) != letters(orig[qid]) else None
        return (tr.get(qid) or {}).get(f"query_{kind}") or None

    done = defaultdict(dict)   # (id, kind) -> {doc_id: điểm}
    if os.path.exists(args.out):
        for r in map(json.loads, open(args.out, encoding="utf-8")):
            done[(int(r["id"]), r["kind"])].update(r["scores"])

    def shortlist(qid):
        """Doc trong top N theo câu gốc hoặc theo điểm cùng ngôn ngữ (cách A)."""
        def matched(d):
            kind = MATCH.get(docs[qid][d], "corrected")
            return done[(qid, kind)].get(d, s_orig[qid][d])
        keep = set()
        for key in (s_orig[qid].get, matched):
            keep |= set(sorted(docs[qid], key=key, reverse=True)[: args.only_top])
        return keep

    if args.only_top and args.scope != "all":
        raise SystemExit("--only-top chỉ dùng với --scope all")
    qids = sorted(docs)[: args.limit] if args.limit else sorted(docs)
    if args.only_top:
        en_zh = [d for q in qids for d, lg in docs[q].items() if lg in ("en", "zh")]
        miss = sum(1 for q in qids for d, lg in docs[q].items()
                   if lg in ("en", "zh") and d not in done[(q, MATCH[lg])])
        if miss > 0.01 * max(len(en_zh), 1):
            raise SystemExit(f"--only-top cần cách A xong trước: còn {miss:,} doc en/zh chưa chấm cùng ngôn ngữ")
    plan = {}
    for qid in qids:
        allowed = shortlist(qid) if args.only_top else None
        for kind in KINDS:
            text = query_text(qid, kind)
            if not text:
                continue
            want = [d for d, lg in docs[qid].items()
                    if (args.scope == "all" and (allowed is None or d in allowed))
                    or MATCH.get(lg, "corrected") == kind]
            todo = [d for d in want if d not in done[(qid, kind)]]
            if todo:
                plan[(qid, kind)] = (text, todo)
    n_pairs = sum(len(t) for _, t in plan.values())
    print(f"[rescore] {len(qids)} câu | TB {sum(len(docs[q]) for q in qids) / len(qids):.0f} doc mỗi câu "
          f"| scope={args.scope} | còn {n_pairs:,} cặp | {rr_model} len={rr_len} -> {args.out}")
    if not plan:
        return

    from src.rag.pipeline.components.reranking import make_reranker
    reranker = make_reranker(model_name=rr_model, max_length=rr_len)
    client = QdrantClient(url=cfg.get("qdrant.url", "http://localhost:6333"),
                          timeout=cfg.get("qdrant.timeout", 600))

    by_q = defaultdict(list)
    for (qid, kind), v in plan.items():
        by_q[qid].append((kind, *v))
    t0, scored = time.time(), 0
    with open(args.out, "a", encoding="utf-8") as fh:
        for i, qid in enumerate(sorted(by_q), 1):
            need = sorted({d for _, _, todo in by_q[qid] for d in todo})
            text_of = {str((p.payload or {}).get("doc_id")): (p.payload or {}).get("text") or ""
                       for p in client.retrieve(collection_name=coll, ids=[point_id(d) for d in need],
                                                with_payload=["doc_id", "text"])}
            # Mọi loại câu của một id chấm chung một lần gọi model.
            pairs, keys = [], []
            for kind, text, todo in by_q[qid]:
                for d in todo:
                    if text_of.get(d):
                        pairs.append((text, text_of[d]))
                        keys.append((kind, d))
            scores = reranker.score_pairs(pairs)
            out = defaultdict(dict)
            for (kind, d), s in zip(keys, scores):
                out[kind][d] = round(s, 6)
            for kind, sc in out.items():
                fh.write(json.dumps({"id": qid, "kind": kind, "scores": sc}) + "\n")
            fh.flush()
            scored += len(pairs)
            if i % 50 == 0 or i == len(by_q):
                el = time.time() - t0
                print(f"[rescore] {i}/{len(by_q)} câu | {scored / el:.0f} cặp/s "
                      f"| còn ~{(n_pairs - scored) / max(scored / el, 1e-9) / 60:.0f} phút", flush=True)
    print(f"[rescore] xong {scored:,} cặp trong {(time.time() - t0) / 60:.0f} phút")


if __name__ == "__main__":
    main()
