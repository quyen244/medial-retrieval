"""Chấm lại một pool doc bằng reranker khác, giữ nguyên tập ứng viên (T14, T05).

Pool vào là dump của scripts/phase1_doc_retrieve.py. Pool ra cùng định dạng, cùng
doc từng câu, chỉ thay `rerank_score` (điểm cũ giữ ở `rerank_score_prev`) và xếp
lại theo điểm mới. Biến duy nhất so với pool vào là reranker, nên bài dựng từ pool
ra (scripts/fuse_submission.py --doc-pool top300q4b) đo đúng hiệu ứng reranker.

Chấm bằng câu GỐC trên text doc trong payload `medical_docs`, như phase1_doc_retrieve.
--queries data/spelling/corrected_queries.jsonl: chấm bằng câu SỬA (quy định 2026-10-27), đặt tên --out có _qcorr.
--doc-text q3e: text doc dựng từ data/raw_documents như bản embed q3e4b (title | description | heading | thân bài,
ngân sách 8.000 ký tự, scripts/embed_q3e.py texts) thay cho payload 1.500 ký tự; ghi `rr_text` để --reuse không trộn hai loại.
Chạy dở thì chạy lại cùng lệnh: câu đã ghi ra --out được bỏ qua.
Cuối lượt in trung vị và p90 điểm theo ngôn ngữ doc (chẩn đoán T05).

    python -m scripts.rerank_pool --limit 5                      # đo tốc độ
    python -m scripts.rerank_pool
    python -m scripts.rerank_pool --set rerank.model=Qwen/Qwen3-Reranker-8B --out ...
    # T30: pool RRF w0.6 (T29) chấm lại bằng Qwen3-Reranker-4B, câu sửa
    python -m scripts.rerank_pool --pool data/T28_rrf_bge_q3e/w0.6_rr_pool.jsonl \
        --queries data/spelling/corrected_queries.jsonl --out data/T28_rrf_bge_q3e/w0.6_q4b_qcorr_pool.jsonl
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


def q3e_text_reader():
    """Text doc giống hệt bản embed q3e4b: doc_text() của build_doc_texts với tham số mặc định của embed_q3e texts."""
    from scripts.build_doc_texts import INDEX, RAW, doc_text
    idx = json.load(open(INDEX, encoding="utf-8"))

    def read(did):
        try:
            with open(os.path.join(RAW, idx[did], did + ".json"), encoding="utf-8") as f:
                return doc_text(json.load(f), 8000, 1000, 30)
        except (KeyError, OSError, json.JSONDecodeError):
            return ""
    return read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=f"{D}/doc_top300_bge_pool.jsonl")
    ap.add_argument("--out", default=f"{D}/doc_top300_qwen3r4b_pool.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="chỉ N câu đầu, để đo tốc độ")
    ap.add_argument("--reuse", nargs="*", default=[],
                    help="pool đã chấm bằng CÙNG model + câu hỏi: cặp (câu, doc) có sẵn điểm thì dùng lại, không chấm lại")
    ap.add_argument("--instruction", default=None,
                    help="instruction cho Qwen3-Reranker (dòng <Instruct>: ...); mặc định QWEN3_INSTRUCTION")
    ap.add_argument("--doc-text", choices=["payload", "q3e"], default="payload",
                    help="payload: text 1.500 ký tự của medical_docs; q3e: text 8.000 ký tự như bản embed q3e4b")
    ap.add_argument("--engine", choices=["hf", "vllm"], default="hf", help="vllm: chỉ cho Qwen3-Reranker")
    ap.add_argument("--queries", default=None, help="jsonl có corrected_query: chấm bằng câu sửa thay câu trong pool")
    add_config_args(ap)
    args = ap.parse_args(["--set", "rerank.model=Qwen/Qwen3-Reranker-4B"] + sys.argv[1:])

    cfg = config_from_args(args)
    coll = cfg.get("qdrant.collection_docs", "medical_docs")
    rr_model = cfg.get("rerank.model")
    rr_len = cfg.get("rerank.max_length", 1024)

    rows = [json.loads(l) for l in open(args.pool, encoding="utf-8")]
    if args.queries:
        corr = {int(r["id"]): r.get("corrected_query") or r["query"]
                for r in map(json.loads, open(args.queries, encoding="utf-8"))}
        rows = [{**r, "query": corr[int(r["id"])]} for r in rows]
    if args.limit:
        rows = rows[: args.limit]
    done = set()
    if os.path.exists(args.out):
        for r in map(json.loads, open(args.out, encoding="utf-8")):
            if (r["rr_model"], r["rr_len"], r.get("rr_text", "payload")) != (rr_model, rr_len, args.doc_text):
                raise SystemExit(f"{args.out} đã có điểm của {r['rr_model']} len={r['rr_len']}; ghi ra --out khác")
            done.add(int(r["id"]))
    todo = [r for r in rows if int(r["id"]) not in done]
    query_of = {int(r["id"]): r["query"] for r in rows}
    cache = {}
    for path in args.reuse:                    # điểm reranker chỉ phụ thuộc (câu hỏi, text doc): dùng lại được
        for r in map(json.loads, open(path, encoding="utf-8")):
            if (r["rr_model"], r["rr_len"], r.get("rr_text", "payload")) == (rr_model, rr_len, args.doc_text) \
                    and r["query"] == query_of.get(int(r["id"])):
                for c in r["candidates"]:
                    cache[(int(r["id"]), c["doc_id"])] = c["rerank_score"]
    n_pairs = sum(1 for r in todo for c in r["candidates"] if (int(r["id"]), c["doc_id"]) not in cache)
    if args.reuse:
        print(f"[rerank_pool] dùng lại {sum(len(r['candidates']) for r in todo) - n_pairs:,} cặp có sẵn từ {args.reuse}")
    print(f"[rerank_pool] {args.pool} -> {args.out} | {rr_model} len={rr_len} "
          f"| {len(todo)}/{len(rows)} câu, {n_pairs:,} cặp")

    if todo:
        from src.rag.pipeline.components.reranking import make_reranker
        reranker = make_reranker(model_name=rr_model, max_length=rr_len, engine=args.engine)
        if args.instruction:
            reranker.instruction = args.instruction
        q3e_text = q3e_text_reader() if args.doc_text == "q3e" else None
        client = QdrantClient(url=cfg.get("qdrant.url", "http://localhost:6333"),
                              timeout=cfg.get("qdrant.timeout", 600))
        t0, scored = time.time(), 0
        with open(args.out, "a", encoding="utf-8") as fh:
            for i, row in enumerate(todo, 1):
                cands = row["candidates"]
                need = [c for c in cands if (int(row["id"]), c["doc_id"]) not in cache]
                if args.doc_text == "q3e":
                    text_of = {c["doc_id"]: q3e_text(c["doc_id"]) for c in need}
                else:
                    text_of = {str((p.payload or {}).get("doc_id")): (p.payload or {}).get("text") or ""
                               for p in client.retrieve(collection_name=coll,
                                                        ids=[point_id(c["doc_id"]) for c in need],
                                                        with_payload=["doc_id", "text"])}
                missing = [c["doc_id"] for c in need if not text_of.get(c["doc_id"])]
                if missing:
                    raise SystemExit(f"câu {row['id']}: {len(missing)} doc không có text trong {coll}: {missing[:5]}")
                got = dict(zip([c["doc_id"] for c in need],
                               reranker.score_pairs([(row["query"], text_of[c["doc_id"]]) for c in need])))
                new = [{**c, "rerank_score_prev": c["rerank_score"],
                        "rerank_score": got.get(c["doc_id"], cache.get((int(row["id"]), c["doc_id"])))}
                       for c in cands]
                new.sort(key=lambda c: c["rerank_score"], reverse=True)
                fh.write(json.dumps({**row, "rr_model": rr_model, "rr_len": rr_len, "rr_text": args.doc_text, "candidates": new},
                                    ensure_ascii=False) + "\n")
                fh.flush()
                scored += len(need)
                if i % 20 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"[rerank_pool] {i}/{len(todo)} câu | {scored / el:.0f} cặp/s "
                          f"| còn ~{(n_pairs - scored) / max(scored / el, 1e-9) / 60:.0f} phút", flush=True)
    language_report(args.out)


if __name__ == "__main__":
    main()
