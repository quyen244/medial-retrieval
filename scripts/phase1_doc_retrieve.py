"""Bước 4: truy xuất CẤP DOCUMENT trên collection `medical_docs`, rerank, lắp bài nộp.

Khác hẳn mọi thứ đã chạy từ trước. Toàn bộ vòng 1, 2, 3 và cả Phase 1.1 đều
tìm trên index CHUNK rồi mới gộp chunk về doc. Cách đó chỉ thấy được tài liệu
có ít nhất một đoạn lọt top-K. Một tài liệu đúng chủ đề nhưng mỗi đoạn chỉ
khớp yếu thì không bao giờ nổi lên.

Với gold là cụm ~51 tài liệu cùng chủ đề mỗi câu, đó đúng là loại tài liệu
đang bị mất. Tìm ở cấp document chấm cả bài như một thể thống nhất.

Hai thí nghiệm tách biến, cùng một lần chạy:
  --top-docs 60    : nông. So với N3 (h_doc=6,93) để đo riêng hiệu ứng BIỂU DIỄN
  --top-docs 1000  : sâu. So với bài nông để đo riêng hiệu ứng CHIỀU SÂU
Hai lần chạy khác nhau đúng một tham số, nên quy được trách nhiệm.

Văn bản nộp ở tầng chunk vẫn là cửa sổ markdown gốc trần 8.000 ký tự - Phase 0
đã đo cấu hình này thắng (N3 = 0.1112 so với D_ft = 0.0805).
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient, models

from src.rag.pipeline.submission import (submission_chunk_text, submission_doc_id,
                                         zip_submission)
from scripts.phase0_assemble import RawDocs, load_index

QUERIES = "data/queries/real_queries.jsonl"
COLL = "medical_docs"
CAP = 8000


def search(client, dense, sparse, top_k, hybrid, ef):
    """Hybrid = dense + sparse, RRF phía server. Không lọc ngôn ngữ: quota cố
    định vi/en/zh áp một prior chưa bao giờ được đo (xem CLAUDE.md)."""
    params = models.SearchParams(hnsw_ef=ef)
    if hybrid and sparse is not None:
        return client.query_points(
            collection_name=COLL,
            prefetch=[
                models.Prefetch(query=dense, using="dense", limit=top_k, params=params),
                models.Prefetch(query=sparse, using="sparse", limit=top_k),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k, with_payload=["doc_id", "text", "language", "title"],
        ).points
    return client.query_points(
        collection_name=COLL, query=dense, using="dense", limit=top_k,
        search_params=params, with_payload=["doc_id", "text", "language", "title"],
    ).points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--name", required=True, help="tên bài nộp")
    ap.add_argument("--top-docs", type=int, default=400, help="số doc lấy về và rerank")
    ap.add_argument("--submit-docs", type=int, default=35, help="m_doc; đỉnh đo được ở Phase 0")
    ap.add_argument("--submit-chunks", type=int, default=43, help="m_chunk")
    ap.add_argument("--hybrid", action="store_true", help="bật nhánh sparse")
    ap.add_argument("--hnsw-ef", type=int, default=256)
    ap.add_argument("--reranker-model", default="models/medical-reranker")
    ap.add_argument("--rerank-max-length", type=int, default=512)
    ap.add_argument("--encode-batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="data/tmp/phase1")
    args = ap.parse_args()

    queries = [json.loads(l) for l in open(QUERIES, encoding="utf-8")]
    if args.limit:
        queries = queries[: args.limit]
    print(f"[docret] {args.name}: {len(queries)} câu | top_docs={args.top_docs} "
          f"| hybrid={args.hybrid} | nộp {args.submit_docs} doc / {args.submit_chunks} chunk")

    client = QdrantClient(url=args.qdrant_url, timeout=600)
    if not client.collection_exists(COLL):
        print(f"chưa có collection {COLL}. Chạy scripts/ingest_doc_index.py trước.")
        return 1

    from src.rag.embedding import BgeM3Embedder
    from src.rag.pipeline.components.reranking import CrossEncoderReranker
    embedder = BgeM3Embedder()
    reranker = CrossEncoderReranker(model_name=args.reranker_model,
                                    max_length=args.rerank_max_length)
    raw_docs = RawDocs(load_index())

    texts = [q["query"] for q in queries]
    t0 = time.time()
    if args.hybrid:
        dense_all, lex_all = embedder.encode_with_sparse(texts, batch_size=args.encode_batch)
    else:
        dense_all = embedder.encode(texts, batch_size=args.encode_batch)
        lex_all = [None] * len(texts)
    print(f"[docret] encode {len(texts)} câu trong {time.time()-t0:.0f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    dump_path = os.path.join(args.out_dir, f"{args.name}_pool.jsonl")
    entries, stats = [], {"pool": [], "chars": [], "fallback": 0}
    t_s = t_r = 0.0

    with open(dump_path, "w", encoding="utf-8") as dump:
        for i, (q, dv) in enumerate(zip(queries, dense_all), 1):
            dense = dv.tolist() if hasattr(dv, "tolist") else list(dv)
            sparse = (embedder.lexical_weights_to_sparse_vector(lex_all[i - 1])
                      if args.hybrid and lex_all[i - 1] is not None else None)

            t = time.time()
            pts = search(client, dense, sparse, args.top_docs, args.hybrid, args.hnsw_ef)
            t_s += time.time() - t

            pairs, meta = [], []
            for p in pts:
                pl = p.payload or {}
                txt = pl.get("text") or ""
                if not txt:
                    continue
                pairs.append((q["query"], txt))
                meta.append({"doc_id": str(pl.get("doc_id")), "language": pl.get("language"),
                             "title": pl.get("title"), "search_score": float(p.score)})

            t = time.time()
            scores = reranker.score_pairs(pairs) if pairs else []
            t_r += time.time() - t
            for m, s in zip(meta, scores):
                m["rerank_score"] = float(s)
            meta.sort(key=lambda m: m["rerank_score"], reverse=True)

            dump.write(json.dumps({"id": int(q["id"]), "query": q["query"],
                                   "candidates": meta}, ensure_ascii=False) + "\n")

            docs = [submission_doc_id(m["doc_id"]) for m in meta[: args.submit_docs]]
            chunks = []
            for m in meta[: args.submit_chunks]:
                raw = raw_docs.markdown(m["doc_id"])
                if not raw:
                    stats["fallback"] += 1
                    continue
                chunks.append({"doc_id": submission_doc_id(m["doc_id"]),
                               "chunk_text": submission_chunk_text(raw[:CAP])})
            stats["pool"].append(len(meta))
            stats["chars"].extend(len(c["chunk_text"]) for c in chunks)
            entries.append({"id": int(q["id"]), "relevant_docs": docs,
                            "relevant_chunks": chunks})

            if i % 50 == 0 or i == len(queries):
                el = t_s + t_r
                print(f"[docret] {i}/{len(queries)} | tìm {t_s/i:.2f}s rerank {t_r/i:.2f}s "
                      f"mỗi câu | còn ~{(len(queries)-i)*el/i/60:.0f} phút", flush=True)

    assert len({e["id"] for e in entries}) == len(queries)
    out = os.path.join(args.out_dir, f"{args.name}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False)
    zp = zip_submission(out)
    ch = sorted(stats["chars"])
    print(f"\n[docret] pool trung bình {statistics.mean(stats['pool']):.0f} doc | "
          f"chunk median {ch[len(ch)//2] if ch else 0} ký tự | "
          f"thiếu bản gốc {stats['fallback']}")
    print(f"[docret] {zp}  ({os.path.getsize(zp)/1e6:.1f} MB) | dump {dump_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
