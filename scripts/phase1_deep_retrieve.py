"""Phase 1.1 - truy xuất sâu. Đổi ĐÚNG MỘT THỨ so với N3: chiều sâu.

Giả thuyết cần trả lời: pool 44 doc hiện tại chứa đúng 7 gold doc (đã đo bằng
bài E1). Nhìn sâu hơn thì có thêm gold lọt vào tầm mắt không?

Giữ nguyên tỷ lệ ba nhánh 35/20/10, chỉ nhân lên ~43 lần:
    vi 1500 / en 850 / zh 450 = 2800 chunk mỗi câu
Rồi gộp chunk về doc (điểm doc = điểm chunk cao nhất), lấy TOP_DOCS doc,
rerank chunk tốt nhất của mỗi doc.

MỘT LẦN CHẠY SINH RA CẢ HAI BÀI:
  - đối chứng  C65 : 35/20/10 chunk đầu của chính lượt tìm sâu = đúng cái mà
                     lượt tìm nông sẽ trả về
  - thí nghiệm T2800: toàn bộ 2800
Hai bài dùng chung một reranker và chung một lần encode, nên hiệu số giữa
chúng là hiệu ứng THUẦN của chiều sâu, không phụ thuộc việc so với điểm cũ.

Reranker mặc định là models/medical-reranker - đúng bản fine-tune mà D_ft và
N3 đã dùng. Checkpoint này từng mất trên máy, tải lại bằng scripts/fetch_reranker.py.

Ba chi tiết chống lãng phí:
  1. with_payload=["doc_id"] ở lượt sâu. Lấy đủ payload cho 2800 chunk là
     ~2,5 MB mỗi câu, nhân 1200 câu thành 3 GB truyền qua mạng vô ích.
  2. hnsw_ef nâng lên. limit=1500 mà ef mặc định thì đuôi danh sách là rác,
     lấy sâu mà không nâng ef là lấy sâu giả.
  3. max_length=512 khi rerank, giảm nửa chi phí trên GPU 6 GB.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient, models

QUERIES = "data/queries/real_queries.jsonl"
COLL = "medical_documents"
DEEP = {"vi": 1500, "en": 850, "zh": 450}
SHALLOW = {"vi": 35, "en": 20, "zh": 10}


def search_branch(client, vec, lang, limit, ef):
    return client.query_points(
        collection_name=COLL, query=vec, using="dense", limit=limit,
        query_filter=models.Filter(must=[models.FieldCondition(
            key="language", match=models.MatchValue(value=lang))]),
        search_params=models.SearchParams(hnsw_ef=ef),
        with_payload=["doc_id"],
    ).points


def best_chunk_per_doc(branch_points):
    """Gộp chunk về doc. Giữ điểm cao nhất và nhớ chunk nào đạt điểm đó.
    Trả về list (doc_id, score, point_id, lang, rank_trong_nhánh)."""
    best = {}
    for lang, pts in branch_points.items():
        for rank, p in enumerate(pts, 1):
            d = str((p.payload or {}).get("doc_id"))
            if d == "None":
                continue
            if d not in best or p.score > best[d][1]:
                best[d] = (d, p.score, p.id, lang, rank)
    return sorted(best.values(), key=lambda t: t[1], reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N câu đầu (0 = tất cả)")
    ap.add_argument("--top-docs", type=int, default=400, help="số doc đưa vào rerank")
    ap.add_argument("--hnsw-ef", type=int, default=512)
    ap.add_argument("--reranker-model", default="models/medical-reranker",
                    help="mặc định bản fine-tune mà N3 đã dùng, tải lại bằng scripts/fetch_reranker.py")
    ap.add_argument("--rerank-max-length", type=int, default=512)
    ap.add_argument("--rerank-batch", type=int, default=16)
    ap.add_argument("--retrieve-batch", type=int, default=64,
                    help="số id mỗi lời gọi retrieve; Qdrant timeout đọc nội bộ ~20s")
    ap.add_argument("--encode-batch", type=int, default=8)
    ap.add_argument("--out", default="data/tmp/phase1/deep.jsonl")
    ap.add_argument("--probe", action="store_true", help="chỉ đo thời gian, không rerank")
    args = ap.parse_args()

    queries = [json.loads(l) for l in open(QUERIES, encoding="utf-8")]
    if args.limit:
        queries = queries[: args.limit]
    print(f"[deep] {len(queries)} câu | sâu {DEEP} | top_docs={args.top_docs} | ef={args.hnsw_ef}")

    client = QdrantClient(url=args.qdrant_url, timeout=300)

    from src.rag.embedding import BgeM3Embedder
    t0 = time.time()
    embedder = BgeM3Embedder()
    print(f"[deep] nạp encoder {time.time()-t0:.0f}s")

    reranker = None
    if not args.probe:
        from src.rag.pipeline.components.reranking import CrossEncoderReranker
        t0 = time.time()
        reranker = CrossEncoderReranker(model_name=args.reranker_model,
                                        max_length=args.rerank_max_length)
        print(f"[deep] nạp reranker {time.time()-t0:.0f}s")

    texts = [q["query"] for q in queries]
    t0 = time.time()
    vecs = embedder.encode(texts, batch_size=args.encode_batch)
    print(f"[deep] encode {len(texts)} câu trong {time.time()-t0:.0f}s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t_search = t_fetch = t_rank = 0.0
    done = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for q, vec in zip(queries, vecs):
            v = vec.tolist() if hasattr(vec, "tolist") else list(vec)

            t = time.time()
            branch = {lang: search_branch(client, v, lang, k, args.hnsw_ef)
                      for lang, k in DEEP.items()}
            t_search += time.time() - t

            ranked = best_chunk_per_doc(branch)
            keep = ranked[: args.top_docs]

            if args.probe:
                done += 1
                continue

            # Lấy payload theo lô nhỏ: Qdrant có timeout đọc nội bộ ~20s, một
            # lời gọi retrieve 400 id đụng trần đó và trả 500.
            t = time.time()
            ids = [r[2] for r in keep]
            by_id = {}
            for i in range(0, len(ids), args.retrieve_batch):
                recs = client.retrieve(collection_name=COLL, ids=ids[i:i + args.retrieve_batch],
                                       with_payload=["content", "chunk_id", "language", "doc_id"],
                                       with_vectors=False)
                by_id.update({str(r.id): (r.payload or {}) for r in recs})
            t_fetch += time.time() - t

            pairs, meta = [], []
            for d, score, pid, lang, rank in keep:
                pl = by_id.get(str(pid), {})
                content = pl.get("content") or pl.get("text") or ""
                if not content:
                    continue
                pairs.append((q["query"], content))
                meta.append({"doc_id": d, "chunk_id": pl.get("chunk_id"), "language": pl.get("language", lang),
                             "dense_score": float(score), "branch_rank": rank})

            t = time.time()
            scores = reranker.score_pairs(pairs) if pairs else []
            t_rank += time.time() - t
            for m, s in zip(meta, scores):
                m["rerank_score"] = float(s)
            meta.sort(key=lambda m: m["rerank_score"], reverse=True)

            fh.write(json.dumps({
                "id": int(q["id"]), "query": q["query"],
                "n_docs_pool": len(ranked),
                "shallow_cut": SHALLOW,
                "candidates": meta,
            }, ensure_ascii=False) + "\n")
            fh.flush()

            done += 1
            if done % 25 == 0 or done == len(queries):
                el = t_search + t_fetch + t_rank
                print(f"[deep] {done}/{len(queries)} | tìm {t_search/done:.2f}s "
                      f"lấy {t_fetch/done:.2f}s rerank {t_rank/done:.2f}s mỗi câu "
                      f"| còn ~{(len(queries)-done)*el/done/60:.0f} phút", flush=True)

    print(f"\n[deep] xong. tìm {t_search:.0f}s | lấy payload {t_fetch:.0f}s | rerank {t_rank:.0f}s")
    print(f"[deep] ghi {args.out}")


if __name__ == "__main__":
    main()
