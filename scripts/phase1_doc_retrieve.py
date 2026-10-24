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

Biến thể truy xuất (retrieval.doc.variants), mỗi biến thể một lượt tìm RIÊNG,
gộp bằng RRF cấp doc rồi giữ top_docs - KHÔNG ghép các bản vào một chuỗi (vòng
1: B, B' ghép chuỗi đều thua A):
  orig       câu gốc                                   (mặc định, = bài mốc)
  corrected  câu đã sửa chính tả (data/spelling/)
  en, zh     bản dịch nguyên câu (data/translate/)
  aspects    mỗi retrieval_target x vi/en/zh (augment_v3)
  hyde       đoạn y văn giả định sinh từ câu đã sửa (data/hyde/, scripts/gen_hyde.py)
Pool rerank vẫn là top_docs doc: thêm biến thể đổi DOC NÀO lọt vào pool, không
đổi pool lớn bao nhiêu - để mỗi bài nộp đo đúng một biến.

    --set 'retrieval.doc.variants=[orig,corrected]'
    --set 'retrieval.doc.variants=[orig,corrected,en,zh]'
    --set 'retrieval.doc.variants=[orig,corrected,en,zh,aspects]'
"""
import argparse
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient, models

from src.rag.config import add_config_args, config_from_args
from src.rag.pipeline.submission import (submission_chunk_text, submission_doc_id,
                                         zip_submission)
from scripts.md_strip import strip_markdown
from scripts.phase0_assemble import RawDocs, load_index

QUERIES = "data/queries/real_queries.jsonl"
PAYLOAD = ["doc_id", "text", "language", "title"]


def request(dense, sparse, top_k, hybrid, ef):
    """Một lượt tìm. Hybrid = dense + sparse, RRF phía server. Không lọc ngôn
    ngữ: quota cố định vi/en/zh áp một prior chưa bao giờ được đo (CLAUDE.md).
    Chỉ lấy doc_id: payload text lấy sau, cho đúng top_docs doc đã gộp."""
    params = models.SearchParams(hnsw_ef=ef)
    if hybrid and sparse is not None:
        return models.QueryRequest(
            prefetch=[models.Prefetch(query=dense, using="dense", limit=top_k, params=params),
                      models.Prefetch(query=sparse, using="sparse", limit=top_k)],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k, with_payload=["doc_id"])
    return models.QueryRequest(query=dense, using="dense", limit=top_k, params=params,
                               with_payload=["doc_id"])


def load_variant_sources(cfg, variants):
    """id -> nguồn văn bản cho từng loại biến thể cần dùng."""
    src = {}
    if "corrected" in variants:
        path = cfg.get("retrieval.doc.corrected_file", "data/spelling/corrected_queries.jsonl")
        src["corrected"] = {int(r["id"]): r["corrected_query"]
                            for r in map(json.loads, open(path, encoding="utf-8"))}
    if "en" in variants or "zh" in variants:
        path = cfg.get("retrieval.doc.translation_file", "data/translate/query_translation_v3.jsonl")
        src["tr"] = {int(r["id"]): r for r in map(json.loads, open(path, encoding="utf-8"))}
    if "aspects" in variants:
        path = cfg.get("retrieval.doc.augmented_file", "data/augment_v3_corrected/augmented.jsonl")
        src["aug"] = {int(r["qid"]): r for r in map(json.loads, open(path, encoding="utf-8"))}
    if "hyde" in variants:
        path = cfg.get("retrieval.doc.hyde_file", "data/hyde/hyde_v1.jsonl")
        src["hyde"] = {int(r["id"]): r["hyde_document"]
                       for r in map(json.loads, open(path, encoding="utf-8"))}
    return src


def variant_texts(q, variants, src):
    """[(nhãn, chuỗi)] của một câu, bỏ chuỗi trùng (giữ nhãn gặp trước). So
    theo CHỮ, bỏ qua dấu câu và hoa/thường: 93 câu đã sửa chỉ khác câu gốc ở
    dấu câu, tìm hai lần là ra cùng một list. Câu đã sửa thật sự khác câu gốc
    ở 131/1.200 câu."""
    qid, out, seen = int(q["id"]), [], set()

    def add(tag, text):
        text = (text or "").strip()
        key = re.sub(r"[\W_]+", "", text.lower())
        if key and key not in seen:
            seen.add(key)
            out.append((tag, text))

    for v in variants:
        if v == "orig":
            add("orig", q["query"])
        elif v == "corrected":
            add("corrected", src["corrected"].get(qid))
        elif v in ("en", "zh"):
            add(v, (src["tr"].get(qid) or {}).get(f"query_{v}"))
        elif v == "aspects":
            for u in (src["aug"].get(qid) or {}).get("units", []):
                for t in u.get("targets", []):
                    for lang in ("vi", "en", "zh"):
                        add(f"aspect_{lang}", t.get(lang))
        elif v == "hyde":
            add("hyde", src["hyde"].get(qid))
        else:
            raise SystemExit(f"biến thể không biết: {v}")
    return out


def rrf(lists, k, top_n):
    """lists: [(nhãn, [(point_id, doc_id), ...] theo hạng)].
    -> [(point_id, doc_id, điểm, [nhãn tìm ra])] top_n theo RRF.
    Một list duy nhất thì giữ nguyên thứ tự của nó (= bài mốc)."""
    score, found, pid_of = {}, {}, {}
    for tag, hits in lists:
        for rank, (pid, did) in enumerate(hits, 1):
            score[did] = score.get(did, 0.0) + 1.0 / (k + rank)
            found.setdefault(did, []).append(tag)
            pid_of.setdefault(did, pid)
    order = sorted(score, key=lambda d: -score[d])[:top_n]
    return [(pid_of[d], d, score[d], found[d]) for d in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="tên bài nộp")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N câu đầu, để đo tốc độ")
    add_config_args(ap)
    args = ap.parse_args()

    # Mọi tham số đến từ config.yaml; ghi đè bằng --set khoa.con=giá_trị.
    cfg = config_from_args(args)
    url = cfg.get("qdrant.url", "http://localhost:6333")
    coll = cfg.get("qdrant.collection_docs", "medical_docs")
    top_docs = cfg.get("retrieval.doc.top_docs", 60)
    hybrid = bool(cfg.get("retrieval.doc.hybrid", False))
    hnsw_ef = cfg.get("retrieval.doc.hnsw_ef", 256)
    encode_batch = cfg.get("retrieval.doc.encode_batch", 8)
    submit_docs = cfg.get("submission.m_doc", 35)
    submit_chunks = cfg.get("submission.m_chunk", 44)
    cap = cfg.get("submission.chunk_text.cap", 8000)
    do_strip = bool(cfg.get("submission.chunk_text.strip_markdown", True))
    rr_model = cfg.get("rerank.model", "models/medical-reranker")
    rr_len = cfg.get("rerank.max_length", 1024)
    out_dir = cfg.get("submission.out_dir", "data/tmp/phase1")
    variants = list(cfg.get("retrieval.doc.variants", ["orig"]))
    # null = bằng top_docs: bài một list (orig) phải lấy đủ top_docs doc.
    per_list = cfg.get("retrieval.doc.per_list_k") or top_docs
    rrf_k = cfg.get("retrieval.doc.rrf_k", 60)

    queries = [json.loads(l) for l in open(QUERIES, encoding="utf-8")]
    if args.limit:
        queries = queries[: args.limit]
    print(f"[docret] {args.name}: {len(queries)} câu | top_docs={top_docs} | hybrid={hybrid} "
          f"| biến thể={variants} (mỗi list {per_list}, rrf_k={rrf_k}) "
          f"| rerank_len={rr_len} | nộp {submit_docs} doc / {submit_chunks} chunk "
          f"| chunk_text trần {cap}, bóc markdown={do_strip}")
    src = load_variant_sources(cfg, variants)
    per_query = [variant_texts(q, variants, src) for q in queries]
    n_lists = [len(v) for v in per_query]
    print(f"[docret] lượt tìm mỗi câu: TB {statistics.mean(n_lists):.1f}, tối đa {max(n_lists)}")

    client = QdrantClient(url=url, timeout=cfg.get("qdrant.timeout", 600))
    if not client.collection_exists(coll):
        print(f"chưa có collection {coll}. Chạy scripts/ingest_doc_index.py trước.")
        return 1

    from src.rag.embedding import BgeM3Embedder
    from src.rag.pipeline.components.reranking import CrossEncoderReranker
    embedder = BgeM3Embedder()
    reranker = CrossEncoderReranker(model_name=rr_model, max_length=rr_len)
    raw_docs = RawDocs(load_index())
    prep = ((lambda x: submission_chunk_text(strip_markdown(x)))
            if do_strip else submission_chunk_text)

    # Encode mọi chuỗi của mọi câu trong một lượt, bỏ trùng giữa các câu.
    texts = list(dict.fromkeys(t for vs in per_query for _, t in vs))
    t0 = time.time()
    if hybrid:
        dense_all, lex_all = embedder.encode_with_sparse(texts, batch_size=encode_batch)
    else:
        dense_all = embedder.encode(texts, batch_size=encode_batch)
        lex_all = [None] * len(texts)
    vec = {t: (d.tolist() if hasattr(d, "tolist") else list(d),
               embedder.lexical_weights_to_sparse_vector(w) if hybrid and w is not None else None)
           for t, d, w in zip(texts, dense_all, lex_all)}
    print(f"[docret] encode {len(texts)} chuỗi trong {time.time()-t0:.0f}s")

    # Điểm rerank chỉ phụ thuộc (câu gốc, text doc): bài biến thể trùng phần lớn
    # pool với bài trước, chấm lại là phí. Dùng lại điểm trong các pool cũ cùng
    # reranker + max_length; pool cũ chưa ghi hai trường đó thì vẫn nhận (mọi pool
    # Phase 1 đều chạy BGE, 1024) nhưng in cảnh báo.
    cache, legacy = {}, 0
    for path in cfg.get("retrieval.doc.rerank_cache") or []:
        for row in map(json.loads, open(path, encoding="utf-8")):
            if "rr_model" not in row:
                legacy += 1
            elif (row["rr_model"], row["rr_len"]) != (rr_model, rr_len):
                continue
            for c in row["candidates"]:
                cache[(int(row["id"]), c["doc_id"])] = c["rerank_score"]
    if cache:
        print(f"[docret] dùng lại {len(cache):,} điểm rerank từ {cfg.get('retrieval.doc.rerank_cache')}"
              + (f" ({legacy} dòng pool cũ không ghi reranker, giả định cùng {rr_model})" if legacy else ""))

    os.makedirs(out_dir, exist_ok=True)
    dump_path = os.path.join(out_dir, f"{args.name}_pool.jsonl")
    entries, stats = [], {"pool": [], "chars": [], "fallback": 0, "union": [], "new_vs_orig": [],
                          "reused": 0, "scored": 0}
    t_s = t_r = 0.0

    with open(dump_path, "w", encoding="utf-8") as dump:
        for i, (q, vs) in enumerate(zip(queries, per_query), 1):
            # Mọi lượt tìm của một câu trong MỘT lời gọi HTTP.
            t = time.time()
            res = client.query_batch_points(collection_name=coll, requests=[
                request(*vec[text], per_list, hybrid, hnsw_ef) for _, text in vs])
            lists = [(tag, [(p.id, str((p.payload or {}).get("doc_id"))) for p in r.points])
                     for (tag, _), r in zip(vs, res)]
            chosen = rrf(lists, rrf_k, top_docs)
            recs = {str(r.id): r.payload or {} for r in client.retrieve(
                collection_name=coll, ids=[pid for pid, _, _, _ in chosen], with_payload=PAYLOAD)}
            t_s += time.time() - t
            union = {d for _, hits in lists for _, d in hits}
            orig_docs = {d for tag, hits in lists if tag == "orig" for _, d in hits[:top_docs]}
            stats["union"].append(len(union))
            stats["new_vs_orig"].append(sum(1 for _, d, _, _ in chosen if d not in orig_docs))

            pairs, meta = [], []
            for pid, did, fused, found in chosen:
                pl = recs.get(str(pid), {})
                txt = pl.get("text") or ""
                if not txt:
                    continue
                pairs.append((q["query"], txt))
                meta.append({"doc_id": did, "language": pl.get("language"),
                             "title": pl.get("title"), "rrf_score": round(fused, 5),
                             "found_by": found})

            qid = int(q["id"])
            todo = [j for j, m in enumerate(meta) if (qid, m["doc_id"]) not in cache]
            t = time.time()
            fresh = reranker.score_pairs([pairs[j] for j in todo]) if todo else []
            t_r += time.time() - t
            fresh = dict(zip(todo, fresh))
            for j, m in enumerate(meta):
                m["rerank_score"] = float(fresh[j] if j in fresh else cache[(qid, m["doc_id"])])
            stats["scored"] += len(todo)
            stats["reused"] += len(meta) - len(todo)
            meta.sort(key=lambda m: m["rerank_score"], reverse=True)

            dump.write(json.dumps({"id": qid, "query": q["query"], "rr_model": rr_model,
                                   "rr_len": rr_len, "candidates": meta}, ensure_ascii=False) + "\n")

            docs = [submission_doc_id(m["doc_id"]) for m in meta[: submit_docs]]
            chunks = []
            for m in meta[: submit_chunks]:
                raw = raw_docs.markdown(m["doc_id"])
                if not raw:
                    stats["fallback"] += 1
                    continue
                chunks.append({"doc_id": submission_doc_id(m["doc_id"]),
                               "chunk_text": prep(raw[:cap])})
            stats["pool"].append(len(meta))
            stats["chars"].extend(len(c["chunk_text"]) for c in chunks)
            entries.append({"id": int(q["id"]), "relevant_docs": docs,
                            "relevant_chunks": chunks})

            if i % 50 == 0 or i == len(queries):
                el = t_s + t_r
                print(f"[docret] {i}/{len(queries)} | tìm {t_s/i:.2f}s rerank {t_r/i:.2f}s "
                      f"mỗi câu | còn ~{(len(queries)-i)*el/i/60:.0f} phút", flush=True)

    assert len({e["id"] for e in entries}) == len(queries)
    out = os.path.join(out_dir, f"{args.name}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False)
    zp = zip_submission(out)
    ch = sorted(stats["chars"])
    print(f"\n[docret] pool trung bình {statistics.mean(stats['pool']):.0f} doc | "
          f"chunk median {ch[len(ch)//2] if ch else 0} ký tự | "
          f"thiếu bản gốc {stats['fallback']}")
    # Đo được mà không cần nhãn: các biến thể có kéo về doc MỚI không, hay chỉ
    # lặp lại list của câu gốc.
    print(f"[docret] hợp các list TB {statistics.mean(stats['union']):.0f} doc | "
          f"trong top {top_docs} sau RRF, TB {statistics.mean(stats['new_vs_orig']):.1f} doc "
          f"KHÔNG có trong list câu gốc")
    print(f"[docret] rerank: chấm mới {stats['scored']:,} cặp, dùng lại {stats['reused']:,} cặp")
    print(f"[docret] {zp}  ({os.path.getsize(zp)/1e6:.1f} MB) | dump {dump_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
