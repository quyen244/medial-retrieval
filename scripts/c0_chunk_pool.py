"""C0 - chunk pool từ index chunk 3,9M (`medical_documents`), đầu vào cho T18a/T18b (vị trí cửa sổ) và T17.

Bốn bước, mỗi bước ghi file vào data/C0_chunk_pool/ và bước sau đọc lại (smoke từng bước được):

    python -m scripts.c0_chunk_pool encode                       # bge-m3 dense + sparse của 1.200 câu SỬA
    python -m scripts.c0_chunk_pool search --top 1000            # hybrid dense + sparse (RRF trong Qdrant)
    python -m scripts.c0_chunk_pool rerank --limit-queries 20    # smoke; bỏ --limit-queries để chạy thật
    python -m scripts.c0_chunk_pool windows --n-docs 200         # file cửa sổ cho fuse_submission + chẩn đoán

`windows` ghi cùng định dạng với scripts/chunk_windows.py, nên nộp bằng
    python -m scripts.fuse_submission ... --window best --windows-file data/C0_chunk_pool/c0_n200_windows.jsonl
Doc trong chunk list không có chunk nào trong C0 giữ cửa sổ đầu bài (score = None, xếp sau ở T18b).
Câu hỏi: câu SỬA (data/spelling/corrected_queries.jsonl) cho cả tìm lẫn rerank, theo quyết định 2026-10-27.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

D = "data/C0_chunk_pool"
COLL = "medical_documents"
QUERIES = "data/spelling/corrected_queries.jsonl"


def jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def queries():
    return {int(r["id"]): r.get("corrected_query") or r["query"] for r in jsonl(QUERIES)}


class Progress:
    def __init__(self, total, unit):
        self.total, self.unit, self.t0, self.last = total, unit, time.time(), time.time()

    def tick(self, done, force=False):
        now = time.time()
        if force or now - self.last >= 60:
            self.last = now
            rate = done / max(now - self.t0, 1e-9)
            print(f"[c0] {done:,}/{self.total:,} {self.unit}, {rate:,.1f} {self.unit}/s, "
                  f"còn ~{(self.total - done) / max(rate, 1e-9) / 60:.0f} phút", flush=True)


def cmd_encode(args):
    import numpy as np
    from FlagEmbedding import BGEM3FlagModel
    q = queries()
    ids = sorted(q)
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices=[args.device])
    out = model.encode([q[i] for i in ids], batch_size=32, max_length=512,
                       return_dense=True, return_sparse=True, return_colbert_vecs=False)
    os.makedirs(D, exist_ok=True)
    np.save(f"{D}/q_dense.npy", np.asarray(out["dense_vecs"], dtype=np.float32))
    with open(f"{D}/q_sparse.jsonl", "w", encoding="utf-8") as fh:
        for i, lw in zip(ids, out["lexical_weights"]):
            fh.write(json.dumps({"id": i, "w": {k: float(v) for k, v in lw.items()}}) + "\n")
    json.dump(ids, open(f"{D}/q_ids.json", "w"))
    print(f"[c0] encode {len(ids)} câu -> {D}/q_dense.npy, q_sparse.jsonl")


def cmd_search(args):
    import numpy as np
    from qdrant_client import QdrantClient, models
    ids = json.load(open(f"{D}/q_ids.json"))
    dense = np.load(f"{D}/q_dense.npy")
    sparse = {r["id"]: r["w"] for r in jsonl(f"{D}/q_sparse.jsonl")}
    c = QdrantClient(url=args.url, timeout=600)
    todo = ids[: args.limit_queries or None]
    out = f"{D}/search_top{args.top}" + (f"_q{len(todo)}" if args.limit_queries else "") + ".jsonl"
    prog = Progress(len(todo), "câu")
    with open(out, "w", encoding="utf-8") as fh:
        for n, (qid, v) in enumerate(zip(todo, dense), 1):
            w = sparse[qid]
            sv = models.SparseVector(indices=[int(k) for k in w], values=list(w.values()))
            pts = c.query_points(
                COLL,
                prefetch=[models.Prefetch(query=v.tolist(), using="dense", limit=args.top,
                                          params=models.SearchParams(hnsw_ef=max(args.top, 128))),
                          models.Prefetch(query=sv, using="sparse", limit=args.top)],
                query=models.FusionQuery(fusion=models.Fusion.RRF), limit=args.top,
                with_payload=["doc_id", "chunk_id", "chunk_index", "language"]).points
            fh.write(json.dumps({"id": qid, "chunks": [
                {"pid": str(p.id), "doc_id": str(p.payload["doc_id"]), "chunk_id": p.payload["chunk_id"],
                 "chunk_index": p.payload["chunk_index"], "language": p.payload.get("language"),
                 "rrf": round(p.score, 6)} for p in pts]}, ensure_ascii=False) + "\n")
            prog.tick(n)
    prog.tick(len(todo), force=True)
    print(f"[c0] search -> {out}")


def cmd_rerank(args):
    from qdrant_client import QdrantClient
    from sentence_transformers import CrossEncoder
    q = queries()
    rows = list(jsonl(args.search))[: args.limit_queries or None]
    keep = None
    if args.chunk_list_docs:                         # chỉ chấm chunk của N doc đầu chunk list (đủ cho T18a/T18b)
        from scripts.chunk_windows import chunk_lists
        keep = {q: set(ds) for q, ds in chunk_lists(args.chunk_pool, args.chunk_list_docs).items()}
    for r in rows:
        r["chunks"] = [ch for ch in r["chunks"][: args.rerank_top] if keep is None or ch["doc_id"] in keep[r["id"]]]
    c = QdrantClient(url=args.url, timeout=600)
    pids = list({ch["pid"] for r in rows for ch in r["chunks"]})
    text, t = {}, time.time()
    for i in range(0, len(pids), 1000):              # lấy content theo lô, chỉ cho chunk cần chấm
        for p in c.retrieve(COLL, ids=pids[i: i + 1000], with_payload=["content"]):
            text[str(p.id)] = p.payload.get("content") or ""
    print(f"[c0] lấy {len(text):,} content trong {time.time() - t:.0f}s", flush=True)
    model = CrossEncoder(args.model, max_length=args.max_length, model_kwargs={"torch_dtype": "float16"})
    pairs = [(q[r["id"]], text[ch["pid"]]) for r in rows for ch in r["chunks"]]
    flat, prog, step = [], Progress(len(pairs), "cặp"), args.batch * 64
    for i in range(0, len(pairs), step):
        flat += model.predict(pairs[i: i + step], batch_size=args.batch, show_progress_bar=False).tolist()
        prog.tick(len(flat))
    prog.tick(len(flat), force=True)
    out = args.out or args.search.replace("search_", "rerank_").replace(
        ".jsonl", f"_cl{args.chunk_list_docs}.jsonl" if args.chunk_list_docs else ".jsonl")
    k = 0
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            for ch in r["chunks"]:
                ch["score"] = round(float(flat[k]), 5)
                ch["len"] = len(text[ch["pid"]])
                ch["head"] = text[ch["pid"]][:120]     # để định vị chunk trong markdown gốc
                k += 1
            r["chunks"].sort(key=lambda ch: -ch["score"])
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[c0] rerank {len(rows)} câu, {len(pairs):,} cặp -> {out}")


def locate(raw, head):
    """Vị trí đầu chunk trong markdown gốc: khớp nguyên 60 ký tự đầu; không được thì khớp 8 từ đầu,
    bỏ qua mọi ký hiệu giữa các từ (markdown gốc còn **, ##, link mà content đã bóc)."""
    h = head.strip()[:60]
    if not h:
        return -1
    i = raw.find(h)
    if i >= 0:
        return i
    words = re.findall(r"\w+", head)[:8]
    m = re.search(r"[\W_]+".join(map(re.escape, words)), raw) if words else None
    return m.start() if m else -1


def cmd_windows(args):
    from scripts.chunk_windows import chunk_lists
    from scripts.phase0_assemble import RawDocs, load_index
    rr = {r["id"]: r["chunks"] for r in jsonl(args.rerank)}
    lists = chunk_lists(args.chunk_pool, args.n_docs)
    missing = set(lists) - set(rr)
    if missing:
        print(f"[c0] CẢNH BÁO: {len(missing)} câu chưa có rerank (file smoke?), chỉ ghi các câu đã có")
    raw_docs = RawDocs(load_index())
    cap = args.cap
    n_doc = n_ev = n_deep = n_loc = 0
    out = args.out or f"{D}/c0_n{args.n_docs}_windows.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for qid in sorted(set(lists) & set(rr)):
            best = {}
            for ch in rr[qid]:                       # đã xếp theo điểm giảm dần: lần đầu gặp doc = chunk tốt nhất
                best.setdefault(ch["doc_id"], ch)
            win = {}
            for d in lists[qid]:
                n_doc += 1
                ch = best.get(d)
                if ch is None:
                    win[d] = {"start": 0, "score": None, "head": None, "n": 0}
                    continue
                n_ev += 1
                raw = raw_docs.markdown(d)
                pos = locate(raw, ch["head"]) if raw else -1
                start = 0
                if pos >= 0:
                    n_loc += 1
                    if pos + ch["len"] > cap:            # chunk tốt nhất không nằm trọn trong cửa sổ đầu bài
                        n_deep += 1
                        start = max(0, min(pos - (cap - ch["len"]) // 2, len(raw) - cap))
                        nl = raw.rfind("\n", start - 300, start + 1)
                        start = nl + 1 if nl >= 0 and nl < pos else start
                win[d] = {"start": start, "score": ch["score"], "head": None, "n": 1, "chunk_pos": pos}
            fh.write(json.dumps({"id": qid, "scorer": "c0", "cap": cap, "windows": win}, ensure_ascii=False) + "\n")
    print(f"[c0] chunk list {n_doc:,} doc | có chunk trong C0: {n_ev:,} ({n_ev / max(n_doc, 1):.0%}) | "
          f"định vị được trong bản gốc: {n_loc:,} ({n_loc / max(n_ev, 1):.0%}) | "
          f"chunk tốt nhất nằm ngoài {cap} ký tự đầu: {n_deep:,} ({n_deep / max(n_doc, 1):.0%} chunk list) -> {out}")

    # Chẩn đoán T17: doc chỉ C0 thấy (ngoài chunk list) mà chunk tốt nhất vượt điểm chunk tốt nhất của doc hạng m_doc
    extra = []
    for qid in set(lists) & set(rr):
        inlist = set(lists[qid])
        bests = {}
        for ch in rr[qid]:
            bests.setdefault(ch["doc_id"], ch["score"])
        ref = sorted((bests[d] for d in lists[qid] if d in bests), reverse=True)
        bar = ref[args.m_doc - 1] if len(ref) >= args.m_doc else (ref[-1] if ref else 1.0)
        extra.append(sum(1 for d, s in bests.items() if d not in inlist and s > bar))
    if extra:
        print(f"[c0] T17: doc ngoài chunk list mà chunk tốt nhất vượt doc hạng {args.m_doc} (theo điểm chunk): "
              f"TB {statistics.mean(extra):.1f}/câu")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("encode")
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(fn=cmd_encode)
    p = sub.add_parser("search")
    p.add_argument("--top", type=int, default=1000)
    p.add_argument("--limit-queries", type=int, default=0)
    p.set_defaults(fn=cmd_search)
    p = sub.add_parser("rerank")
    p.add_argument("--search", default=f"{D}/search_top1000.jsonl")
    p.add_argument("--rerank-top", type=int, default=1000, help="chỉ chấm N chunk đầu theo RRF")
    p.add_argument("--chunk-list-docs", type=int, default=0,
                   help="chỉ chấm chunk thuộc N doc đầu chunk list (0 = mọi chunk, cần cho chẩn đoán T17)")
    p.add_argument("--chunk-pool", nargs="+", default=["top60", "top300"])
    p.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--limit-queries", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_rerank)
    p = sub.add_parser("windows")
    p.add_argument("--rerank", default=f"{D}/rerank_top1000_cl200.jsonl")
    p.add_argument("--chunk-pool", nargs="+", default=["top60", "top300"])
    p.add_argument("--n-docs", type=int, default=200)
    p.add_argument("--m-doc", type=int, default=160)
    p.add_argument("--cap", type=int, default=2700)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_windows)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
