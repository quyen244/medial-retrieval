"""Chuẩn bị dev set gán tay (docs/dev-set-gan-tay.md): chọn 100 query thật, lấy
ứng viên, xuất file để người chấm 0-3, rồi nhập điểm ngược lại.

    # 1. tạo file để chấm (cần Qdrant; --with-pipeline cần thêm GPU cho reranker)
    python -m scripts.build_dev_set export --with-pipeline

    # 2. mở data/labeling/dev_set_template.tsv bằng Excel / Google Sheets,
    #    điền cột human_score (0-3) cho từng dòng, lưu lại dạng TSV

    # 3. nhập điểm vào dev_set.jsonl (chạy lại được, cộng dồn)
    python -m scripts.build_dev_set import --tsv data/labeling/dev_set_template.tsv

    python -m scripts.build_dev_set status     # đã chấm bao nhiêu

    # 4. chấm một bài nộp bằng nhãn người (chỉ trên các query dev có trong bài)
    python -m scripts.build_dev_set eval --submission tmp/sub/Bp.json

Thang điểm giống bước 5: 3 trả lời trực tiếp và đầy đủ, 2 có bằng chứng hữu ích
nhưng chưa đủ, 1 cùng chủ đề nhưng không trả lời được, 0 không liên quan.

Ứng viên mỗi query = hợp của hai nguồn, bỏ trùng:
  - hybrid_search top-20 không lọc ngôn ngữ (đúng thiết kế trong doc);
  - với --with-pipeline: 20 chunk mà pipeline chính (A, câu gốc) sẽ nộp.
Nguồn thứ hai bảo đảm nhãn phủ đúng những chunk hệ thống thật sự nộp, để tính
F2 và chọn cutoff. Mỗi dòng ghi lại nguồn nào đưa chunk lên.

Nhãn gắn với cặp (query_id, chunk_id), không gắn với thứ hạng: chạy export lại
sau khi corpus đổi chỉ thêm cặp mới, điểm đã chấm được giữ.
"""
import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

OUT_DIR = Path("data/labeling")
DEV_JSONL = OUT_DIR / "dev_set.jsonl"
TEMPLATE_TSV = OUT_DIR / "dev_set_template.tsv"
SEED = 42
# Dải độ dài theo doc: ≤15 từ / 16-30 từ / >30 từ, lấy 40 / 30 / 30.
BANDS = [("ngan", 0, 15, 40), ("vua", 16, 30, 30), ("dai", 31, 10**9, 30)]
_WORD = re.compile(r"[0-9a-zA-ZÀ-ỹ]+")


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def band_of(n: int) -> str:
    return next(name for name, lo, hi, _ in BANDS if lo <= n <= hi)


def sample_queries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(SEED)
    picked = []
    for name, lo, hi, k in BANDS:
        pool = [r for r in rows if lo <= word_count(r["query"]) <= hi]
        rng.shuffle(pool)
        picked += [{**r, "band": name} for r in pool[:k]]
    return sorted(picked, key=lambda r: int(r["id"]))


def load_dev() -> Dict[str, Dict[str, Any]]:
    """(query_id, chunk_id) -> dòng đã có."""
    out = {}
    if DEV_JSONL.exists():
        for line in DEV_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[f"{r['query_id']}|{r['chunk_id']}"] = r
    return out


def save_dev(rows: Dict[str, Dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda r: (int(r["query_id"]), r.get("rank", 999)))
    with DEV_JSONL.open("w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_export(args) -> int:
    import torch
    from src.ingestion.utils import get_dataset
    from src.rag.embedding import BgeM3Embedder
    from src.rag.vectorstore import VectorStore

    rows = list(get_dataset("query")["train"])
    queries = sample_queries(rows)
    print(f"Chọn {len(queries)} query: " +
          ", ".join(f"{b} {sum(1 for q in queries if q['band'] == b)}" for b, *_ in BANDS))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = BgeM3Embedder("BAAI/bge-m3", device=device, use_fp16=False, max_length=1024)
    vs = VectorStore(encoder, url=args.qdrant_url, collection_name=args.collection)

    pipeline = None
    if args.with_pipeline:
        from src.rag.pipeline.pipeline import PipelineConfig, RetrievalPipeline
        pipeline = RetrievalPipeline(vs, PipelineConfig())

    dev = load_dev()
    n_new = 0
    for i, q in enumerate(queries, 1):
        cands: Dict[str, Dict[str, Any]] = {}
        for rank, pt in enumerate(vs.hybrid_search(q["query"], top_k=args.top_k), 1):
            pl = pt.payload or {}
            cid = str(pl.get("chunk_id", pt.id))
            cands.setdefault(cid, {"payload": pl, "rank": rank, "sources": []})["sources"].append("hybrid")
        if pipeline is not None:
            from src.rag.pipeline.models import QueryItem
            res = pipeline.run(QueryItem(id=q["id"], text=q["query"]))
            for rank, c in enumerate(res.reranked, 1):
                e = cands.setdefault(str(c.chunk_id), {
                    "payload": {"doc_id": c.doc_id, "language": c.language, "content": c.content,
                                "title": c.title, "url": c.url},
                    "rank": 100 + rank, "sources": []})
                e["sources"].append("pipeline")
        for cid, e in cands.items():
            key = f"{q['id']}|{cid}"
            pl = e["payload"]
            if key in dev:
                dev[key]["sources"] = sorted(set(dev[key].get("sources", [])) | set(e["sources"]))
                continue
            dev[key] = {
                "query_id": int(q["id"]), "query": q["query"], "band": q["band"],
                "chunk_id": cid, "doc_id": pl.get("doc_id"), "language": pl.get("language"),
                "title": pl.get("title"), "url": pl.get("url"), "content": pl.get("content", ""),
                "rank": e["rank"], "sources": e["sources"], "human_score": None,
            }
            n_new += 1
        print(f"  [{i}/{len(queries)}] id {q['id']}: {len(cands)} ứng viên", flush=True)

    save_dev(dev)
    write_tsv(dev)
    print(f"\n{DEV_JSONL}: {len(dev)} cặp ({n_new} cặp mới, điểm đã chấm được giữ)")
    print(f"Chấm ở: {TEMPLATE_TSV}  (điền cột human_score 0-3, giữ nguyên các cột khác)")
    return 0


def write_tsv(dev: Dict[str, Dict[str, Any]]) -> None:
    rows = sorted(dev.values(), key=lambda r: (int(r["query_id"]), r.get("rank", 999)))
    with TEMPLATE_TSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["query_id", "chunk_id", "human_score", "band", "language", "sources", "query", "title", "content"])
        for r in rows:
            w.writerow([r["query_id"], r["chunk_id"], "" if r["human_score"] is None else r["human_score"],
                        r["band"], r.get("language") or "", ",".join(r.get("sources", [])),
                        r["query"], (r.get("title") or "").replace("\t", " "),
                        (r.get("content") or "").replace("\t", " ").replace("\n", " ")])


def cmd_import(args) -> int:
    dev = load_dev()
    if not dev:
        print(f"Chưa có {DEV_JSONL}. Chạy export trước.")
        return 1
    n_set, bad = 0, []
    with open(args.tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            v = (row.get("human_score") or "").strip()
            if not v:
                continue
            key = f"{row['query_id']}|{row['chunk_id']}"
            if key not in dev or v not in {"0", "1", "2", "3"}:
                bad.append((row.get("query_id"), row.get("chunk_id"), v))
                continue
            dev[key]["human_score"] = int(v)
            n_set += 1
    save_dev(dev)
    print(f"Đã nhập {n_set} điểm vào {DEV_JSONL}")
    if bad:
        print(f"Bỏ qua {len(bad)} dòng không hợp lệ (điểm phải là 0-3, cặp phải có trong dev set), ví dụ: {bad[:3]}")
    return cmd_status(args)


def cmd_status(args) -> int:
    dev = load_dev()
    done = [r for r in dev.values() if r["human_score"] is not None]
    qs = {r["query_id"] for r in dev.values()}
    full = {q for q in qs if all(r["human_score"] is not None for r in dev.values() if r["query_id"] == q)}
    dist = {s: sum(1 for r in done if r["human_score"] == s) for s in range(4)}
    print(f"{len(done)}/{len(dev)} cặp đã chấm | {len(full)}/{len(qs)} query chấm xong | phân bố điểm {dist}")
    return 0


def cmd_eval(args) -> int:
    """Precision / recall / F2 mức chunk của một bài nộp trên dev set.

    Chunk liên quan = human_score >= args.min_relevant. Bài nộp chỉ có chunk_text,
    nên khớp theo (doc_id, chunk_text). Chunk nộp mà chưa có nhãn bị tính là
    không liên quan (cận dưới) và được đếm riêng để biết dev set phủ đủ chưa.
    Chỉ tính query đã chấm xong toàn bộ."""
    dev = load_dev()
    by_q: Dict[int, List[Dict[str, Any]]] = {}
    for r in dev.values():
        by_q.setdefault(int(r["query_id"]), []).append(r)
    done_q = {q for q, rs in by_q.items() if all(r["human_score"] is not None for r in rs)}
    sub = {int(e["id"]): e for e in json.load(open(args.submission, encoding="utf-8"))}
    rows, unknown, submitted_total = [], 0, 0
    for q in sorted(done_q):
        e = sub.get(q)
        if not e or not e["relevant_chunks"]:
            continue
        label = {(str(r["doc_id"]), r["content"]): r["human_score"] for r in by_q[q]}
        rel = {k for k, v in label.items() if v >= args.min_relevant}
        got = [(str(c["doc_id"]), c["chunk_text"]) for c in e["relevant_chunks"]]
        unknown += sum(1 for g in got if g not in label)
        submitted_total += len(got)
        tp = sum(1 for g in got if g in rel)
        p_ = tp / len(got) if got else 0.0
        r_ = tp / len(rel) if rel else 0.0
        f2 = 5 * p_ * r_ / (4 * p_ + r_) if (p_ + r_) else 0.0
        rows.append((q, p_, r_, f2))
    if not rows:
        print("Không có query dev nào vừa chấm xong vừa có trong bài nộp.")
        return 1
    n = len(rows)
    mean = lambda i: sum(r[i] for r in rows) / n
    print(f"{n} query dev (đã chấm xong, có trong bài nộp) | chunk liên quan = điểm >= {args.min_relevant}")
    print(f"P {mean(1):.3f} | R {mean(2):.3f} | F2 {mean(3):.3f}")
    print(f"{unknown}/{submitted_total} chunk nộp chưa có nhãn (tính là không liên quan; "
          f"nhiều thì số này thấp hơn thật - dev set chưa phủ các chunk bài này nộp)")
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--collection", default="medical_documents")
    e.add_argument("--qdrant-url", default="http://localhost:6333")
    e.add_argument("--top-k", type=int, default=20)
    e.add_argument("--with-pipeline", action="store_true", help="thêm 20 chunk pipeline chính sẽ nộp")
    e.set_defaults(fn=cmd_export)
    i = sub.add_parser("import")
    i.add_argument("--tsv", default=str(TEMPLATE_TSV))
    i.set_defaults(fn=cmd_import)
    s = sub.add_parser("status")
    s.set_defaults(fn=cmd_status)
    ev = sub.add_parser("eval")
    ev.add_argument("--submission", required=True)
    ev.add_argument("--min-relevant", type=int, default=2, help="điểm tối thiểu để tính là liên quan")
    ev.set_defaults(fn=cmd_eval)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
