"""Đo thời gian từng bước của pha tạo submission (encode -> Qdrant -> rerank)
và thử xem fp16 / encode theo lô có đáng làm không.

    python -m scripts.profile_inference --n 20 --augmented data/augment_v2/augmented.jsonl

Chạy 3 chế độ trên N query đầu: A (câu gốc), B' (chuỗi ghép, mode concat), C
(từng dòng, mode lines). Với mỗi chế độ in thời gian cộng dồn theo bước. Sau đó:
  - chạy lại B' với encoder + reranker fp16, so thời gian và độ chồng lấn top-k
    với bản fp32;
  - đo riêng encode theo lô (1/16/32) và rerank với batch_size khác nhau.

Số đo chỉ đúng khi GPU không bận việc khác (ingest, vLLM đang chạy sẽ làm lệch).
Bước attribution chạy các nhánh ngôn ngữ TUẦN TỰ để cộng thời gian không bị chồng;
thời gian thực (wall) đo riêng với chế độ song song mặc định của pipeline.
"""
import argparse
import json
import statistics
import time
from collections import defaultdict
from typing import Dict, List

import torch
from rich.console import Console
from sentence_transformers import CrossEncoder

from src.rag.embedding import BgeM3Embedder
from src.rag.pipeline.models import QueryItem
from src.rag.pipeline.multiquery import MultiQueryConfig, MultiQueryPipeline, load_augmented
from src.rag.pipeline.pipeline import PipelineConfig, RetrievalPipeline
from src.rag.vectorstore import VectorStore

console = Console()


class Stage:
    """Cộng dồn thời gian và số lần gọi theo tên bước."""

    def __init__(self):
        self.t: Dict[str, float] = defaultdict(float)
        self.n: Dict[str, int] = defaultdict(int)
        self.pairs = 0

    def reset(self):
        self.t.clear(); self.n.clear(); self.pairs = 0


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def instrument(pipeline: RetrievalPipeline, stage: Stage):
    """Bọc encode / search / rerank.predict để cộng thời gian. Gọi lại sau khi
    thay model (fp16) để bọc lại."""
    vs = pipeline.vectorstore
    enc, search = vs.encoder.encode, vs.search
    predict = pipeline.reranker.model.predict

    def w_encode(*a, **k):
        sync(); t = time.perf_counter(); out = enc(*a, **k); sync()
        stage.t["encode"] += time.perf_counter() - t; stage.n["encode"] += 1
        return out

    def w_search(*a, **k):
        t = time.perf_counter(); out = search(*a, **k)
        stage.t["qdrant"] += time.perf_counter() - t; stage.n["qdrant"] += 1
        return out

    def w_predict(pairs, *a, **k):
        sync(); t = time.perf_counter(); out = predict(pairs, *a, **k); sync()
        stage.t["rerank"] += time.perf_counter() - t; stage.n["rerank"] += 1
        stage.pairs += len(pairs)
        return out

    vs.encoder.encode = w_encode
    vs.search = w_search
    pipeline.reranker.model.predict = w_predict

    def restore():
        vs.encoder.encode, vs.search, pipeline.reranker.model.predict = enc, search, predict
    return restore


def run_mode(name: str, runner, queries: List[QueryItem], stage: Stage, parallel: bool):
    stage.reset()
    tops: Dict[int, List[str]] = {}
    sync(); t0 = time.perf_counter()
    for q in queries:
        out = runner.run(q, parallel)
        res = getattr(out, "result", out)
        tops[q.id] = [str(c.chunk_id) for c in res.reranked]
    sync(); wall = time.perf_counter() - t0
    return wall, tops


def report(title: str, wall: float, stage: Stage, n_q: int):
    other = wall - sum(stage.t.values())
    console.print(f"[bold]{title}[/bold]  wall {wall:.1f}s = {wall / n_q * 1000:.0f} ms/query")
    for k in ("encode", "qdrant", "rerank"):
        pct = 100 * stage.t[k] / wall if wall else 0
        console.print(f"   {k:7s} {stage.t[k]:7.2f}s  {pct:4.0f}%   {stage.n[k]:4d} lần gọi  "
                      f"({stage.t[k] / max(stage.n[k], 1) * 1000:.0f} ms/lần)")
    console.print(f"   khác    {other:7.2f}s  {100 * other / wall:4.0f}%   (tạo Candidate, fusion, Python)")
    if stage.pairs:
        console.print(f"   rerank: {stage.pairs / n_q:.0f} cặp/query")


def overlap(a: Dict[int, List[str]], b: Dict[int, List[str]]) -> float:
    js = []
    for i in a:
        sa, sb = set(a[i]), set(b[i])
        if sa | sb:
            js.append(len(sa & sb) / len(sa | sb))
    return statistics.mean(js) if js else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--queries", default="data/queries/real_queries.jsonl")
    p.add_argument("--augmented", default="data/augment_v2/augmented.jsonl")
    p.add_argument("--collection", default="medical_documents")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.queries, encoding="utf-8") if l.strip()]
    queries = [QueryItem(id=r["id"], text=r["query"]) for r in rows[: args.n]]
    warm = [QueryItem(id=r["id"], text=r["query"]) for r in rows[args.n: args.n + 2]]
    augmented = load_augmented(args.augmented)
    miss = [q.id for q in queries if q.id not in augmented]
    console.print(f"{len(queries)} query, {len(miss)} thiếu trong {args.augmented}: {miss[:5]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = BgeM3Embedder("BAAI/bge-m3", device=device, use_fp16=False, max_length=1024)
    vs = VectorStore(encoder, url=args.qdrant_url, collection_name=args.collection)
    pipeline = RetrievalPipeline(vs, PipelineConfig())
    runners = {
        "A  (câu gốc)": pipeline,
        "B' (concat)": MultiQueryPipeline(pipeline, augmented, MultiQueryConfig(mode="concat")),
        "C  (lines)": MultiQueryPipeline(pipeline, augmented, MultiQueryConfig(mode="lines")),
    }

    stage = Stage()
    for r in runners.values():
        for w in warm:
            r.run(w, False)

    console.rule("fp32: attribution (nhánh ngôn ngữ chạy tuần tự)")
    fp32_tops: Dict[str, Dict[int, List[str]]] = {}
    for name, r in runners.items():
        restore = instrument(pipeline, stage)
        wall, tops = run_mode(name, r, queries, stage, parallel=False)
        restore()
        fp32_tops[name] = tops
        report(name, wall, stage, len(queries))

    console.rule("fp32: wall thật (nhánh ngôn ngữ song song, như run_full_submission)")
    for name, r in runners.items():
        wall, _ = run_mode(name, r, queries, stage, parallel=True)
        console.print(f"{name}: {wall:.1f}s = {wall / len(queries) * 1000:.0f} ms/query")

    console.rule("fp16: encoder + reranker")
    del vs.encoder
    torch.cuda.empty_cache()
    vs.encoder = BgeM3Embedder("BAAI/bge-m3", device=device, use_fp16=True, max_length=1024)
    pipeline.reranker.model = CrossEncoder(
        pipeline.config.reranker_model, device=device, max_length=512,
        model_kwargs={"torch_dtype": torch.float16},
    )
    for r in runners.values():
        for w in warm:
            r.run(w, False)
    for name, r in runners.items():
        restore = instrument(pipeline, stage)
        wall, tops = run_mode(name, r, queries, stage, parallel=False)
        restore()
        report(name + " fp16", wall, stage, len(queries))
        console.print(f"   chồng lấn top-20 với fp32 (Jaccard chunk): {overlap(fp32_tops[name], tops):.3f}")

    console.rule("micro: encode theo lô (fp16) trên các chuỗi truy xuất của B'")
    texts = []
    for q in queries:
        rec = augmented.get(q.id)
        if rec:
            texts += [(u.get("augmented") or u.get("text")) for u in rec["units"]]
    texts = texts or [q.text for q in queries]
    for bs in (1, 8, 16, 32):
        sync(); t = time.perf_counter()
        vs.encoder.encode(texts, batch_size=bs)
        sync()
        dt = time.perf_counter() - t
        console.print(f"   {len(texts)} chuỗi, batch_size={bs:2d}: {dt:.2f}s = {dt / len(texts) * 1000:.0f} ms/chuỗi")

    console.rule("micro: rerank batch_size (fp16) trên 150 cặp thật")
    q0 = queries[0]
    cands = runners["B' (concat)"].run(q0, False)
    res = getattr(cands, "result", cands)
    pool = [c.content for c in res.fused][:150]
    pairs = [(q0.text, c) for c in pool]
    for bs in (16, 32, 64, 128):
        sync(); t = time.perf_counter()
        pipeline.reranker.model.predict(pairs, batch_size=bs)
        sync()
        console.print(f"   {len(pairs)} cặp, batch_size={bs:3d}: {time.perf_counter() - t:.2f}s")


if __name__ == "__main__":
    main()
