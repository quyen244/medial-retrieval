"""Run the full baseline pipeline over the official 1200-query set and emit
the competition submission JSON.

Critical-path requirement: a submission missing ANY query id is rejected
outright. This script therefore:
  - iterates every row of get_dataset('query') (not a sample),
  - wraps each query in try/except so one failure cannot drop it or crash
    the run - a failed/empty query still gets a valid empty entry
    (relevant_docs=[], relevant_chunks=[]),
  - writes incrementally to a JSONL checkpoint file so a long run can be
    resumed instead of restarted from zero,
  - verifies at the end that the output has exactly one entry per query id
    in the official set, in order, before calling the run complete.

Usage:
    python -m src.rag.run_full_submission --collection vibiomir_smoke_test --qdrant-url http://localhost:6333
"""
import argparse
import json
import time
from pathlib import Path

import mlflow
import torch
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from src.ingestion.utils import get_dataset
from src.rag.embedding import BgeM3Embedder
from src.rag.pipeline.models import QueryItem
from src.rag.pipeline.pipeline import PipelineConfig, RetrievalPipeline
from src.rag.pipeline.submission import empty_submission_entry, result_to_submission_entry
from src.rag.vectorstore import VectorStore

console = Console()


def load_checkpoint(path: Path):
    """id -> entry, for whatever has already been written."""
    done = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                done[entry["id"]] = entry
    return done


def append_checkpoint(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, default="vibiomir_smoke_test")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--qdrant-path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="tmp/smoke_test/full_submission_checkpoint.jsonl")
    parser.add_argument("--out", type=str, default="tmp/smoke_test/full_submission.json")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N queries (debug use)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[full_submission] device={device}")

    # fp32, not fp16: measured slower on this GTX 1660 Ti (no Tensor Cores) - see report.
    encoder = BgeM3Embedder("BAAI/bge-m3", device=device, use_fp16=False, max_length=1024)
    vectorstore = VectorStore(encoder, url=args.qdrant_url, path=args.qdrant_path, collection_name=args.collection)

    dataset = get_dataset("query")
    rows = list(dataset["train"])
    if args.limit:
        rows = rows[: args.limit]
    all_ids = [r["id"] for r in rows]
    console.print(f"[full_submission] {len(rows)} queries to cover")

    checkpoint_path = Path(args.checkpoint)
    done = load_checkpoint(checkpoint_path)
    console.print(f"[full_submission] checkpoint: {checkpoint_path} ({len(done)} queries already done)")

    config = PipelineConfig()
    pipeline = RetrievalPipeline(vectorstore, config)

    mlflow.set_experiment("retrieval_pipeline_baseline")
    n_ok, n_empty, n_error = 0, 0, 0
    t_start = time.perf_counter()

    with mlflow.start_run(run_name=f"full_submission_{int(t_start)}"):
        flat_params = {}
        for stage, params in pipeline.component_params().items():
            if isinstance(params, dict):
                for k, v in params.items():
                    flat_params[f"{stage}.{k}"] = str(v)
            else:
                flat_params[stage] = str(params)
        flat_params["num_queries"] = len(rows)
        mlflow.log_params(flat_params)

        columns = [SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
                   TimeElapsedColumn(), TimeRemainingColumn()]
        with Progress(*columns, console=console) as progress:
            task = progress.add_task("running queries", total=len(rows))
            for row in rows:
                qid = row["id"]
                progress.update(task, advance=1)
                if qid in done:
                    continue
                query = QueryItem(id=qid, text=row["query"])
                try:
                    result = pipeline.run(query)
                    entry = result_to_submission_entry(result)
                    if not entry["relevant_chunks"]:
                        n_empty += 1
                    else:
                        n_ok += 1
                except Exception as e:
                    console.print(f"[full_submission] WARN: query {qid} failed: {type(e).__name__}: {e}")
                    entry = empty_submission_entry(qid)
                    n_error += 1
                append_checkpoint(checkpoint_path, entry)
                done[qid] = entry

        elapsed = time.perf_counter() - t_start
        mlflow.log_metrics({
            "n_ok": n_ok, "n_empty": n_empty, "n_error": n_error,
            "elapsed_seconds": elapsed, "queries_per_second": len(rows) / elapsed if elapsed else 0,
        })

    # Assemble final submission strictly in official query-id order, verifying completeness.
    missing = [qid for qid in all_ids if qid not in done]
    if missing:
        console.print(f"[full_submission] ERROR: {len(missing)} query ids missing after run: {missing[:10]}...")
        for qid in missing:
            done[qid] = empty_submission_entry(qid)

    final_entries = [done[qid] for qid in all_ids]
    assert len(final_entries) == len(all_ids), "submission entry count must match query count"
    assert [e["id"] for e in final_entries] == all_ids, "submission must cover every query id, in order"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_entries, f, ensure_ascii=False, indent=2)

    console.print(f"\n[full_submission] done: {len(final_entries)} entries written to {out_path}")
    console.print(f"  ok (non-empty): {n_ok}, empty: {n_empty}, errored->empty: {n_error}")
    console.print(f"  wall time this run: {elapsed:.1f}s")
    console.print(f"  VERIFIED: all {len(all_ids)} official query ids present, in order")


if __name__ == "__main__":
    main()
