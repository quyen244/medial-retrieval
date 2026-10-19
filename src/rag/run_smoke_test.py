"""Task C: one real query end-to-end through the full baseline pipeline.

Ingests a small multi-language subset (or reuses one already ingested into
the given collection), then runs ONE Vietnamese query from the official
query set through the pipeline, printing:
  - the trace (per-stage timings + candidate counts)
  - an mlflow run with logged params/metrics
  - the top-20 reranked chunks (doc_id, language, scores)
  - the submission-format JSON for that query
  - a verbatim round-trip check (submission chunk_text vs stored chunk content)

Usage:
    python -m src.rag.run_smoke_test --query-id 1 --collection vibiomir_smoke_test --qdrant-url http://localhost:6333
"""
import argparse
import json

import mlflow
import torch

from src.ingestion.utils import get_dataset
from src.rag.embedding import BgeM3Embedder
from src.rag.pipeline.models import QueryItem
from src.rag.pipeline.pipeline import PipelineConfig, RetrievalPipeline
from src.rag.pipeline.submission import result_to_submission_entry
from src.rag.vectorstore import VectorStore, deterministic_point_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-id", type=int, default=1)
    parser.add_argument("--collection", type=str, default="vibiomir_smoke_test")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--qdrant-path", type=str, default=None)
    parser.add_argument("--out", type=str, default="tmp/smoke_test/submission_sample.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}")

    # fp32, not fp16: measured on this GTX 1660 Ti (no Tensor Cores), fp16 was
    # consistently 2-4x SLOWER than fp32 across repeated trials (see report).
    encoder = BgeM3Embedder("BAAI/bge-m3", device=device, use_fp16=False, max_length=1024)
    vectorstore = VectorStore(
        encoder,
        url=args.qdrant_url,
        path=args.qdrant_path,
        collection_name=args.collection,
    )

    dataset = get_dataset("query")
    row = next(r for r in dataset["train"] if r["id"] == args.query_id)
    query = QueryItem(id=row["id"], text=row["query"])
    print(f"[smoke] query id={query.id}: {query.text}")

    config = PipelineConfig()
    pipeline = RetrievalPipeline(vectorstore, config)

    mlflow.set_experiment("retrieval_pipeline_baseline")
    with mlflow.start_run(run_name=f"smoke_query_{query.id}"):
        flat_params = {}
        for stage, params in pipeline.component_params().items():
            if isinstance(params, dict):
                for k, v in params.items():
                    flat_params[f"{stage}.{k}"] = str(v)
            else:
                flat_params[stage] = str(params)
        mlflow.log_params(flat_params)

        result = pipeline.run(query)

        mlflow.log_metrics(result.trace.mlflow_metrics())

        trace_path = "tmp/smoke_test/trace.json"
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write(result.trace.to_json())
        mlflow.log_artifact(trace_path)

        print("\n=== TRACE ===")
        print(result.trace.to_json())

        print("\n=== BRANCH CANDIDATE COUNTS ===")
        for lang, cands in result.branch_results.items():
            print(f"  {lang}: {len(cands)} candidates")

        print(f"\n=== TOP {len(result.reranked)} RERANKED CHUNKS ===")
        for i, c in enumerate(result.reranked, start=1):
            print(f"{i:2d}. doc_id={c.doc_id} lang={c.language} domain={c.domain} "
                  f"retrieval_rank={c.retrieval_rank} fusion_score={c.fusion_score:.5f} "
                  f"rerank_score={c.rerank_score:.4f}")
            print(f"    title: {c.title}")
            print(f"    text:  {c.content[:150]!r}...")

        entry = result_to_submission_entry(result)
        import os
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump([entry], f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(args.out)

        print("\n=== SUBMISSION JSON ===")
        print(json.dumps([entry], ensure_ascii=False, indent=2)[:3000])

        # Verbatim round-trip check: submission chunk_text must exactly match
        # what is stored in Qdrant payload for that chunk_id.
        print("\n=== VERBATIM ROUND-TRIP CHECK ===")
        all_ok = True
        for c in result.reranked:
            stored = vectorstore.client.retrieve(
                collection_name=args.collection,
                ids=[deterministic_point_id(c.chunk_id)],
                with_payload=True,
            )
            if not stored:
                print(f"  MISMATCH: chunk_id={c.chunk_id} not found in store")
                all_ok = False
                continue
            stored_content = stored[0].payload.get("content")
            match = stored_content == c.content
            if not match:
                all_ok = False
            print(f"  chunk_id={c.chunk_id} byte_identical={match}")
        print(f"\nALL CHUNKS VERBATIM-MATCH STORED CONTENT: {all_ok}")

        languages_present = sorted(set(c.language for c in result.reranked))
        print(f"\nLanguage mix in top-{len(result.reranked)}: {languages_present}")
        for lang in config.languages:
            n = len(result.branch_results.get(lang, []))
            print(f"  branch '{lang}' returned {n} candidates "
                  f"({'OK' if n > 0 else 'EMPTY - check coverage/ingestion for this language'})")


if __name__ == "__main__":
    main()
