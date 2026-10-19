"""Document -> vector store ingestion entrypoint (Task A).

Walks data/raw_documents/{domain}/{doc_id}.json, chunks each doc via the
stable default_dispatcher, embeds with bge-m3, and upserts into Qdrant.

Correctness contract (the user will re-crawl and re-run this repeatedly, so
all cases below must be handled, not just "skip what's done"). Two
INDEPENDENT fingerprints are tracked per document (see ingest_ledger.py):
vector_fingerprint (chunker params/version, embedding model, max_seq_length,
sparse on/off) and payload_fingerprint (just the extra metadata schema
version - see payload_backfill.py). This split matters: collapsing them
would mean adding one metadata field triggers a full GPU re-embed of the
whole corpus.

  - NEW document (not in the ledger)                 -> chunk, embed, insert
    (vectors + full payload).
  - UNCHANGED vector AND payload (same sha256,
    same vector_fingerprint, same payload_fingerprint) -> skip entirely, no
    GPU touch, no Qdrant write.
  - UNCHANGED vector, CHANGED payload (e.g. a new
    metadata field was added)                        -> payload-only
    backfill: look up the doc's existing points by doc_id, set_payload() the
    recomputed metadata fields, NO re-chunk, NO re-embed.
  - CHANGED vector (sha256 or vector_fingerprint
    differs)                                          -> delete every
    existing point for that doc_id first (by payload filter), THEN insert
    the fresh chunk set. Deterministic point ids alone are not enough here:
    if the new chunking yields fewer chunks than before, the surplus old
    points (..._c005, _c006, ...) would otherwise survive as orphaned stale
    content forever.

Crash-safe write ordering: Qdrant is updated (delete-if-changed, then
upsert) BEFORE the ledger is recorded. If the process dies in between, the
ledger still shows the old/no entry, so the next run just redoes that one
document - a harmless overwrite thanks to deterministic point ids - rather
than silently losing it.

Usage examples:
    python -m src.rag.ingest_documents --languages vi,en,zh --max-docs-per-language 300
    python -m src.rag.ingest_documents --domains bachmai.gov.vn --max-docs 50
    python -m src.rag.ingest_documents --qdrant-url http://localhost:6333 --fp16 --max-seq-length 1024
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)

from src.rag.chunker import default_dispatcher
from src.rag.ingest_ledger import IngestLedger, compute_payload_fingerprint, compute_vector_fingerprint
from src.rag.payload_backfill import PAYLOAD_VERSION, extract_metadata_fields
from src.rag.vectorstore import VectorStore

ROOT = Path(__file__).parent.parent.parent
RAW_DOCS_DIR = ROOT / "data" / "raw_documents"
LEDGER_DIR = ROOT / "data" / "ingest_progress"
CHUNKS_CACHE_DIR = ROOT / "data" / "chunks_cache"

console = Console()


def iter_raw_documents(
    root: Path,
    domains: Optional[Set[str]] = None,
) -> Iterator[Path]:
    """Yield doc json paths under root, optionally filtered by domain name."""
    for domain_dir in sorted(root.iterdir()):
        if not domain_dir.is_dir():
            continue
        if domains and domain_dir.name not in domains:
            continue
        for doc_path in sorted(domain_dir.glob("*.json")):
            yield doc_path


def append_chunks_cache(cache_path: Path, chunks: list) -> None:
    """Persist chunks to disk independently of embedding (JSONL keyed by
    chunk_id), so re-embedding with a different model later does not require
    re-chunking the raw documents."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Ingest data/raw_documents into Qdrant.")
    parser.add_argument("--languages", type=str, default=None, help="Comma-separated language_category filter, e.g. vi,en,zh")
    parser.add_argument("--domains", type=str, default=None, help="Comma-separated domain filter, e.g. bachmai.gov.vn,vnexpress.net")
    parser.add_argument("--max-docs", type=int, default=None, help="Global cap on number of documents processed (new+changed) this run")
    parser.add_argument("--max-docs-per-language", type=int, default=None, help="Cap on documents ingested per language, counted cumulatively via the ledger across runs")
    parser.add_argument("--batch-size", type=int, default=256, help="Chunks accumulated before one embed+upsert call (ledger-commit and Qdrant-upsert granularity; NOT the GPU sub-batch size - see --encode-batch-size)")
    parser.add_argument("--encode-batch-size", type=int, default=4, help="encoder.encode_with_sparse() internal GPU sub-batch size. Default 4: measured fastest "
                         "(10.4-10.9 chunks/s) on a GTX 1660 Ti 6GB at max_length=1024; throughput falls off inversely as this "
                         "grows (bs=32 -> 6.82 chunks/s), and large values (e.g. the old default of 256, which equaled --batch-size) "
                         "were measured catastrophically slower still (~0.6-1 chunk/s, i.e. 1000ms+/doc) on a 6GB card under real "
                         "desktop GPU contention - almost certainly a CUDA allocator/WDDM effect of requesting one huge activation "
                         "buffer rather than a FlagEmbedding or Qdrant issue. Re-measure on different hardware before trusting 4.")
    parser.add_argument("--qdrant-url", type=str, default=None, help="Qdrant server URL, e.g. http://localhost:6333. Omit to use embedded on-disk mode.")
    parser.add_argument("--qdrant-path", type=str, default="data/qdrant_db", help="Embedded Qdrant on-disk path (used when --qdrant-url is not given)")
    parser.add_argument("--collection", type=str, default="medical_documents")
    parser.add_argument("--ledger-path", type=str, default=None, help="Override path to the SQLite ledger")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu; default: autodetect")
    parser.add_argument("--raw-docs-dir", type=str, default=str(RAW_DOCS_DIR))
    parser.add_argument("--max-seq-length", type=int, default=1024,
                         help="bge-m3 defaults to 8192; chunks are capped at ~800-1000 chars so this is "
                              "measured-safe headroom (see report) at a fraction of the O(n^2) attention cost.")
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=False,
                         help="Run the encoder in fp16. Default OFF: measured 2-4x SLOWER than fp32 on a "
                              "GTX 1660 Ti (no Tensor Cores) across repeated trials - see report. Pass "
                              "--fp16 explicitly only on a GPU with Tensor Cores (RTX/A/H-series) where it "
                              "should actually help; benchmark before trusting it there too.")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--bulk-mode", action="store_true", default=True,
                         help="Disable incremental HNSW indexing during the run, restore it at the end (default on)")
    parser.add_argument("--no-bulk-mode", dest="bulk_mode", action="store_false")
    parser.add_argument("--chunks-cache", type=str, default=None, help="Path to a JSONL file to append produced chunks to, independent of embedding. Default: data/chunks_cache/<collection>.jsonl")
    parser.add_argument("--no-chunks-cache", action="store_true", help="Disable chunk-cache persistence")
    args = parser.parse_args()

    languages = set(x.strip().lower() for x in args.languages.split(",")) if args.languages else None
    domains = set(x.strip() for x in args.domains.split(",")) if args.domains else None

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[ingest] device={device} cuda_available={torch.cuda.is_available()} fp16={args.fp16} max_seq_length={args.max_seq_length}")

    from src.rag.embedding import BgeM3Embedder
    embedding_model_name = "BAAI/bge-m3"
    encoder = BgeM3Embedder(embedding_model_name, device=device, use_fp16=args.fp16, max_length=args.max_seq_length)

    vectorstore = VectorStore(
        encoder,
        url=args.qdrant_url,
        path=None if args.qdrant_url else args.qdrant_path,
        collection_name=args.collection,
    )
    vectorstore.ensure_payload_index("language", "keyword")
    vectorstore.ensure_payload_index("domain", "keyword")
    vectorstore.ensure_payload_index("doc_id", "integer")

    ledger_path = Path(args.ledger_path) if args.ledger_path else LEDGER_DIR / f"{args.collection}.sqlite"
    ledger = IngestLedger(ledger_path)
    console.print(f"[ingest] ledger: {ledger_path} ({ledger.total_count()} docs tracked; per language: {ledger.count_by_language()})")

    chunks_cache_path = None
    if not args.no_chunks_cache:
        chunks_cache_path = Path(args.chunks_cache) if args.chunks_cache else CHUNKS_CACHE_DIR / f"{args.collection}.jsonl"
        console.print(f"[ingest] chunk cache: {chunks_cache_path} (append-only; lets you re-embed later without re-chunking)")

    # Per-language vector-fingerprint cache (one chunker instance per
    # language, so params are computed once, not per document).
    vector_fp_cache: Dict[str, str] = {}
    sparse_enabled = hasattr(encoder, "encode_with_sparse")

    def vector_fingerprint_for(lang: str) -> str:
        if lang not in vector_fp_cache:
            chunker = default_dispatcher._resolve_chunker(lang)
            vector_fp_cache[lang] = compute_vector_fingerprint(
                chunker.get_params(), embedding_model_name, args.max_seq_length, sparse_enabled
            )
        return vector_fp_cache[lang]

    payload_fp = compute_payload_fingerprint(PAYLOAD_VERSION)
    console.print(f"[ingest] sparse_enabled={sparse_enabled} payload_fingerprint={payload_fp} (PAYLOAD_VERSION={PAYLOAD_VERSION})")

    # Seed per-language counts from the ledger so --max-docs-per-language caps
    # the *collection's* contents cumulatively across runs, not just this run.
    per_lang_cap: Dict[str, int] = dict(ledger.count_by_language())

    raw_root = Path(args.raw_docs_dir)
    paths = list(iter_raw_documents(raw_root, domains=domains))

    counters = {"new": 0, "unchanged": 0, "changed": 0, "payload_only": 0, "warn": 0, "chunks": 0}
    t_start = time.perf_counter()
    t_gpu = 0.0

    pending_chunks = []
    pending_ledger_rows: List[tuple] = []  # (doc_key, doc_id, domain, language, sha256, vector_fp)
    pending_delete_doc_ids: List[Any] = []

    def flush():
        nonlocal t_gpu
        if not pending_chunks and not pending_delete_doc_ids:
            return
        if pending_delete_doc_ids:
            vectorstore.delete_by_doc_ids(pending_delete_doc_ids)
            pending_delete_doc_ids.clear()
        if pending_chunks:
            t0 = time.perf_counter()
            vectorstore.upsert_chunks(
                pending_chunks,
                batch_size=len(pending_chunks),
                encode_batch_size=args.encode_batch_size,
            )
            t_gpu += time.perf_counter() - t0
            counters["chunks"] += len(pending_chunks)
            if chunks_cache_path:
                append_chunks_cache(chunks_cache_path, pending_chunks)
            pending_chunks.clear()
        # Ledger is recorded AFTER Qdrant is durably updated (crash-safe order).
        for row in pending_ledger_rows:
            doc_key, doc_id, domain, lang, sha256, vfp, chunk_count = row
            ledger.record(doc_key, doc_id, domain, lang, sha256, chunk_count, vfp, payload_fp)
        ledger.commit()
        pending_ledger_rows.clear()

    if args.bulk_mode:
        vectorstore.set_bulk_load_mode(True)

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("new={task.fields[new]} unchanged={task.fields[unchanged]} changed={task.fields[changed]} "
                   "payload_only={task.fields[payload_only]} chunks={task.fields[chunks]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    try:
        with Progress(*progress_columns, console=console) as progress:
            task = progress.add_task(
                "ingesting", total=len(paths), new=0, unchanged=0, changed=0, payload_only=0, chunks=0
            )

            for doc_path in paths:
                progress.update(task, advance=1)

                if args.max_docs is not None and (counters["new"] + counters["changed"]) >= args.max_docs:
                    break
                if args.max_docs_per_language is not None and languages:
                    if all(per_lang_cap.get(lang, 0) >= args.max_docs_per_language for lang in languages):
                        break

                domain = doc_path.parent.name
                doc_key = f"{domain}/{doc_path.stem}"

                try:
                    with open(doc_path, "r", encoding="utf-8") as f:
                        doc = json.load(f)
                except Exception as e:
                    counters["warn"] += 1
                    continue

                lang = (doc.get("source", {}) or {}).get("language_category")
                if languages and lang not in languages:
                    continue
                if args.max_docs_per_language is not None and per_lang_cap.get(lang, 0) >= args.max_docs_per_language:
                    continue

                doc_id = doc.get("source", {}).get("doc_id")
                sha256 = doc.get("content", {}).get("sha256", "")
                vfp = vector_fingerprint_for(lang)

                state = ledger.classify(doc_key, sha256, vfp, payload_fp)

                if state.vector == "unchanged" and state.payload == "unchanged":
                    counters["unchanged"] += 1
                    continue  # fast path: no chunking, no embedding, no Qdrant write at all

                if state.vector == "unchanged" and state.payload == "changed":
                    # Payload-only backfill: no re-chunk, no re-embed - just
                    # patch the recomputed metadata fields on existing points.
                    point_ids = vectorstore.point_ids_by_doc_id(doc_id)
                    if point_ids:
                        vectorstore.set_payload_fields(point_ids, extract_metadata_fields(doc))
                        ledger.record_payload_only(doc_key, payload_fp)
                        ledger.commit()
                        counters["payload_only"] += 1
                    else:
                        counters["warn"] += 1
                    progress.update(task, payload_only=counters["payload_only"])
                    continue

                # state.vector is "new" or "changed": full re-chunk + re-embed path.
                try:
                    chunks = default_dispatcher.chunk(doc, versioning=False)
                except Exception as e:
                    console.print(f"[ingest] WARN: chunking failed for {doc_path}: {e}")
                    counters["warn"] += 1
                    continue

                extra_fields = extract_metadata_fields(doc)
                for c in chunks:
                    c.metadata.update(extra_fields)  # flattened to top-level payload keys by chunk_to_payload()

                if state.vector == "changed":
                    pending_delete_doc_ids.append(doc_id)
                    counters["changed"] += 1
                else:
                    counters["new"] += 1

                pending_chunks.extend(chunks)
                pending_ledger_rows.append((doc_key, doc_id, domain, lang, sha256, vfp, len(chunks)))
                per_lang_cap[lang] = per_lang_cap.get(lang, 0) + 1

                progress.update(task, new=counters["new"], unchanged=counters["unchanged"],
                                 changed=counters["changed"], chunks=counters["chunks"])

                if len(pending_chunks) >= args.batch_size:
                    flush()
                    progress.update(task, chunks=counters["chunks"])

            flush()
    finally:
        if args.bulk_mode:
            vectorstore.set_bulk_load_mode(False)
        ledger.close()

    elapsed = time.perf_counter() - t_start
    docs_processed = counters["new"] + counters["changed"]

    console.print("\n[ingest] summary")
    console.print(f"  candidate docs:   {len(paths)} (may have early-stopped before scanning all of them)")
    console.print(f"  new:              {counters['new']}")
    console.print(f"  changed:          {counters['changed']}")
    console.print(f"  payload-only:     {counters['payload_only']}")
    console.print(f"  unchanged(skip):  {counters['unchanged']}")
    console.print(f"  warnings:         {counters['warn']}")
    console.print(f"  chunks upserted:  {counters['chunks']}")
    console.print(f"  wall time:        {elapsed:.1f}s (GPU embed+upsert: {t_gpu:.1f}s)")
    if docs_processed > 0:
        docs_per_sec = docs_processed / elapsed
        chunks_per_sec = counters["chunks"] / elapsed if elapsed > 0 else 0
        chunks_per_doc = counters["chunks"] / docs_processed
        console.print(f"  throughput:       {docs_per_sec:.2f} docs/s, {chunks_per_sec:.2f} chunks/s, {chunks_per_doc:.1f} chunks/doc")
        target_docs = 100_000
        full_corpus_docs = 553_047
        console.print(f"  extrapolated 100K-doc ingest (target): {target_docs / docs_per_sec / 3600:.1f} hours at this throughput")
        console.print(f"  extrapolated full ingest ({full_corpus_docs} docs, secondary): {full_corpus_docs / docs_per_sec / 3600:.1f} hours")


if __name__ == "__main__":
    main()
