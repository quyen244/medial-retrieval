"""Payload backfill: the first-class operation for adding/changing metadata
fields on already-ingested chunks WITHOUT re-chunking or re-embedding.

PAYLOAD_VERSION is the single source of truth for "what extra metadata do we
attach beyond the chunk's own attributes" (doc_id, chunk_id, chunk_index,
content, language, domain, url, title, heading_path, char_length come from
the Chunk object itself and never need backfilling). Bump PAYLOAD_VERSION
and extend `extract_metadata_fields()` whenever a new field is added
(date-based filtering, source-quality signals, topic labels, ...); every
already-ingested document's payload_fingerprint will then mismatch the new
version and get backfilled - cheaply, since this never touches vectors.

Two ways to run a backfill:
  1. Opportunistically, inline in src/rag/ingest_documents.py's normal scan
     (a doc with vector="unchanged" but payload="changed" gets its payload
     patched in place as the scan passes over it).
  2. Standalone via this module's CLI, which only visits documents already
     in the ledger (no raw-file scan needed) - useful to backfill a new field
     across the whole index right away instead of waiting for the next full
     ingest scan to pass over every document.

Resumability and correctness: each doc's payload_fingerprint is recorded
only after Qdrant's set_payload() call for it durably succeeds (ledger write
is last, same crash-safe ordering as the main ingest). Qdrant filters like
`field == X` silently exclude points that lack the field entirely, so a
half-finished backfill wouldn't otherwise surface as an error anywhere -
count_missing_field()/--report-missing make that state detectable instead of
invisible.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from src.rag.ingest_ledger import IngestLedger, compute_payload_fingerprint
from src.rag.vectorstore import VectorStore

ROOT = Path(__file__).parent.parent.parent
RAW_DOCS_DIR = ROOT / "data" / "raw_documents"
LEDGER_DIR = ROOT / "data" / "ingest_progress"

# Bump this whenever extract_metadata_fields() changes what it returns.
PAYLOAD_VERSION = "v1-published_date"

console = Console()


def extract_metadata_fields(doc: Dict[str, Any]) -> Dict[str, Any]:
    """The extra (non-chunk-native) fields attached to every chunk of a
    document. Currently just published_date - free, already in the source
    JSON. Extend this (and bump PAYLOAD_VERSION) to add more later."""
    source = doc.get("source", {}) or {}
    return {
        "published_date": source.get("published_date"),
    }


def backfill_one_doc(vectorstore: VectorStore, ledger: IngestLedger, raw_docs_dir: Path,
                      doc_key: str, domain: str, doc_id: Any, payload_fp: str) -> bool:
    """Patch payload fields for every existing point of one document. Returns
    True if the doc's file was found and the patch was applied."""
    doc_path = raw_docs_dir / domain / f"{doc_id}.json"
    if not doc_path.exists():
        console.print(f"[backfill] WARN: source file missing for {doc_key}, skipping")
        return False

    with open(doc_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    fields = extract_metadata_fields(doc)
    point_ids = vectorstore.point_ids_by_doc_id(doc_id)
    if not point_ids:
        console.print(f"[backfill] WARN: no points found in Qdrant for doc_id={doc_id} ({doc_key}), skipping")
        return False

    vectorstore.set_payload_fields(point_ids, fields)  # Qdrant updated FIRST
    ledger.record_payload_only(doc_key, payload_fp)     # ledger updated AFTER (crash-safe order)
    ledger.commit()
    return True


def main():
    parser = argparse.ArgumentParser(description="Backfill payload metadata fields without re-chunking/re-embedding.")
    parser.add_argument("--collection", type=str, default="medical_documents")
    parser.add_argument("--qdrant-url", type=str, default=None)
    parser.add_argument("--qdrant-path", type=str, default="data/qdrant_db")
    parser.add_argument("--ledger-path", type=str, default=None)
    parser.add_argument("--raw-docs-dir", type=str, default=str(RAW_DOCS_DIR))
    parser.add_argument("--report-missing", type=str, default=None,
                         help="Instead of backfilling, report how many points are missing this payload field (detects a half-finished backfill) and exit.")
    args = parser.parse_args()

    # No embedder needed for a payload-only operation - vectors are never touched.
    vectorstore = VectorStore(encoder=None, url=args.qdrant_url,
                               path=None if args.qdrant_url else args.qdrant_path,
                               collection_name=args.collection)

    if args.report_missing:
        missing = vectorstore.count_missing_field(args.report_missing)
        console.print(f"[backfill] points missing '{args.report_missing}': {missing}")
        return

    ledger_path = Path(args.ledger_path) if args.ledger_path else LEDGER_DIR / f"{args.collection}.sqlite"
    ledger = IngestLedger(ledger_path)
    payload_fp = compute_payload_fingerprint(PAYLOAD_VERSION)
    console.print(f"[backfill] target payload_fingerprint={payload_fp} (PAYLOAD_VERSION={PAYLOAD_VERSION})")

    rows = list(ledger.rows_with_stale_payload(payload_fp))
    console.print(f"[backfill] {len(rows)} documents need a payload backfill")

    raw_docs_dir = Path(args.raw_docs_dir)
    ok, failed = 0, 0
    columns = [SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
               TimeElapsedColumn(), TimeRemainingColumn()]
    with Progress(*columns, console=console) as progress:
        task = progress.add_task("backfilling payload", total=len(rows))
        for row in rows:
            success = backfill_one_doc(vectorstore, ledger, raw_docs_dir,
                                        row["doc_key"], row["domain"], row["doc_id"], payload_fp)
            ok += int(success)
            failed += int(not success)
            progress.update(task, advance=1)

    ledger.close()
    console.print(f"\n[backfill] done: {ok} patched, {failed} failed/skipped")
    for field in extract_metadata_fields({}).keys():
        missing = vectorstore.count_missing_field(field)
        console.print(f"[backfill] points still missing '{field}': {missing}")


if __name__ == "__main__":
    main()
