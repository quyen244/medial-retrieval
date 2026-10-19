"""SQLite ledger tracking per-document ingestion state.

A JSON/JSONL ledger does not scale to ~553k rows (must load/rewrite the
whole file every run); SQLite with an index on doc_key is the right size for
this: O(1) point lookups, no full-file rewrite, safe concurrent reads.

Two INDEPENDENT fingerprints are tracked per document, not one combined
fingerprint:

  - vector_fingerprint: chunker params/version, embedding model name,
    max_seq_length, and whether sparse vectors are being computed. A mismatch
    means the stored VECTORS (and chunk text/boundaries) are stale -> the
    document must be re-chunked and re-embedded.
  - payload_fingerprint: a version tag over the set of extra metadata fields
    written into the payload (e.g. published_date). A mismatch means only
    the PAYLOAD is stale -> existing points get set_payload'd with the
    recomputed fields, with no re-chunking and no re-embedding.

Keeping these separate is the whole point: collapsing them into one
fingerprint would mean adding a trivial metadata field (e.g. a new quality
signal) invalidates every document and triggers a full GPU re-embed of the
entire corpus - exactly the outcome this split is designed to prevent.
"""
import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS ingested_docs (
    doc_key TEXT PRIMARY KEY,
    doc_id INTEGER,
    domain TEXT,
    language TEXT,
    sha256 TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    vector_fingerprint TEXT NOT NULL,
    payload_fingerprint TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    payload_updated_at TEXT
);
"""


def _stringify(value: Any) -> str:
    """Stable-ish string form of a value for fingerprinting: primitives are
    repr()'d directly, anything else (chunker helper instances etc.) falls
    back to its class name so swapping an implementation changes the
    fingerprint even though the live object isn't otherwise hashable."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stringify(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}={_stringify(v)}" for k, v in sorted(value.items())) + "}"
    return type(value).__name__


def compute_vector_fingerprint(
    chunker_params: Dict[str, Any],
    embedding_model_name: str,
    max_seq_length: int,
    sparse_enabled: bool,
) -> str:
    """16-hex-char fingerprint of everything that affects stored vectors/text
    for one language branch: chunker params (incl. class names of its
    sub-components), embedding model name, max_seq_length, and whether
    sparse vectors are being computed."""
    raw = (_stringify(chunker_params) + f"|model={embedding_model_name}"
           f"|max_seq_length={max_seq_length}|sparse={sparse_enabled}")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def compute_payload_fingerprint(payload_version: str) -> str:
    """16-hex-char fingerprint over the payload schema version tag - bump
    PAYLOAD_VERSION (see src/rag/payload_backfill.py) whenever a field is
    added/changed in what gets written beyond the chunk's own attributes."""
    return hashlib.sha256(f"payload_version={payload_version}".encode("utf-8")).hexdigest()[:16]


class DocState(NamedTuple):
    vector: str   # "new" | "unchanged" | "changed"
    payload: str  # "new" | "unchanged" | "changed"  ("new" iff vector == "new")


class IngestLedger:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def get(self, doc_key: str) -> Optional[sqlite3.Row]:
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.execute(
            "SELECT * FROM ingested_docs WHERE doc_key = ?", (doc_key,)
        )
        return cur.fetchone()

    def classify(self, doc_key: str, sha256: str, vector_fp: str, payload_fp: str) -> DocState:
        row = self.get(doc_key)
        if row is None:
            return DocState(vector="new", payload="new")
        vector_state = "unchanged" if (row["sha256"] == sha256 and row["vector_fingerprint"] == vector_fp) else "changed"
        payload_state = "unchanged" if row["payload_fingerprint"] == payload_fp else "changed"
        return DocState(vector=vector_state, payload=payload_state)

    def record(self, doc_key: str, doc_id: Any, domain: str, language: str,
               sha256: str, chunk_count: int, vector_fingerprint: str, payload_fingerprint: str) -> None:
        """Full record - used after a new/changed-vector ingest (chunk+embed+payload all fresh)."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.conn.execute(
            """INSERT INTO ingested_docs (doc_key, doc_id, domain, language, sha256, chunk_count, vector_fingerprint, payload_fingerprint, ingested_at, payload_updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(doc_key) DO UPDATE SET
                 doc_id=excluded.doc_id, domain=excluded.domain, language=excluded.language,
                 sha256=excluded.sha256, chunk_count=excluded.chunk_count,
                 vector_fingerprint=excluded.vector_fingerprint, payload_fingerprint=excluded.payload_fingerprint,
                 ingested_at=excluded.ingested_at, payload_updated_at=excluded.payload_updated_at""",
            (doc_key, doc_id, domain, language, sha256, chunk_count, vector_fingerprint, payload_fingerprint, now, now),
        )

    def record_payload_only(self, doc_key: str, payload_fingerprint: str) -> None:
        """Update only the payload_fingerprint - used after a payload-only
        backfill (set_payload, no re-chunk, no re-embed). Leaves sha256,
        chunk_count, vector_fingerprint, ingested_at untouched."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.conn.execute(
            "UPDATE ingested_docs SET payload_fingerprint = ?, payload_updated_at = ? WHERE doc_key = ?",
            (payload_fingerprint, now, doc_key),
        )

    def rows_with_stale_payload(self, payload_fp: str):
        """Yield ledger rows whose payload_fingerprint != payload_fp - the
        work list for a standalone payload backfill run."""
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.execute(
            "SELECT * FROM ingested_docs WHERE payload_fingerprint != ?", (payload_fp,)
        )
        for row in cur:
            yield row

    def commit(self) -> None:
        self.conn.commit()

    def count_by_language(self) -> Dict[str, int]:
        cur = self.conn.execute("SELECT language, COUNT(*) FROM ingested_docs GROUP BY language")
        return dict(cur.fetchall())

    def total_count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM ingested_docs")
        return cur.fetchone()[0]

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
