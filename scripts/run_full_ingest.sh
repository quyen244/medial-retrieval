#!/usr/bin/env bash
# Full ingest into the Qdrant server: fp16, encode-batch-size 4 (fastest in benchmark).
# Resumable: the per-collection ledger skips docs that were already ingested.
set -euo pipefail
cd "$(dirname "$0")/.."

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
PYTHON="${PYTHON:-venv/bin/python}"

exec "$PYTHON" -m src.rag.ingest_documents \
  --qdrant-url "$QDRANT_URL" \
  --collection medical_documents \
  --languages vi,en,zh \
  --fp16 \
  --encode-batch-size 4 \
  "$@"
