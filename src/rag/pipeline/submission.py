"""Writes pipeline results in the exact competition submission format:

[{"id": <int query id>, "relevant_docs": ["<doc id>"],
  "relevant_chunks": [{"doc_id": "<doc id>", "chunk_text": "<verbatim text>"}]}]
"""
import json
from typing import Any, Dict, List

from src.rag.pipeline.pipeline import PipelineResult


def empty_submission_entry(query_id: int) -> Dict[str, Any]:
    """A valid, schema-complete entry for a query that produced no results
    (or whose pipeline run failed). A submission missing any query id is
    rejected outright, so every query must get an entry - an empty one is
    valid; a missing one is not."""
    return {"id": int(query_id), "relevant_docs": [], "relevant_chunks": []}


def result_to_submission_entry(result: PipelineResult) -> Dict[str, Any]:
    doc_ids_seen = []
    seen = set()
    chunks = []
    for c in result.reranked:
        doc_id_str = str(c.doc_id)
        if doc_id_str not in seen:
            seen.add(doc_id_str)
            doc_ids_seen.append(doc_id_str)
        chunks.append({"doc_id": doc_id_str, "chunk_text": c.content})

    return {
        "id": int(result.query.id),
        "relevant_docs": doc_ids_seen,
        "relevant_chunks": chunks,
    }


def write_submission(results: List[PipelineResult], out_path: str) -> List[Dict[str, Any]]:
    entries = [result_to_submission_entry(r) for r in results]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    return entries
