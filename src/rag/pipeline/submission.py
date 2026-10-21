"""Writes pipeline results in the exact competition submission format:

[{"id": <int query id>, "relevant_docs": ["<doc id>"],
  "relevant_chunks": [{"doc_id": "<doc id>", "chunk_text": "<verbatim text>"}]}]
"""
import json
import re
from typing import Any, Dict, List

from src.rag.pipeline.pipeline import PipelineResult


# Ký tự thoát của markdown (`\-`, `\*`, `\.`...) do bước crawl sinh ra; trang gốc
# không có dấu `\` này. BTC yêu cầu chunk_text là nội dung trích từ tài liệu gốc.
_MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~<])")


def submission_doc_id(doc_id: Any) -> Any:
    """doc_id theo đặc tả BTC: trường `id` KIỂU SỐ NGUYÊN của links_corpus.parquet."""
    try:
        return int(doc_id)
    except (TypeError, ValueError):
        return doc_id


def submission_chunk_text(text: str) -> str:
    return _MD_ESCAPE.sub(r"\1", text or "")


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
        doc_id = submission_doc_id(c.doc_id)
        if doc_id not in seen:
            seen.add(doc_id)
            doc_ids_seen.append(doc_id)
        chunks.append({"doc_id": doc_id, "chunk_text": submission_chunk_text(c.content)})

    return {
        "id": int(result.query.id),
        "relevant_docs": doc_ids_seen,
        "relevant_chunks": chunks,
    }


def result_to_top_record(result: PipelineResult, stats: Dict[str, Any] = None) -> Dict[str, Any]:
    """Toàn bộ tập ứng viên cuối (50 chunk ở branch-rerank) kèm điểm, để lắp
    nhiều kiểu bài nộp offline mà không chạy GPU lại (scripts/assemble_submission.py).

    `submitted` = thứ tự chunk_id của bài nộp thật, nên lắp lại V1 k=20 phải ra
    đúng bài đã nộp. `content` giữ nguyên văn vì bài nộp cần chunk_text y hệt."""
    cands = sorted(result.fused, key=lambda c: (c.rerank_score is None, -(c.rerank_score or 0.0)))
    return {
        "id": int(result.query.id),
        "query": result.query.text,
        "submitted": [str(c.chunk_id) for c in result.reranked],
        "stats": stats or {},
        "candidates": [
            {
                "chunk_id": str(c.chunk_id),
                "doc_id": str(c.doc_id),
                "language": c.language,
                "branch": c.branch,
                "retrieval_rank": c.retrieval_rank,
                "rerank_score": c.rerank_score,
                "sub_scores": dict(getattr(c, "sub_scores", {}) or {}),
                "reserved_slot": bool(getattr(c, "reserved_slot", False)),
                "content": c.content,
            }
            for c in cands
        ],
    }


def zip_submission(json_path) -> str:
    """Nén bài nộp theo yêu cầu BTC: ZIP chứa ĐÚNG MỘT file .json, không nằm trong
    thư mục con. Trả về đường dẫn file zip (cùng tên, đuôi .zip, cạnh file json)."""
    import zipfile
    from pathlib import Path

    p = Path(json_path)
    zp = p.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(p, arcname=p.name)
    return str(zp)


def write_submission(results: List[PipelineResult], out_path: str) -> List[Dict[str, Any]]:
    entries = [result_to_submission_entry(r) for r in results]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    return entries
