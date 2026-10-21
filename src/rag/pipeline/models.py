"""Shared data structures passed between retrieval pipeline stages.

Mirrors the role `Chunk` plays for the chunker package: one plain dataclass
that every stage reads/writes, so stages stay decoupled from each other's
internals.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Candidate:
    """A retrieved chunk as it flows through retrieval -> fusion -> rerank.

    `content` must stay byte-identical to the chunk text stored at ingestion
    time (the submission format requires verbatim `chunk_text`), so stages
    must never rewrite it - they may only attach scores/ranks/metadata.
    """
    chunk_id: str
    doc_id: Any
    content: str
    language: str
    domain: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    heading_path: str = ""
    char_length: int = 0

    # Populated progressively by later stages.
    branch: Optional[str] = None          # which language branch produced this candidate
    retrieval_score: Optional[float] = None
    retrieval_rank: Optional[int] = None  # 1-based rank within its branch
    fusion_score: Optional[float] = None
    rerank_score: Optional[float] = None  # điểm reranker theo CÂU GỐC
    # Query phân rã: điểm reranker theo từng sub-query đã tìm ra chunk này,
    # khoá "sub1".."subN" theo thứ tự unit. Lưu riêng, không đè rerank_score.
    sub_scores: Dict[str, float] = field(default_factory=dict)
    # True nếu chunk vào bài nộp nhờ giữ chỗ cho sub-query.
    reserved_slot: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryItem:
    """A single query as submitted for retrieval."""
    id: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteDecision:
    """Output of the routing stage: which branches to run and with what top_k.

    The current StaticRouter always returns the same fixed decision; a future
    learned router slots in here without the rest of the pipeline changing.
    """
    branch_top_k: Dict[str, int]  # e.g. {"vi": 35, "en": 20, "zh": 10}
    params: Dict[str, Any] = field(default_factory=dict)
