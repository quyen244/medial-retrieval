"""Per-language retrieval branches: a filtered vector search against the
shared Qdrant collection, one branch per language (vi/en/zh), run against a
`language` payload filter on a single collection.

QdrantLanguageRetriever (dense-only) is the DEFAULT used by the baseline
pipeline. HybridQdrantLanguageRetriever (dense+sparse, server-side RRF) is
registered alongside it behind the same Retriever interface so it is a
drop-in swap (PipelineConfig.retriever="qdrant-hybrid") once there is a dev
set to measure whether hybrid actually helps - it is not the default.
"""
from typing import Any, Dict, List

from src.rag.pipeline.base import Retriever
from src.rag.pipeline.models import Candidate, QueryItem
from src.rag.pipeline.registry import ComponentRegistry
from src.rag.vectorstore import VectorStore


def _points_to_candidates(points, language: str) -> List[Candidate]:
    candidates: List[Candidate] = []
    for rank, p in enumerate(points, start=1):
        payload = p.payload or {}
        candidates.append(
            Candidate(
                chunk_id=payload.get("chunk_id", str(p.id)),
                doc_id=payload.get("doc_id"),
                content=payload.get("content", ""),
                language=payload.get("language", language),
                domain=payload.get("domain"),
                url=payload.get("url"),
                title=payload.get("title"),
                heading_path=payload.get("heading_path", ""),
                char_length=payload.get("char_length", len(payload.get("content", ""))),
                branch=language,
                retrieval_score=p.score,
                retrieval_rank=rank,
            )
        )
    return candidates


@ComponentRegistry.register("retriever", "qdrant-language")
class QdrantLanguageRetriever(Retriever):
    """DEFAULT: dense-only. Embeds the query once and searches the shared
    collection filtered to a single `language` payload value."""

    def __init__(self, vectorstore: VectorStore, language: str):
        self.vectorstore = vectorstore
        self.language = language

    def retrieve(self, query: QueryItem, top_k: int) -> List[Candidate]:
        # Pipeline encode sẵn một lần cho cả 3 nhánh (metadata["dense_vec"]); gọi
        # lẻ thì tự encode. Cùng vector nên kết quả y hệt, chỉ bớt 2/3 số lần encode.
        query_vector = (query.metadata or {}).get("dense_vec")
        if query_vector is None:
            query_vector = self.vectorstore.encoder.encode(query.text, normalize_embeddings=True).tolist()
        points = self.vectorstore.search(query_vector, top_k=top_k, language=self.language)
        return _points_to_candidates(points, self.language)

    def get_params(self) -> Dict[str, Any]:
        return {"type": "qdrant-language", "language": self.language, "collection": self.vectorstore.collection_name}


@ComponentRegistry.register("retriever", "qdrant-hybrid")
class HybridQdrantLanguageRetriever(Retriever):
    """NOT the default. Dense + sparse hybrid search per language branch,
    fused server-side via Qdrant RRF (VectorStore.hybrid_search). Requires
    the vectorstore's encoder to support encode_with_sparse() (BgeM3Embedder)
    and the collection to have been ingested with sparse vectors populated."""

    def __init__(self, vectorstore: VectorStore, language: str):
        self.vectorstore = vectorstore
        self.language = language

    def retrieve(self, query: QueryItem, top_k: int) -> List[Candidate]:
        points = self.vectorstore.hybrid_search(query.text, top_k=top_k, language=self.language)
        return _points_to_candidates(points, self.language)

    def get_params(self) -> Dict[str, Any]:
        return {"type": "qdrant-hybrid", "language": self.language, "collection": self.vectorstore.collection_name}
