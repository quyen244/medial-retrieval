"""Abstract base classes for retrieval pipeline stages.

Follows the same idiom as src/rag/chunker/{base,registry,dispatcher}.py:
one small ABC per stage role, a registry stages register into, and a
dispatcher-like orchestrator (pipeline.py) that runs stages in order.

Each ABC is intentionally narrow (Interface Segregation) so a stage can be
swapped independently of the others - this is the "easy to swap/add
components" requirement. One seat is reserved for future work and must stay
drop-in:

  - QueryRouter: seat for the future router model that will decide which
    language branches / strategies to use per query. StaticRouter is the
    baseline (fixed vi/en/zh top_k).

(A query-preprocessing seat for spelling correction was considered and
dropped - not part of this baseline.)
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.rag.pipeline.models import Candidate, QueryItem, RouteDecision


class QueryRouter(ABC):
    """Decides which language branches / strategies to run for a query, and
    with what top_k each.

    SEAT FOR: the future router model between query input and retrieval.
    Register it under e.g. "learned-router" and select it via
    PipelineConfig.router - the retrieval stage only ever consumes a
    RouteDecision, so it does not need to know whether it came from the
    StaticRouter or a model.
    """

    @abstractmethod
    def route(self, query: QueryItem) -> RouteDecision:
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        return {}


class Retriever(ABC):
    """A single retrieval branch (e.g. one language) over the vector store."""

    @abstractmethod
    def retrieve(self, query: QueryItem, top_k: int) -> List[Candidate]:
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        return {}


class FusionStrategy(ABC):
    """Merges candidate lists from multiple retrieval branches into one
    ranked list. Default implementation is Reciprocal Rank Fusion (RRF) -
    see components/fusion.py for the rationale."""

    @abstractmethod
    def fuse(self, branch_results: Dict[str, List[Candidate]], top_k: int) -> List[Candidate]:
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        return {}


class Reranker(ABC):
    """Re-scores/re-orders fused candidates using a cross-encoder (query, chunk)."""

    @abstractmethod
    def rerank(self, query: QueryItem, candidates: List[Candidate], top_k: int) -> List[Candidate]:
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        return {}
