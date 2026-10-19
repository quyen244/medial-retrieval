"""Fusion stage: merges per-language candidate lists into one ranked list.

Default: Reciprocal Rank Fusion (RRF).

Rationale: the three branches search the same Qdrant collection with the
same cosine metric, but cosine similarity is still only comparable *within*
a query/branch, not *across* branches - a vi chunk scoring 0.81 is not
reliably "more relevant" than a zh chunk scoring 0.76, because the score
distribution shifts with corpus density and language (the zh branch has far
fewer candidates to begin with, per the known 4% crawl coverage). Rank-based
fusion (RRF) sidesteps this miscalibration entirely: it only uses each
candidate's rank within its own branch, so no cross-branch score
normalization is needed. This is the standard reason RRF is preferred for
merging heterogeneous retrievers/indices.

An alternative (WeightedScoreFusion) is included to show the interface
supports swapping fusion strategies without touching pipeline code.
"""
from typing import Any, Dict, List, Optional

from src.rag.pipeline.base import FusionStrategy
from src.rag.pipeline.models import Candidate
from src.rag.pipeline.registry import ComponentRegistry


@ComponentRegistry.register("fusion", "rrf")
class ReciprocalRankFusion(FusionStrategy):
    """score(c) = sum over branches containing c of 1 / (k + rank_in_branch)."""

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, branch_results: Dict[str, List[Candidate]], top_k: int) -> List[Candidate]:
        merged: Dict[str, Candidate] = {}
        scores: Dict[str, float] = {}

        for branch, candidates in branch_results.items():
            for c in candidates:
                key = c.chunk_id
                rrf_contribution = 1.0 / (self.k + (c.retrieval_rank or 1))
                scores[key] = scores.get(key, 0.0) + rrf_contribution
                # Keep the first-seen candidate payload (content/metadata are
                # identical regardless of which branch's copy we keep, since
                # a chunk only ever belongs to one language).
                merged.setdefault(key, c)

        for key, c in merged.items():
            c.fusion_score = scores[key]

        ranked = sorted(merged.values(), key=lambda c: c.fusion_score, reverse=True)
        return ranked[:top_k]

    def get_params(self) -> Dict[str, Any]:
        return {"type": "rrf", "k": self.k}


@ComponentRegistry.register("fusion", "weighted-score")
class WeightedScoreFusion(FusionStrategy):
    """Alternative: per-branch-weighted raw cosine score. Demonstrates that
    fusion strategies are swappable; not the default because cosine scores
    across branches are not calibrated (see module docstring)."""

    def __init__(self, branch_weights: Optional[Dict[str, float]] = None):
        self.branch_weights = branch_weights or {}

    def fuse(self, branch_results: Dict[str, List[Candidate]], top_k: int) -> List[Candidate]:
        merged: Dict[str, Candidate] = {}
        scores: Dict[str, float] = {}

        for branch, candidates in branch_results.items():
            weight = self.branch_weights.get(branch, 1.0)
            for c in candidates:
                key = c.chunk_id
                contribution = weight * (c.retrieval_score or 0.0)
                scores[key] = max(scores.get(key, float("-inf")), contribution)
                merged.setdefault(key, c)

        for key, c in merged.items():
            c.fusion_score = scores[key]

        ranked = sorted(merged.values(), key=lambda c: c.fusion_score, reverse=True)
        return ranked[:top_k]

    def get_params(self) -> Dict[str, Any]:
        return {"type": "weighted-score", "branch_weights": self.branch_weights}
