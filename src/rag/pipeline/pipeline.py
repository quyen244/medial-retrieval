"""The retrieval pipeline orchestrator.

    query -> [router]                  seat for future router model
          -> per-language retrievers   (parallel conceptually; run sequentially here)
          -> [fusion]                  default RRF
          -> [reranker]                cross-encoder
          -> top_k Candidates

Every stage is resolved from PipelineConfig (plain strings/params) via
ComponentRegistry, so swapping a stage means changing the config, not this
file. The full resolved config is captured as `self.config_dict()` for
mlflow logging, and every run produces one Trace (see tracer.py).
"""
import concurrent.futures as cf
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.rag.pipeline.base import FusionStrategy, QueryRouter, Reranker, Retriever
from src.rag.pipeline.models import Candidate, QueryItem
from src.rag.pipeline.registry import ComponentRegistry
from src.rag.pipeline.tracer import Trace
from src.rag.vectorstore import VectorStore

# Import components so their @ComponentRegistry.register decorators run.
import src.rag.pipeline.components  # noqa: F401


@dataclass
class PipelineConfig:
    """Declarative pipeline configuration - this whole object is what gets
    logged to mlflow so a run's exact pipeline is reconstructable."""

    router: str = "static"
    retriever: str = "qdrant-language"  # DEFAULT dense-only; "qdrant-hybrid" for dense+sparse RRF (not default - see components/retrieval.py)
    fusion: str = "rrf"
    reranker: str = "cross-encoder"
    fusion_top_k: int = 50
    rerank_top_k: int = 20
    languages: List[str] = field(default_factory=lambda: ["vi", "en", "zh"])
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    fusion_rrf_k: int = 60
    branch_top_k: Optional[Dict[str, int]] = None  # None -> router's own default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "router": self.router,
            "retriever": self.retriever,
            "fusion": self.fusion,
            "reranker": self.reranker,
            "fusion_top_k": self.fusion_top_k,
            "rerank_top_k": self.rerank_top_k,
            "languages": list(self.languages),
            "reranker_model": self.reranker_model,
            "fusion_rrf_k": self.fusion_rrf_k,
            "branch_top_k": self.branch_top_k,
        }


class RetrievalPipeline:
    def __init__(self, vectorstore: VectorStore, config: PipelineConfig):
        self.vectorstore = vectorstore
        self.config = config

        self.router: QueryRouter = self._build(
            "router", config.router,
            branch_top_k=config.branch_top_k,
        ) if config.branch_top_k else self._build("router", config.router)
        self.fusion: FusionStrategy = self._build("fusion", config.fusion, k=config.fusion_rrf_k) \
            if config.fusion == "rrf" else self._build("fusion", config.fusion)
        self.reranker: Reranker = self._build("reranker", config.reranker, model_name=config.reranker_model)

        self.retrievers: Dict[str, Retriever] = {
            lang: self._build("retriever", config.retriever, vectorstore=vectorstore, language=lang)
            for lang in config.languages
        }

    @staticmethod
    def _build(category: str, key: str, **kwargs):
        target = ComponentRegistry.get(category, key)
        if target is None:
            raise RuntimeError(
                f"No component registered for category='{category}' key='{key}'. "
                f"Available: {ComponentRegistry.list_registered(category)}"
            )
        if isinstance(target, type):
            return target(**kwargs)
        if callable(target):
            return target(**kwargs) if kwargs else target()
        return target  # already an instance

    def component_params(self) -> Dict[str, Any]:
        """Full resolved per-stage params, for mlflow / reproducibility."""
        return {
            "config": self.config.to_dict(),
            "router": self.router.get_params(),
            "retrievers": {lang: r.get_params() for lang, r in self.retrievers.items()},
            "fusion": self.fusion.get_params(),
            "reranker": self.reranker.get_params(),
        }

    def run(self, query: QueryItem, parallel_retrieval: bool = True) -> "PipelineResult":
        trace = Trace(query_id=query.id, query_text=query.text)

        with trace.span("route", params=self.router.get_params(), n_in=1) as span:
            decision = self.router.route(query)
            span.n_out = len(decision.branch_top_k)

        branch_results: Dict[str, List[Candidate]] = {}

        def run_branch(lang: str, top_k: int):
            return lang, self.retrievers[lang].retrieve(query, top_k)

        active_branches = {lang: k for lang, k in decision.branch_top_k.items() if lang in self.retrievers}

        if parallel_retrieval and len(active_branches) > 1:
            with trace.span("retrieve.all", params={"branches": active_branches}, n_in=1) as span:
                with cf.ThreadPoolExecutor(max_workers=len(active_branches)) as ex:
                    futures = [ex.submit(run_branch, lang, k) for lang, k in active_branches.items()]
                    for fut in cf.as_completed(futures):
                        lang, results = fut.result()
                        branch_results[lang] = results
                span.n_out = sum(len(v) for v in branch_results.values())
        else:
            for lang, k in active_branches.items():
                with trace.span(f"retrieve.{lang}", params={"top_k": k}, n_in=1) as span:
                    _, results = run_branch(lang, k)
                    branch_results[lang] = results
                    span.n_out = len(results)

        n_in_fusion = sum(len(v) for v in branch_results.values())
        with trace.span("fusion", params=self.fusion.get_params(), n_in=n_in_fusion) as span:
            fused = self.fusion.fuse(branch_results, top_k=self.config.fusion_top_k)
            span.n_out = len(fused)

        with trace.span("rerank", params=self.reranker.get_params(), n_in=len(fused)) as span:
            reranked = self.reranker.rerank(query, fused, top_k=self.config.rerank_top_k)
            span.n_out = len(reranked)

        trace.finish()
        return PipelineResult(
            query=query,
            branch_results=branch_results,
            fused=fused,
            reranked=reranked,
            trace=trace,
        )


@dataclass
class PipelineResult:
    query: QueryItem
    branch_results: Dict[str, List[Candidate]]
    fused: List[Candidate]
    reranked: List[Candidate]
    trace: Trace
