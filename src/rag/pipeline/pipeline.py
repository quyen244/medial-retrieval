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
from typing import Any, Dict, List, Optional, Tuple

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
    # Tính cho CẢ cặp (câu hỏi + chunk). 512 cắt mất phần lớn chunk khi câu hỏi dài;
    # bge-reranker-v2-m3 hỗ trợ tới 8192, đổi lại chậm hơn.
    reranker_max_length: int = 1024
    reranker_fp16: bool = True
    fusion_rrf_k: int = 60
    branch_top_k: Optional[Dict[str, int]] = None  # None -> router's own default
    # Cách chọn ứng viên sau khi các nhánh ngôn ngữ trả về:
    # - "rrf": gộp các nhánh bằng RRF, giữ fusion_top_k, rerank, lấy rerank_top_k.
    # - "branch-rerank" (mặc định): router lấy vi 35 / en 20 / zh 10, rerank từng
    #   nhánh, giữ branch_keep[lang] tốt nhất của mỗi nhánh (ghép lại thành 50),
    #   xếp chung theo điểm rerank, lấy rerank_top_k.
    #   Quota sống tới tận bước rerank, nên chunk en/zh luôn được reranker chấm
    #   thay vì bị RRF cắt mất trước đó.
    selection: str = "branch-rerank"
    branch_keep: Dict[str, int] = field(default_factory=lambda: {"vi": 26, "en": 16, "zh": 8})

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
            "reranker_max_length": self.reranker_max_length,
            "reranker_fp16": self.reranker_fp16,
            "fusion_rrf_k": self.fusion_rrf_k,
            "branch_top_k": self.branch_top_k,
            "selection": self.selection,
            "branch_keep": dict(self.branch_keep),
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
        self.reranker: Reranker = self._build("reranker", config.reranker, model_name=config.reranker_model,
                                              max_length=config.reranker_max_length,
                                              fp16=config.reranker_fp16)

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

    def retrieve_and_fuse(self, query: QueryItem, trace: Trace,
                          parallel_retrieval: bool = True,
                          span_prefix: str = "") -> Tuple[Dict[str, List[Candidate]], List[Candidate]]:
        """Mọi thứ TRƯỚC khi rerank: route -> các nhánh ngôn ngữ -> fusion.

        Tách khỏi run() để chạy được nhiều lượt truy xuất cho cùng một câu hỏi
        (xem multiquery.py: mỗi sub-query một lượt) rồi mới rerank một lần duy
        nhất. run() vẫn gọi đúng hàm này nên luồng một lượt không đổi hành vi.
        """
        p = f"{span_prefix}." if span_prefix else ""

        with trace.span(f"{p}route", params=self.router.get_params(), n_in=1) as span:
            decision = self.router.route(query)
            span.n_out = len(decision.branch_top_k)

        branch_results: Dict[str, List[Candidate]] = {}

        if self.config.retriever == "qdrant-hybrid" and hasattr(self.vectorstore, "prime_hybrid"):
            self.vectorstore.prime_hybrid([query.text])  # encode 1 lần cho cả 3 nhánh
        if self.config.retriever == "qdrant-language" and "dense_vec" not in (query.metadata or {}):
            with trace.span(f"{p}encode", params={}, n_in=1) as span:
                vec = self.vectorstore.encoder.encode(query.text, normalize_embeddings=True).tolist()
                span.n_out = 1
            query = QueryItem(id=query.id, text=query.text,
                              metadata={**(query.metadata or {}), "dense_vec": vec})

        def run_branch(lang: str, top_k: int):
            return lang, self.retrievers[lang].retrieve(query, top_k)

        active_branches = {lang: k for lang, k in decision.branch_top_k.items() if lang in self.retrievers}

        if parallel_retrieval and len(active_branches) > 1:
            with trace.span(f"{p}retrieve.all", params={"branches": active_branches}, n_in=1) as span:
                with cf.ThreadPoolExecutor(max_workers=len(active_branches)) as ex:
                    futures = [ex.submit(run_branch, lang, k) for lang, k in active_branches.items()]
                    for fut in cf.as_completed(futures):
                        lang, results = fut.result()
                        branch_results[lang] = results
                span.n_out = sum(len(v) for v in branch_results.values())
        else:
            for lang, k in active_branches.items():
                with trace.span(f"{p}retrieve.{lang}", params={"top_k": k}, n_in=1) as span:
                    _, results = run_branch(lang, k)
                    branch_results[lang] = results
                    span.n_out = len(results)

        n_in_fusion = sum(len(v) for v in branch_results.values())
        with trace.span(f"{p}fusion", params=self.fusion.get_params(), n_in=n_in_fusion) as span:
            fused = self.fusion.fuse(branch_results, top_k=self.config.fusion_top_k)
            span.n_out = len(fused)

        return branch_results, fused

    def rerank_by_branch(self, query: QueryItem, branch_results: Dict[str, List[Candidate]],
                         trace: Trace) -> Tuple[List[Candidate], List[Candidate]]:
        """selection="branch-rerank": chấm mọi ứng viên của mọi nhánh, giữ
        branch_keep[lang] tốt nhất mỗi nhánh, rồi xếp chung theo điểm rerank.

        Chấm một lượt trên cả tập thay vì ba lượt riêng: cùng model, cùng câu hỏi
        nên điểm y hệt nhau, chỉ ít lần gọi GPU hơn. Cũng vì điểm cùng thước đo
        nên bước "rerank lại 50 ứng viên" chính là sắp xếp theo điểm đã có - chấm
        lại lần hai bằng cùng model sẽ ra đúng các con số đó.

        Trả (50 ứng viên đã chọn, top rerank_top_k)."""
        seen: Dict[str, Candidate] = {}
        for lang, cands in branch_results.items():
            for c in cands:
                seen.setdefault(str(c.chunk_id), c)
        pool = list(seen.values())
        with trace.span("rerank.branches", params=self.reranker.get_params(), n_in=len(pool)) as span:
            scored = self.reranker.rerank(query, pool, top_k=len(pool))
            kept: List[Candidate] = []
            for lang, keep in self.config.branch_keep.items():
                kept += [c for c in scored if c.branch == lang][:keep]
            kept.sort(key=lambda c: c.rerank_score, reverse=True)
            span.n_out = len(kept)
        return kept, kept[: self.config.rerank_top_k]

    def run(self, query: QueryItem, parallel_retrieval: bool = True) -> "PipelineResult":
        trace = Trace(query_id=query.id, query_text=query.text)
        branch_results, fused = self.retrieve_and_fuse(query, trace, parallel_retrieval)

        if self.config.selection == "branch-rerank":
            fused, reranked = self.rerank_by_branch(query, branch_results, trace)
        else:
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
