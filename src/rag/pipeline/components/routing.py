"""Query routing stage: decides which language branches to run and with
what top_k each.

StaticRouter is the baseline (always runs vi/en/zh with the competition's
suggested top_k split). The seat for the future router model is this same
QueryRouter interface: implement .route() to inspect the query and return a
RouteDecision that drops/adds branches or changes top_k per query, register
it, e.g.

    @ComponentRegistry.register("router", "learned-router")
    class LearnedRouter(QueryRouter): ...

then select it via PipelineConfig(router="learned-router", ...). The
retrieval stage only ever consumes a RouteDecision, so it is unaffected.
"""
from typing import Any, Dict, Optional

from src.rag.pipeline.base import QueryRouter
from src.rag.pipeline.models import QueryItem, RouteDecision
from src.rag.pipeline.registry import ComponentRegistry

DEFAULT_BRANCH_TOP_K = {"vi": 50, "en": 30, "zh": 20}


@ComponentRegistry.register("router", "static")
class StaticRouter(QueryRouter):
    """Baseline: always routes to every configured language branch with a
    fixed top_k, regardless of the query."""

    def __init__(self, branch_top_k: Optional[Dict[str, int]] = None):
        self.branch_top_k = dict(branch_top_k or DEFAULT_BRANCH_TOP_K)

    def route(self, query: QueryItem) -> RouteDecision:
        return RouteDecision(branch_top_k=dict(self.branch_top_k), params={"strategy": "static"})

    def get_params(self) -> Dict[str, Any]:
        return {"type": "static", "branch_top_k": self.branch_top_k}
