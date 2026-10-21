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

DEFAULT_BRANCH_TOP_K = {"vi": 35, "en": 20, "zh": 10}


def quota_from_counts(counts: Dict[str, int], total: int, power: float = 0.5) -> Dict[str, int]:
    """Chia `total` ứng viên theo phân bố corpus, mỗi ngôn ngữ nặng count**power.

    power=1: đúng tỉ lệ corpus (ngôn ngữ ít gần như bị loại). power=0.5 (căn
    bậc hai): vẫn theo corpus nhưng nâng ngôn ngữ ít. Làm tròn theo phần dư lớn
    nhất để tổng đúng bằng `total`; mỗi ngôn ngữ có trong corpus được ít nhất 1.
    """
    weights = {lang: max(c, 0) ** power for lang, c in counts.items() if c > 0}
    s = sum(weights.values())
    raw = {lang: total * w / s for lang, w in weights.items()}
    out = {lang: max(1, int(v)) for lang, v in raw.items()}
    rest = total - sum(out.values())
    for lang in sorted(raw, key=lambda l: (-(raw[l] - int(raw[l])), l))[:max(rest, 0)]:
        out[lang] += 1
    return out


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
