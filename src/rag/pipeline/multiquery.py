"""Truy xuất nhiều lượt cho một câu hỏi, rồi hợp tập và rerank một lần.

Đọc `augmented.jsonl` do pha augment sinh ra (src/rag/augment). Nhánh ngắn có
một unit nên chạy y như baseline; nhánh dài có 3 sub-query cộng câu gốc, mỗi
unit một lượt truy xuất riêng.

Hai điểm then chốt của thiết kế, cả hai đều dễ làm sai:

1. **Rerank bằng câu GỐC, không phải chuỗi đã augment.** Re-ranker được train
   trên câu người thật viết. Đưa chuỗi ghép đầy thuật ngữ đa ngữ vào là đẩy nó
   ra ngoài phân phối train. Chuỗi augment chỉ để TÌM ứng viên, không để CHẤM.

2. **Hợp tập rồi mới rerank, không rerank từng lượt rồi mới hợp.** Rerank
   từng lượt thì điểm của hai lượt khác nhau không so được với nhau, hợp lại
   là so nhầm. Rerank một lần trên cả tập cho mọi ứng viên cùng một thước đo.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag.pipeline.models import Candidate, QueryItem
from src.rag.pipeline.pipeline import PipelineResult, RetrievalPipeline
from src.rag.pipeline.tracer import Trace


@dataclass
class MultiQueryConfig:
    augmented_file: str = "data/augment/augmented.jsonl"

    # Cắt bớt tập hợp trước khi rerank. None = không cắt. Mặc định không cắt
    # vì cắt theo điểm fusion của các lượt khác nhau là so nhầm thước đo (xem
    # ghi chú 2 ở đầu file) - muốn giảm chi phí rerank thì giảm fusion_top_k.
    max_rerank_candidates: Optional[int] = None

    # Query không có trong augmented.jsonl thì chạy baseline một lượt với câu
    # gốc, chứ không bỏ query: bài nộp thiếu một id là bị loại cả bài.
    fallback_to_baseline: bool = True

    # Chuỗi đưa cho re-ranker. Mặc định là câu GỐC, không phải chuỗi augment,
    # vì dữ liệu train re-ranker (judged.jsonl, trường `query`) là câu hỏi
    # tiếng Việt tự nhiên chưa augment - train bằng câu tự nhiên rồi chạy
    # bằng chuỗi ghép đa ngữ là lệch phân phối train/test.
    #
    # Bật cờ này để đo hướng ngược lại. Lưu ý nó chỉ thật sự nhất quán nếu
    # tập train cũng được augment rồi train lại re-ranker; bật không mà vẫn
    # dùng checkpoint train trên câu tự nhiên thì đang đo một hệ lệch.
    rerank_with_augmented: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "augmented_file": self.augmented_file,
            "max_rerank_candidates": self.max_rerank_candidates,
            "fallback_to_baseline": self.fallback_to_baseline,
            "rerank_with_augmented": self.rerank_with_augmented,
        }


def load_augmented(path: str) -> Dict[int, Dict[str, Any]]:
    """qid -> bản ghi AugmentedQuery đã ghi ra jsonl."""
    out: Dict[int, Dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Không thấy {p}. Chạy `python -m src.rag.augment.run_augment` trước."
        )
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[int(row["qid"])] = row
    return out


def unit_texts(record: Dict[str, Any]) -> List[str]:
    """Chuỗi đem đi truy xuất của từng unit, theo đúng thứ tự đã ghi.

    `augmented` rỗng (unit raw, hoặc augment hỏng) thì tụt về `text`.
    """
    texts: List[str] = []
    for u in record.get("units", []):
        t = (u.get("augmented") or u.get("text") or "").strip()
        if t:
            texts.append(t)
    return texts


def union_candidates(per_unit: List[List[Candidate]]) -> tuple:
    """Hợp các danh sách, khử trùng theo chunk_id.

    Giữ bản gặp đầu tiên: mọi bản sao có cùng `content` nên chọn bản nào cũng
    như nhau với re-ranker, và giữ-bản-đầu cho kết quả cố định giữa các lần
    chạy. Trả thêm số lượt tìm ra mỗi chunk để đo xem phân rã có thực sự mở
    rộng vùng phủ hay ba sub-query chỉ lặp lại nhau.
    """
    merged: Dict[str, Candidate] = {}
    hits: Dict[str, int] = {}
    for candidates in per_unit:
        for c in candidates:
            key = str(c.chunk_id)
            hits[key] = hits.get(key, 0) + 1
            merged.setdefault(key, c)
    return list(merged.values()), hits


@dataclass
class MultiQueryResult:
    """Kết quả một câu hỏi, kèm số liệu riêng của nhánh nhiều truy vấn."""
    result: PipelineResult
    n_unit: int = 0
    n_union: int = 0
    n_overlap: int = 0  # số chunk được từ 2 lượt trở lên tìm ra
    used_fallback: bool = False
    stats: Dict[str, Any] = field(default_factory=dict)


class MultiQueryPipeline:
    """Bọc ngoài RetrievalPipeline. Không thay thế nó: mọi thành phần (router,
    retriever, fusion, reranker) vẫn là của pipeline gốc, chỉ khác ở chỗ phần
    truy xuất chạy nhiều lượt trước khi rerank."""

    def __init__(self, pipeline: RetrievalPipeline,
                 augmented: Dict[int, Dict[str, Any]],
                 config: Optional[MultiQueryConfig] = None):
        self.pipeline = pipeline
        self.augmented = augmented
        self.config = config or MultiQueryConfig()

    def component_params(self) -> Dict[str, Any]:
        params = self.pipeline.component_params()
        params["multiquery"] = self.config.to_dict()
        return params

    def run(self, query: QueryItem, parallel_retrieval: bool = True) -> MultiQueryResult:
        record = self.augmented.get(int(query.id))
        texts = unit_texts(record) if record else []

        if not texts:
            if not self.config.fallback_to_baseline:
                raise KeyError(f"query {query.id} không có trong augmented.jsonl")
            return MultiQueryResult(
                result=self.pipeline.run(query, parallel_retrieval),
                n_unit=1, used_fallback=True,
            )

        trace = Trace(query_id=query.id, query_text=query.text)
        per_unit: List[List[Candidate]] = []
        branch_results: Dict[str, List[Candidate]] = {}

        # Các lượt chạy tuần tự: mỗi lượt đã tự chạy song song 3 nhánh ngôn
        # ngữ rồi. Cho cả 4 lượt chạy song song nữa là nhân bốn số truy vấn
        # đồng thời vào Qdrant và số lần encode trên GPU - đáng làm nhưng phải
        # đo trước, không bật mặc định.
        for i, text in enumerate(texts):
            unit_query = QueryItem(id=query.id, text=text, metadata=dict(query.metadata))
            branches, fused = self.pipeline.retrieve_and_fuse(
                unit_query, trace, parallel_retrieval, span_prefix=f"unit{i}"
            )
            per_unit.append(fused)
            # Giữ kết quả nhánh của lượt ĐẦU cho PipelineResult: đó là lượt
            # của câu hỏi chính, các lượt sau là mở rộng.
            if i == 0:
                branch_results = branches

        union, hits = union_candidates(per_unit)
        cap = self.config.max_rerank_candidates
        if cap and len(union) > cap:
            union = union[:cap]

        # Mặc định rerank bằng CÂU GỐC. Khi bật rerank_with_augmented thì dùng
        # chuỗi của unit ĐẦU: unit đầu là câu hỏi chính (nhánh ngắn) hoặc
        # sub-query thứ nhất (nhánh dài), và nó chứa đủ khối thuật ngữ dùng
        # chung. Không ghép cả 4 unit lại: chuỗi sẽ dài gấp bốn và lặp khối
        # thuật ngữ bốn lần.
        rerank_query = query
        if self.config.rerank_with_augmented:
            rerank_query = QueryItem(id=query.id, text=texts[0],
                                     metadata=dict(query.metadata))

        with trace.span("rerank.union", params=self.pipeline.reranker.get_params(),
                        n_in=len(union)) as span:
            reranked = self.pipeline.reranker.rerank(
                rerank_query, union, top_k=self.pipeline.config.rerank_top_k
            )
            span.n_out = len(reranked)

        trace.finish()
        return MultiQueryResult(
            result=PipelineResult(
                query=query,
                branch_results=branch_results,
                fused=union,
                reranked=reranked,
                trace=trace,
            ),
            n_unit=len(texts),
            n_union=len(union),
            n_overlap=sum(1 for v in hits.values() if v > 1),
            stats={
                "n_per_unit": [len(c) for c in per_unit],
                "is_long": bool(record.get("is_long")),
            },
        )
