"""Reranking stage: cross-encoder over (query, chunk_content) pairs.

Default model: BAAI/bge-reranker-v2-m3 - multilingual (handles a Vietnamese
query against Chinese/English chunk text in the same pass), open-weights,
complies with the competition's model rules, and is the natural counterpart
to the bge-m3 embedder already used for retrieval.

`models/medical-reranker` was inspected and rejected: it is a
cross-encoder/ms-marco-MiniLM-L6-v2 finetune trained on 3 example pairs
(see its model card - "Training Dataset: Size: 3 training samples"), it is
English-only (MiniLM base), and the fine-tune is not meaningful at that
sample size. It is unsuitable for this multilingual task.
"""
from typing import Any, Dict, List

from sentence_transformers import CrossEncoder

from src.rag.pipeline.base import Reranker
from src.rag.pipeline.models import Candidate, QueryItem
from src.rag.pipeline.registry import ComponentRegistry

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@ComponentRegistry.register("reranker", "cross-encoder")
class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, device: str = None, max_length: int = 1024,
                 fp16: bool = True):
        import torch

        self.model_name = model_name
        self.max_length = max_length
        # fp16 trên GPU có Tensor Core (RTX 5060 Ti): nhanh hơn fp32 nhiều, điểm
        # chỉ lệch ở chữ số nhỏ. CPU thì giữ fp32.
        self.fp16 = fp16 and torch.cuda.is_available()
        kwargs = {"model_kwargs": {"torch_dtype": torch.float16}} if self.fp16 else {}
        self.model = CrossEncoder(model_name, device=device, max_length=max_length, **kwargs)

    def rerank(self, query: QueryItem, candidates: List[Candidate], top_k: int) -> List[Candidate]:
        if not candidates:
            return []
        pairs = [(query.text, c.content) for c in candidates]
        scores = self.model.predict(pairs, batch_size=32)
        for c, score in zip(candidates, scores):
            c.rerank_score = float(score)
        ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
        return ranked[:top_k]

    def get_params(self) -> Dict[str, Any]:
        return {"type": "cross-encoder", "model_name": self.model_name, "max_length": self.max_length,
                "fp16": self.fp16}
