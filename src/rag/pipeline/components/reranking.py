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
        # Tokenizer "fast" của HF không an toàn khi nhiều luồng cùng gọi
        # ("Already borrowed"); khoá lại để nhiều query chạy song song được.
        import threading
        self._lock = threading.Lock()

    def score_pairs(self, pairs: List[tuple]) -> List[float]:
        """Chấm một lô cặp (câu hỏi, đoạn văn) bất kỳ trong MỘT lần gọi model, kể
        cả các cặp thuộc câu hỏi khác nhau - điểm từng cặp độc lập với nhau."""
        if not pairs:
            return []
        # Xếp theo độ dài rồi mới chia lô: mỗi lô bị độn tới cặp dài nhất của
        # nó, text doc dài từ vài chục tới 1.500 ký tự nên độn lẫn lộn là
        # phí GPU. Điểm từng cặp không đổi (có attention mask), trả đúng thứ tự cũ.
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        with self._lock:
            scores = self.model.predict([pairs[i] for i in order], batch_size=32)
        out = [0.0] * len(pairs)
        for i, s in zip(order, scores):
            out[i] = float(s)
        return out

    def rerank(self, query: QueryItem, candidates: List[Candidate], top_k: int) -> List[Candidate]:
        if not candidates:
            return []
        pairs = [(query.text, c.content) for c in candidates]
        scores = self.score_pairs(pairs)
        for c, score in zip(candidates, scores):
            c.rerank_score = float(score)
        ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
        return ranked[:top_k]

    def get_params(self) -> Dict[str, Any]:
        return {"type": "cross-encoder", "model_name": self.model_name, "max_length": self.max_length,
                "fp16": self.fp16}
