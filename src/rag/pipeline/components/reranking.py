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
# Câu lệnh tác vụ cho Qwen3-Reranker; model card khuyên viết bằng tiếng Anh và theo đúng tác vụ.
QWEN3_INSTRUCTION = ("Given a medical question written in Vietnamese, judge whether the document, "
                     "which may be in Vietnamese, English or Chinese, is relevant to the question")


def make_reranker(model_name: str = DEFAULT_RERANKER_MODEL, max_length: int = 1024, fp16: bool = True,
                  engine: str = "hf"):
    """Chọn lớp theo tên model: họ Qwen3-Reranker là LM nhân quả, còn lại là cross-encoder.
    engine="vllm": Qwen3-Reranker (cùng template, cùng điểm P(yes)) hoặc cross-encoder (cùng token như HF)."""
    if "qwen3-reranker" in model_name.lower() and engine == "vllm":
        return VLLMQwen3Reranker(model_name=model_name, max_length=max_length)
    if "qwen3-reranker" in model_name.lower():
        return Qwen3Reranker(model_name=model_name, max_length=max_length, fp16=fp16)
    if engine == "vllm":
        return VLLMCrossEncoderReranker(model_name=model_name, max_length=max_length)
    return CrossEncoderReranker(model_name=model_name, max_length=max_length, fp16=fp16)


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


class VLLMCrossEncoderReranker(CrossEncoderReranker):
    """Cross-encoder (bge-reranker-v2-m3) chạy trên vLLM runner pooling. Token hoá cặp bằng chính tokenizer HF
    (truncation=True, max_length) như CrossEncoder.predict, rồi đưa id vào classify, nên điểm phải khớp lớp HF
    tới sai số fp. vLLM không độn lô và gom lô liên tục: lợi khi độ dài cặp lệch nhau nhiều."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, max_length: int = 1024, gpu_mem: float = 0.3):
        import threading

        from transformers import AutoTokenizer
        from vllm import LLM

        self.model_name, self.max_length, self.fp16 = model_name, max_length, True
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(model=model_name, runner="pooling", max_model_len=max_length, dtype="float16",
                       gpu_memory_utilization=gpu_mem)
        self._lock = threading.Lock()

    def score_pairs(self, pairs: List[tuple]) -> List[float]:
        if not pairs:
            return []
        from vllm.inputs import TokensPrompt

        with self._lock:
            ids = self.tokenizer([q for q, _ in pairs], [d for _, d in pairs], truncation=True,
                                 max_length=self.max_length)["input_ids"]
            outs = self.llm.classify([TokensPrompt(prompt_token_ids=x) for x in ids], use_tqdm=False)
        return [float(o.outputs.probs[0]) for o in outs]

    def get_params(self) -> Dict[str, Any]:
        return {"type": "cross-encoder-vllm", "model_name": self.model_name, "max_length": self.max_length}


@ComponentRegistry.register("reranker", "qwen3")
class Qwen3Reranker(CrossEncoderReranker):
    """Qwen3-Reranker (Qwen3 Embedding, arXiv 2506.05176): LM nhân quả đọc
    (instruction, query, document) theo template chat rồi trả lời "yes"/"no".
    Điểm = P(yes) / (P(yes) + P(no)) ở token cuối, thang [0, 1] như bge sau sigmoid.
    Template và cách tính theo đúng model card của Qwen."""

    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
              "the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
              "<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-4B", device: str = None, max_length: int = 1024,
                 fp16: bool = True, instruction: str = QWEN3_INSTRUCTION, batch_size: int = 8):
        import threading

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.instruction = instruction
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and self.device.startswith("cuda")
        # Qwen3 train ở bf16; fp16 dễ tràn số. GPU không có bf16 (Turing trở về trước) mới dùng fp16.
        half = torch.bfloat16 if self.fp16 and torch.cuda.is_bf16_supported() else torch.float16
        # padding trái: token cuối của mọi dòng trong lô đều là vị trí sinh câu trả lời.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=half if self.fp16 else torch.float32).to(self.device).eval()
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_ids = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)
        self._lock = threading.Lock()

    def _encode(self, query: str, doc: str) -> List[int]:
        """max_length tính cho phần instruction + query + doc như bge. Mã hoá cả chuỗi một lần
        (tách đôi thì BPE ráp khác ở chỗ nối) rồi cắt đuôi, tức cắt cuối doc."""
        ids = self.tokenizer.encode(f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}",
                                    add_special_tokens=False)[: self.max_length]
        return self.prefix_ids + ids + self.suffix_ids

    def score_pairs(self, pairs: List[tuple]) -> List[float]:
        if not pairs:
            return []
        import torch

        ids = [self._encode(q, d) for q, d in pairs]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        out = [0.0] * len(ids)
        with self._lock, torch.no_grad():
            for b in range(0, len(order), self.batch_size):
                idx = order[b: b + self.batch_size]
                batch = self.tokenizer.pad({"input_ids": [ids[i] for i in idx]}, padding=True,
                                           return_tensors="pt").to(self.device)
                # logits_to_keep=1: chỉ chiếu lm_head (151k từ) ở token cuối, không phải mọi vị trí
                last = self.model(**batch, logits_to_keep=1).logits[:, -1, :]
                two = torch.stack([last[:, self.no_id], last[:, self.yes_id]], dim=1).float()
                for i, p in zip(idx, torch.softmax(two, dim=1)[:, 1].tolist()):
                    out[i] = float(p)
        return out

    def get_params(self) -> Dict[str, Any]:
        return {"type": "qwen3", "model_name": self.model_name, "max_length": self.max_length,
                "fp16": self.fp16, "instruction": self.instruction}


class VLLMQwen3Reranker(Qwen3Reranker):
    """Qwen3-Reranker chạy trên vLLM (runner pooling, đổi lm_head thành bộ phân loại yes-no như model card
    vLLM hướng dẫn). Chuỗi token giữ nguyên _encode() của lớp HF, nên điểm phải khớp lớp HF tới sai số fp."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-4B", max_length: int = 1024,
                 instruction: str = QWEN3_INSTRUCTION, gpu_mem: float = 0.6):
        import threading

        from transformers import AutoTokenizer
        from vllm import LLM

        self.model_name, self.max_length, self.instruction, self.fp16 = model_name, max_length, instruction, True
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prefix_ids = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)
        self.llm = LLM(model=model_name, runner="pooling", max_model_len=max_length + 256,
                       gpu_memory_utilization=gpu_mem, enable_prefix_caching=True,
                       hf_overrides={"architectures": ["Qwen3ForSequenceClassification"],
                                     "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": True})
        self._lock = threading.Lock()

    def score_pairs(self, pairs: List[tuple]) -> List[float]:
        if not pairs:
            return []
        from vllm.inputs import TokensPrompt

        prompts = [TokensPrompt(prompt_token_ids=self._encode(q, d)) for q, d in pairs]
        with self._lock:
            outs = self.llm.classify(prompts, use_tqdm=False)
        return [float(o.outputs.probs[0]) for o in outs]

    def get_params(self) -> Dict[str, Any]:
        return {"type": "qwen3-vllm", "model_name": self.model_name, "max_length": self.max_length,
                "instruction": self.instruction}
