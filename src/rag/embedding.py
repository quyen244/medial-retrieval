"""BGE-M3 embedder: dense vector, and optionally the lexical sparse vector,
from a single forward pass.

sentence-transformers' SentenceTransformer("BAAI/bge-m3") only exposes the
dense embedding. Getting the sparse (lexical) vector from the same model
requires FlagEmbedding's BGEM3FlagModel with return_sparse=True, which
returns `lexical_weights`: a list of {token_id(str): weight} dicts, one per
input text. This module wraps that so the rest of the codebase talks to a
small, stable interface instead of FlagEmbedding's raw output shape.

Computing the sparse vector at index time (instead of skipping it) is nearly
free here: it's a small linear head over hidden states the forward pass
already computes for the dense vector. Skipping it now would mean re-running
the entire GPU pass over the corpus later if hybrid retrieval turns out to
help - exactly the cost we want to avoid paying twice.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from qdrant_client import models


class BgeM3Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        use_fp16: bool = True,
        max_length: int = 1024,
    ):
        from FlagEmbedding import BGEM3FlagModel
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device.startswith("cuda")
        self.model = BGEM3FlagModel(model_name, use_fp16=self.use_fp16, devices=[self.device])
        # Gọi từ nhiều luồng (nhiều query song song) phải lần lượt: tokenizer không
        # an toàn đa luồng.
        import threading
        self._lock = threading.Lock()
        self.max_length = max_length

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize_embeddings: bool = True,  # bge-m3 dense_vecs are already ~unit-norm; kept for API compat
        batch_size: int = 4,
        **kwargs,
    ) -> np.ndarray:
        """Dense-only encode, API-compatible with the subset of
        sentence_transformers.SentenceTransformer.encode() this codebase uses
        (accepts str or List[str], returns one vector or a 2D array)."""
        single = isinstance(texts, str)
        texts_list = [texts] if single else list(texts)
        with self._lock:
            out = self.model.encode(
                texts_list, return_dense=True, return_sparse=False, return_colbert_vecs=False,
                max_length=self.max_length, batch_size=batch_size,
            )
        dense = out["dense_vecs"]
        return dense[0] if single else dense

    def encode_with_sparse(
        self, texts: List[str], batch_size: int = 4
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """Dense + sparse (lexical_weights) from one forward pass."""
        with self._lock:
            out = self.model.encode(
                list(texts), return_dense=True, return_sparse=True, return_colbert_vecs=False,
                max_length=self.max_length, batch_size=batch_size,
            )
        return out["dense_vecs"], out["lexical_weights"]

    @staticmethod
    def lexical_weights_to_sparse_vector(lexical_weights: Dict[str, float]) -> models.SparseVector:
        """Convert one {token_id_str: weight} dict into a Qdrant SparseVector.
        IDF weighting is applied server-side (collection's sparse vector
        `modifier=IDF`), not here - baking IDF in at index time would go
        stale as more documents are added to this incrementally-growing
        corpus."""
        if not lexical_weights:
            return models.SparseVector(indices=[], values=[])
        indices = [int(tok_id) for tok_id in lexical_weights.keys()]
        values = [float(w) for w in lexical_weights.values()]
        return models.SparseVector(indices=indices, values=values)
