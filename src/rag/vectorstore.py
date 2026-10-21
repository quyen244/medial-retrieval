import os
import uuid
from typing import Dict, Any, List, Optional, Union
from qdrant_client import QdrantClient, models
from src.rag.chunker import default_dispatcher

# Fixed namespace so chunk_id -> point_id mapping is stable across processes/runs.
# (uuid5 is deterministic: same chunk_id always yields the same point id, which is
# what makes re-running ingestion idempotent instead of creating duplicate points.)
_POINT_ID_NAMESPACE = uuid.UUID("5b6a1f1e-7e3b-4a8c-9b0a-1c2d3e4f5a6b")


def deterministic_point_id(chunk_id: str) -> str:
    """Derive a stable UUID5 point id from a chunk_id so re-ingesting the same
    chunk overwrites the same point instead of creating a duplicate."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, str(chunk_id)))


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def chunk_to_payload(chunk) -> Dict[str, Any]:
    """Chunk.to_dict() with chunk.metadata flattened into top-level payload
    keys (e.g. {"metadata": {"published_date": "..."}} -> {"published_date": "..."}).

    Flat top-level keys matter for backfill: Qdrant's set_payload() merges at
    the top level only, so a payload shaped {"metadata": {...}} would have
    its *entire* nested dict replaced on every set_payload call, clobbering
    any other metadata field set in a previous backfill pass. Flat keys let
    each future metadata field (date filtering, quality signals, topic
    labels, ...) be backfilled independently without touching the others.
    """
    payload = chunk.to_dict() if hasattr(chunk, "to_dict") else dict(vars(chunk))
    extra = payload.pop("metadata", None) or {}
    payload.update(extra)
    return payload


class VectorStore:
    def __init__(
        self,
        encoder: Any,  # BgeM3Embedder (dense+sparse) or a plain SentenceTransformer (dense-only)
        path: Optional[str] = "data/qdrant_db",
        url: Optional[str] = None,
        collection_name: str = "medical_documents",
    ):
        self.encoder = encoder
        self.collection_name = collection_name

        # Auto-select embedded local mode or remote URL mode.
        if url:
            # Mặc định của client REST là 5 giây: đủ khi Qdrant rảnh, nhưng lúc nó
            # đang tối ưu index hoặc tạo snapshot thì search lọc ngôn ngữ vượt quá.
            self.client = QdrantClient(url=url, timeout=120)
        elif path:
            os.makedirs(path, exist_ok=True)
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")

        print("Qdrant Client initialized:", self.client)

        if not self.client.collection_exists(collection_name=self.collection_name):
            # Named vectors: "dense" (bge-m3 dense embedding) and "sparse"
            # (bge-m3 lexical weights) are both populated at ingest time, from
            # the same forward pass (see src/rag/embedding.py) - computing the
            # sparse vector later would mean re-running the whole GPU pass
            # over the corpus again. The sparse vector uses Qdrant's
            # server-side IDF modifier rather than baking IDF in at index
            # time: IDF depends on the whole collection and would go stale as
            # more documents are added to this incrementally-growing corpus.
            # Retrieval still defaults to dense-only (see pipeline/components/
            # retrieval.py) - hybrid is a switchable option, not the default,
            # until there is a dev set to measure whether it helps.
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(size=1024, distance=models.Distance.COSINE),
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF),
                },
            )

    def ensure_payload_index(self, field_name: str = "language", schema: str = "keyword") -> None:
        """Create a payload index for fast filtered search (e.g. per-language branches).

        Idempotent: Qdrant no-ops / raises harmlessly if the index already exists,
        which we swallow so this is safe to call on every startup.
        """
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema,
            )
        except Exception:
            pass  # already exists

    def upsert_text(
        self,
        text: Union[str, List[str]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        point_id: Optional[Union[str, int, List[Any]]] = None
    ):
        """Upsert a single text or automatically delegate to upsert_texts if a list is passed."""
        if isinstance(text, (list, tuple)):
            metas = metadata if isinstance(metadata, list) else None
            pids = point_id if isinstance(point_id, list) else None
            return self.upsert_texts(list(text), metadatas=metas, ids=pids)

        vector = self.encoder.encode(text, normalize_embeddings=True)

        payload = {"content": text, "text": text}
        if isinstance(metadata, dict):
            payload.update(metadata)

        points = [
            models.PointStruct(
                id=str(point_id) if point_id is not None else str(uuid.uuid4()),
                vector={DENSE_VECTOR_NAME: vector.tolist()},
                payload=payload,
            )
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def upsert_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[Any]] = None, batch_size: int = 128):
        """Upsert a batch of texts/queries efficiently."""
        total = len(texts)
        for i in range(0, total, batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_vectors = self.encoder.encode(batch_texts, normalize_embeddings=True)

            points = []
            for j, (txt, vec) in enumerate(zip(batch_texts, batch_vectors)):
                idx = i + j
                pid = str(ids[idx]) if ids and idx < len(ids) else str(uuid.uuid4())
                payload = {"content": txt, "text": txt}
                if metadatas and idx < len(metadatas):
                    payload.update(metadatas[idx])
                points.append(
                    models.PointStruct(
                        id=pid,
                        vector={DENSE_VECTOR_NAME: vec.tolist()},
                        payload=payload,
                    )
                )
            self.client.upsert(collection_name=self.collection_name, points=points)

    def upsert_chunks(self, chunks: list, batch_size: int = 128, encode_batch_size: Optional[int] = None, wait: bool = True):
        """Receives a list of Chunk objects (e.g. produced by the dispatcher) and
        upserts them with deterministic point ids derived from chunk_id, so
        re-running ingestion over the same chunks is idempotent (no duplicates).

        `batch_size` controls the Qdrant upsert chunk size; `encode_batch_size`
        (if given) is passed through to the embedder's own internal
        sub-batching so its length-sorting has a large pool of texts to
        minimize padding waste - pass the whole `chunks` list in one call
        with a large batch_size to get this benefit.

        If the encoder supports encode_with_sparse() (BgeM3Embedder), both
        the dense and sparse vectors are computed from the same forward pass
        and stored; otherwise (plain SentenceTransformer) only dense is
        stored and the sparse slot is left empty for that point.
        """
        total = len(chunks)
        supports_sparse = hasattr(self.encoder, "encode_with_sparse")
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]

            if supports_sparse:
                dense_vecs, lexical_weights = self.encoder.encode_with_sparse(
                    texts, batch_size=encode_batch_size or batch_size
                )
                points = []
                for dense_vec, lw, c in zip(dense_vecs, lexical_weights, batch):
                    sparse_vec = self.encoder.lexical_weights_to_sparse_vector(lw)
                    points.append(models.PointStruct(
                        id=deterministic_point_id(c.chunk_id),
                        vector={
                            DENSE_VECTOR_NAME: dense_vec.tolist() if hasattr(dense_vec, "tolist") else list(dense_vec),
                            SPARSE_VECTOR_NAME: sparse_vec,
                        },
                        payload=chunk_to_payload(c),
                    ))
            else:
                encode_kwargs = {"normalize_embeddings": True}
                if encode_batch_size:
                    encode_kwargs["batch_size"] = encode_batch_size
                vectors = self.encoder.encode(texts, **encode_kwargs)
                points = [
                    models.PointStruct(
                        id=deterministic_point_id(c.chunk_id),
                        vector={DENSE_VECTOR_NAME: vec.tolist()},
                        payload=chunk_to_payload(c),
                    )
                    for vec, c in zip(vectors, batch)
                ]

            self.client.upsert(collection_name=self.collection_name, points=points, wait=wait)

    def delete_by_doc_ids(self, doc_ids: List[Any]) -> None:
        """Delete every point whose payload.doc_id is in doc_ids.

        Used before re-upserting a *changed* document: deterministic point ids
        alone only guarantee overwrite-in-place for chunk_ids that still
        exist in the new chunking - if the new version of a document produces
        fewer chunks than before, the surplus old points (..._c005, _c006, ...)
        would otherwise survive forever as stale/orphaned content. Deleting by
        doc_id first and then inserting the fresh chunk set guarantees no
        orphans regardless of how the chunk count changed.
        """
        if not doc_ids:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))]
                )
            ),
        )

    def point_ids_by_doc_id(self, doc_id: Any) -> List[Any]:
        """All existing point ids for one doc_id - a payload-backfill lookup
        (filter scroll), not a vector search. Used so a payload-only
        metadata update can target exactly the right points without
        re-chunking the document."""
        ids = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                ),
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.extend(p.id for p in points)
            if offset is None:
                break
        return ids

    def set_payload_fields(self, point_ids: List[Any], fields: Dict[str, Any]) -> None:
        """Patch top-level payload fields on existing points, without
        touching their vectors. `fields` must be flat top-level keys (see
        chunk_to_payload) so this merges cleanly instead of clobbering
        sibling fields set by a previous backfill pass."""
        if not point_ids or not fields:
            return
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=fields,
            points=point_ids,
        )

    def count_missing_field(self, field_name: str) -> int:
        """Count points that do NOT have `field_name` set at all.

        Important correctness note: a Qdrant filter like `field == X` SILENTLY
        excludes points that lack the field entirely, so a half-finished
        backfill does not surface as an error anywhere - it just quietly
        under-retrieves. This is the explicit detection path for that: run it
        after a backfill to confirm 0 remain, or periodically to catch a
        backfill that was interrupted partway through.
        """
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(
                must=[models.IsEmptyCondition(is_empty=models.PayloadField(key=field_name))]
            ),
            exact=True,
        )
        return result.count

    def count_by_doc_id(self, doc_id: Any) -> int:
        """Point count for a single doc_id - used to verify no orphan chunks
        remain after a changed-document re-ingest."""
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            ),
            exact=True,
        )
        return result.count

    def set_bulk_load_mode(self, enabled: bool) -> None:
        """Toggle Qdrant's incremental HNSW indexing during bulk ingestion.

        indexing_threshold=0 disables automatic indexing while bulk-inserting,
        avoiding repeated incremental HNSW rebuilds; call with enabled=False
        afterwards to restore normal indexing and trigger a single build.
        """
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(
                indexing_threshold=0 if enabled else 20000
            ),
        )

    def upsert_doc(self, doc: Dict[str, Any], versioning: bool = False):
        """Convenience helper: receives a raw doc dict, chunks it via the
        default dispatcher, and stores the resulting chunks."""
        if versioning:
            chunks, params = default_dispatcher.chunk(doc, versioning=True)
            self.upsert_chunks(chunks)
            return params
        chunks = default_dispatcher.chunk(doc, versioning=False)
        self.upsert_chunks(chunks)
        return None

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        language: Optional[str] = None,
        query_filter: Optional[models.Filter] = None,
    ):
        """Vector search with an optional language payload filter (and/or an
        arbitrary caller-supplied filter), used by the per-language retrieval
        branches in the retrieval pipeline."""
        effective_filter = query_filter
        if language is not None:
            lang_condition = models.FieldCondition(key="language", match=models.MatchValue(value=language))
            if effective_filter is None:
                effective_filter = models.Filter(must=[lang_condition])
            else:
                must = list(effective_filter.must or []) + [lang_condition]
                effective_filter = models.Filter(must=must)

        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            query_filter=effective_filter,
            limit=top_k,
            with_payload=True,
        ).points

    def hybrid_search(
        self,
        query_text: str,
        top_k: int = 5,
        language: Optional[str] = None,
        dense_prefetch_limit: int = 100,
        sparse_prefetch_limit: int = 100,
    ):
        """Dense + sparse hybrid search, fused server-side with Qdrant's RRF
        (Prefetch + FusionQuery). NOT the default retrieval path (see
        pipeline/components/retrieval.py) - exposed so hybrid can be switched
        on later behind the same Retriever interface once there is a dev set
        to measure whether it actually helps, per the user's decision to
        keep dense-only as the default for the first submission.

        Requires self.encoder to support encode_with_sparse() (BgeM3Embedder).
        """
        if not hasattr(self.encoder, "encode_with_sparse"):
            raise RuntimeError("hybrid_search requires an encoder with encode_with_sparse() (BgeM3Embedder)")

        dense_vecs, lexical_weights = self.encoder.encode_with_sparse([query_text], batch_size=1)
        dense_vec = dense_vecs[0]
        sparse_vec = self.encoder.lexical_weights_to_sparse_vector(lexical_weights[0])

        query_filter = None
        if language is not None:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="language", match=models.MatchValue(value=language))]
            )

        result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vec.tolist() if hasattr(dense_vec, "tolist") else list(dense_vec),
                    using=DENSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=dense_prefetch_limit,
                ),
                models.Prefetch(
                    query=sparse_vec,
                    using=SPARSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=sparse_prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return result.points

    def retrieve(self, query: str, top_k: int = 5, language: Optional[str] = None, verbose: bool = False):
        query_vector = self.encoder.encode(query, normalize_embeddings=True).tolist()
        results = self.search(query_vector, top_k=top_k, language=language)

        if verbose:
            for p in results:
                payload = p.payload or {}
                text = payload.get("content") or payload.get("text") or payload.get("query") or ""
                print(f"Score: {p.score:.4f} | ID: {p.id}")
                print(f"Content: {text[:150]}...\n")
        return results
