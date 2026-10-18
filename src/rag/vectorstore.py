import os
from uuid import uuid4
from typing import Dict, Any, List, Optional, Union
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from src.rag.chunker import default_dispatcher

class VectorStore:
    def __init__(
        self, 
        encoder: SentenceTransformer, 
        path: Optional[str] = "data/qdrant_db", 
        url: Optional[str] = None,
        collection_name: str = "medical_documents",
    ):
        self.encoder = encoder
        self.collection_name = collection_name
        
        # Tự động chọn Embedded local hoặc Remote URL
        if url:
            self.client = QdrantClient(url=url)
        elif path:
            os.makedirs(path, exist_ok=True)
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")
            
        print("Qdrant Client initialized:", self.client)
        
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE,
                ),
            )
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
                id=str(point_id) if point_id is not None else str(uuid4()),
                vector=vector.tolist(),
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
                pid = str(ids[idx]) if ids and idx < len(ids) else str(uuid4())
                payload = {"content": txt, "text": txt}
                if metadatas and idx < len(metadatas):
                    payload.update(metadatas[idx])
                points.append(
                    models.PointStruct(
                        id=pid,
                        vector=vec.tolist(),
                        payload=payload,
                    )
                )
            self.client.upsert(collection_name=self.collection_name, points=points)

    def upsert_chunks(self, chunks: list):
        """Nhận trực tiếp list các Chunk đã được tạo bởi Dispatcher."""
        texts = [c.content for c in chunks]
        vectors = self.encoder.encode(texts, normalize_embeddings=True)

        points = [
            models.PointStruct(
                id=str(uuid4()),
                vector=vec.tolist(),
                payload=c.to_dict() if hasattr(c, "to_dict") else vars(c),
            )
            for vec, c in zip(vectors, chunks)
        ]
        # collection_name số ít
        self.client.upsert(collection_name=self.collection_name, points=points)
        
    def upsert_doc(self, doc: Dict[str, Any], versoning : bool = False):
        """Hàm tiện ích nhận doc thô, tự chunk và lưu trữ."""
        chunks, params = default_dispatcher.chunk(doc , versoning)
        self.upsert_chunks(chunks)
        return params

    def retrieve(self, query: str, top_k: int = 5, verbose: bool = False):
        query_vector = self.encoder.encode(query, normalize_embeddings=True).tolist()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        if verbose:
            for p in results:
                payload = p.payload or {}
                text = payload.get("content") or payload.get("text") or payload.get("query") or ""
                print(f"Score: {p.score:.4f} | ID: {p.id}")
                print(f"Content: {text[:150]}...\n")
        return results