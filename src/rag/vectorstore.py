import os
from uuid import uuid4
from typing import Dict, Any, List, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from src.rag.chunking import default_dispatcher

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

    def upsert_doc(self, doc: Dict[str, Any]):
        """Hàm tiện ích nhận doc thô, tự chunk và lưu trữ."""
        chunks = default_dispatcher.chunk(doc)
        self.upsert_chunks(chunks)

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
                print(f"Score: {p.score:.4f} | ID: {p.payload.get('chunk_id')}")
                print(f"Content: {p.payload.get('content')[:150]}...\n")
        return results