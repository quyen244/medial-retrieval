from src.rag.indexing import VectorStore
from sentence_transformers import SentenceTransformer
import json 

# 1. Khởi tạo embedding model
encoder = SentenceTransformer("BAAI/bge-m3")
vectorstore = VectorStore(encoder, url = 'http://localhost:6333')


with open(r'D:\r2ai\data\raw_documents\baotayninh.vn\90106.json' , 'r' , encoding='utf-8') as f:
    doc = json.load(f)

vectorstore.upsert_doc(doc)

query = "Hoa cúc vạn thọ được dân dạng ưa chuộng vào dịp nào ?"

vectorstore.retrieve(query=query, verbose=True)






