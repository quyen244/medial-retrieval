from src.rag.vectorstore import VectorStore
import json 
from sentence_transformers import SentenceTransformer
from src.ingestion.utils import get_dataset

# 1. Khởi tạo embedding model
encoder = SentenceTransformer("BAAI/bge-m3")
vectorstore = VectorStore(encoder, url = 'http://localhost:6333', collection_name='query_documents')

dataset = get_dataset('query')

print("dataset : " ,dataset)
    
if __name__ == "__main__":  
    
    texts = list(map( lambda x : x['query'] , dataset['train']))
    
    print(texts[:5])
    print()

    vectorstore.upsert_texts(texts)

    query = 'bạn em thắc mắc về vấn đề đau răng âm ỉ hơn 1 tuần'
    
    vectorstore.retrieve(query = query, verbose=True)


