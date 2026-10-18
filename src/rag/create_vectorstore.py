from src.rag.vectorstore import VectorStore
import json 
from sentence_transformers import SentenceTransformer
from src.ingestion.utils import get_dataset

# 1. Khởi tạo embedding model
encoder = SentenceTransformer("BAAI/bge-m3")
vectorstore = VectorStore(encoder, url = 'http://localhost:6333', collection_name='query_documents')

dataset = get_dataset('query')

print("dataset : " ,dataset)

def smoke_test():
    first_doc = dataset['train'][0]
    
    print('First doc : ' , first_doc)
    
    vectorstore.upsert_doc(first_doc)

    query = 'bạn em thắc mắc về vấn đề đau răng'
    
    vectorstore.retrieve(query = query, verbose=True)
    

if __name__ == "__main__":  
    smoke_test()



