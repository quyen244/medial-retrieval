from src.rag.vectorstore import VectorStore
import json 
from sentence_transformers import SentenceTransformer
import mlflow


mlflow.set_experiment('chunking_processing_v1')

encoder = SentenceTransformer("BAAI/bge-m3")
vectorstore = VectorStore(encoder, url = 'http://localhost:6333', collection_name='medical_documents')

if __name__ == "__main__":
    sample_file = r'D:\r2ai\data\raw_documents\bachmai.gov.vn\590.json'
    testset_file = r'D:\r2ai\data\testset\chunking_domain_sample.json'
    
    with mlflow.start_run():
        with open(sample_file , 'r', encoding='utf-8') as f:
            doc = json.load(f)
        
        params = vectorstore.upsert_doc(doc, versoning=True)
        
        mlflow.log_params(params)
        
        query = 'khói thuốc lá chứa bao nhiêu chất độc ?'
        
        vectorstore.retrieve(query, top_k = 2, verbose=True)
        
        
    
