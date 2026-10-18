"""RAG Chunking Module for Media & Medical Document Retrieval.

Supports backward-compatible simple splitters as well as the new Open/Closed (OCP)
multi-tier language dispatcher and Vietnamese hierarchical chunker.
"""
import os
import json
from typing import Any, Dict, List, Optional
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, 
    MarkdownHeaderTextSplitter,
)

# Export OCP modular components
from src.rag.chunker import (
    Chunk,
    BaseChunker,
    BasePreprocessor,
    BaseSentenceSplitter,
    ChunkerRegistry,
    ChunkerDispatcher,
    default_dispatcher,
    VietnameseHeadingPromoter,
    VietnameseSentenceSplitter,
    VietnameseHierarchicalChunker,
    EnglishChunker,
    ChineseChunker,
)
from pathlib import Path 

def chunk_document(doc: Dict[str, Any], language: Optional[str] = None, **kwargs) -> List[Chunk]:
    """Top-level convenience function to chunk any document via the language dispatcher."""
    return default_dispatcher.dispatch(doc, language=language, **kwargs)

def __to_txt(doc , chunks , root):
    end_file = doc['source']['domain'] + '_' + str(doc['source']['doc_id']) + '.txt'
    path = Path(root) / end_file
    data = ""
    
    data += '=' * 50 + '\n'
    data += 'Before chunking\n' 
    data += '=' * 50  + '\n'
    data += doc['content']['markdown'] + '\n'
    
    data += '=' * 50 + '\n'
    data += 'After chunking\n'
    data += '=' * 50 + '\n'
    
    for c in chunks:
        data += f"\n--- Chunk {c.chunk_id} ({c.char_length} chars, metadata: '{c.metadata}') ---"
        data += f"\n--- Content ---"
        data += c.content + '\n'

            
    with open(path , 'w' , encoding='utf-8') as f:
        f.write(data)
    
    print('Wrote successfully !!!')
    
if __name__ == "__main__":
    testset_file = r'D:\r2ai\data\testset\chunking_domain_sample.json'
    md_folder = r'D:\r2ai\tmp\test_chunking'

    os.makedirs(md_folder, exist_ok=True)
    
    with open(testset_file, 'r', encoding="utf-8") as f:
            docs = json.load(f)
            
    for doc in docs:
        chunks = chunk_document(doc)
        
        print(f"Produced {len(chunks)} chunks for doc_id {doc.get('source', {}).get('doc_id')}")
        
        __to_txt(doc, chunks , md_folder)
