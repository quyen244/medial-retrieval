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

headers = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

REGISTRY_SPLITTERS = {
    'markdown': MarkdownHeaderTextSplitter,
    'recursive': RecursiveCharacterTextSplitter,
}

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=850,
    chunk_overlap=150,
    length_function=len,
)


def chunk_document(doc: Dict[str, Any], language: Optional[str] = None, **kwargs) -> List[Chunk]:
    """Top-level convenience function to chunk any document via the language dispatcher."""
    return default_dispatcher.dispatch(doc, language=language, **kwargs)


def run(name: str, doc: str, **kwargs):
    """Legacy runner for backward compatibility."""
    print('=' * 50)
    print(f"{name} splitter")
    print('=' * 50)
    
    splitter = REGISTRY_SPLITTERS[name](**kwargs)
    sections = splitter.split_text(doc)
    
    chunks = recursive_splitter.split_documents(sections)

    for i, chunk in enumerate(chunks):
        print(f"\n{'=' * 20} CHUNK {i + 1} {'=' * 20}")
        print("Metadata:", chunk.metadata)
        print("Content:", chunk.page_content)


if __name__ == "__main__":
    sample_file = r'D:\r2ai\data\raw_documents\baodanang.vn\28721.json'
    testset_file = r'D:\r2ai\data\testset\chunking_domain_sample.json'
    
    target_path = sample_file if os.path.exists(sample_file) else testset_file
    print(f"Loading sample from: {target_path}")
    
    with open(target_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        
    doc = data[0] if isinstance(data, list) else data
    chunks = chunk_document(doc)
    
    print(f"Produced {len(chunks)} chunks for doc_id {doc.get('source', {}).get('doc_id')}")
    for c in chunks[:3]:
        print(f"\n--- Chunk {c.chunk_id} ({c.char_length} chars, heading: '{c.heading_path}') ---")
        print(c.content[:200] + "...")
