from typing import Any, Dict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.chunker.base import BaseChunker
from src.rag.chunker.models import Chunk
from src.rag.chunker.registry import ChunkerRegistry


@ChunkerRegistry.register("en")
@ChunkerRegistry.register("english")
class EnglishChunker(BaseChunker):
    """Chunker strategy for English documents adhering to OCP."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> List[Chunk]:
        doc = {"source": metadata or {}, "content": {"markdown": text}}
        return self.chunk(doc, **kwargs)

    def chunk(self, doc: Dict[str, Any], **kwargs) -> List[Chunk]:
        source = doc.get("source", {})
        doc_id = source.get("doc_id", "en_doc")
        raw_markdown = doc.get("content", {}).get("markdown", "")
        if not raw_markdown.strip():
            return []

        splits = self.splitter.split_text(raw_markdown)
        chunks: List[Chunk] = []
        for idx, text in enumerate(splits):
            chunk_id = f"{doc_id}_c{idx:03d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    chunk_index=idx,
                    content=text,
                    char_length=len(text),
                    doc_id=doc_id,
                    domain=source.get("domain", ""),
                    url=source.get("final_url", ""),
                    title=source.get("title", ""),
                    heading_path="",
                    language="en",
                    metadata={"word_count": len(text.split())},
                )
            )
        return chunks
