from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from src.rag.chunker.models import Chunk


class BasePreprocessor(ABC):
    """Abstract base class for document text preprocessors (e.g. heading promotion, noise cleaning)."""

    @abstractmethod
    def process(self, text: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Process text and return tuple of (transformed_text, statistics_dict)."""
        pass


class BaseSentenceSplitter(ABC):
    """Abstract base class for sentence splitters."""

    @abstractmethod
    def split_sentences(self, text: str) -> List[str]:
        """Split a block of text into a list of sentences without cutting abbreviations or decimals."""
        pass


class BaseChunker(ABC):
    """Abstract base class for language-specific chunking strategies.
    
    Adheres to Open/Closed Principle: new language chunkers can extend this class
    without modifying core pipeline or orchestrator logic.
    """

    @abstractmethod
    def chunk(self, doc: Dict[str, Any], **kwargs) -> List[Chunk]:
        """Chunk a document dictionary conforming to ingestion schema."""
        pass

    @abstractmethod
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> List[Chunk]:
        """Chunk raw text with optional initial metadata."""
        pass
    
    @abstractmethod
    def get_params(self):
        """Return parameters of a chunker."""
        pass
