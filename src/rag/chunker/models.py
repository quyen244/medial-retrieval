from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterator, Optional


@dataclass
class Chunk:
    """Represents a discrete text chunk produced by the hierarchical chunking pipeline.
    
    Supports both object attribute access (chunk.content, chunk.metadata)
    and dict-style / indexing access (chunk['content'], chunk['metadata'], dict(chunk)).
    """
    chunk_id: str
    chunk_index: int
    content: str
    char_length: int
    doc_id: Any
    domain: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    heading_path: str = ""
    language: str = "vi"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to a flat dictionary suitable for serialization / indexing."""
        data = asdict(self)
        return data

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style indexing access e.g. chunk['content'], chunk['metadata']."""
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.metadata:
            return self.metadata[key]
        raise KeyError(f"Key '{key}' not found in Chunk or Chunk.metadata")

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-style setting e.g. chunk['metadata'] = ..."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.metadata[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in chunk attributes or metadata."""
        return hasattr(self, key) or (key in self.metadata)

    def get(self, key: str, default: Any = None) -> Any:
        """Safe get method like dict.get()."""
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.metadata:
            return self.metadata[key]
        return default

    def keys(self):
        """Enable dict(chunk) conversion and dict-like key iteration."""
        return self.to_dict().keys()

    def values(self):
        """Return values of chunk fields."""
        return self.to_dict().values()

    def items(self):
        """Return key-value items of chunk fields."""
        return self.to_dict().items()

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

