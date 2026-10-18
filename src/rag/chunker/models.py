from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class Chunk:
    """Represents a discrete text chunk produced by the hierarchical chunking pipeline."""
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
