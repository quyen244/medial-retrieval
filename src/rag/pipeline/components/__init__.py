"""Importing this package registers all baseline pipeline components."""
from src.rag.pipeline.components import routing  # noqa: F401
from src.rag.pipeline.components import retrieval  # noqa: F401
from src.rag.pipeline.components import fusion  # noqa: F401
from src.rag.pipeline.components import reranking  # noqa: F401
