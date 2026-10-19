"""Central registry for retrieval pipeline components.

Generalizes src/rag/chunker/registry.py's ChunkerRegistry to multiple stage
"categories" (preprocessor, router, retriever, fusion, reranker) instead of
one. Components register under (category, key) and the pipeline resolves
them by name from a plain config dict - this is what makes a stage
swappable by changing one string rather than editing pipeline code.
"""
from typing import Any, Callable, Dict, List, Type, Union

Registrable = Union[Type, object, Callable[[], Any]]


class ComponentRegistry:
    _registry: Dict[str, Dict[str, Registrable]] = {}

    @classmethod
    def register(cls, category: str, key: str):
        """Decorator: @ComponentRegistry.register("retriever", "qdrant-language")"""
        def decorator(subclass):
            cls._registry.setdefault(category, {})[key.strip().lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def register_instance(cls, category: str, key: str, instance_or_cls: Registrable) -> None:
        cls._registry.setdefault(category, {})[key.strip().lower()] = instance_or_cls

    @classmethod
    def get(cls, category: str, key: str, default=None):
        return cls._registry.get(category, {}).get(key.strip().lower(), default)

    @classmethod
    def has(cls, category: str, key: str) -> bool:
        return key.strip().lower() in cls._registry.get(category, {})

    @classmethod
    def list_registered(cls, category: str) -> List[str]:
        return sorted(cls._registry.get(category, {}).keys())

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
