from typing import Any, Callable, Dict, List, Optional, Type, Union
from src.rag.chunker.base import BaseChunker


class ChunkerRegistry:
    """Central registry for language chunkers and custom strategies.
    
    Enforces the Open/Closed Principle (OCP): new language strategies can be
    registered dynamically via decorator or explicit register() call without
    altering existing dispatching or chunking implementations.
    """

    _registry: Dict[str, Union[Type[BaseChunker], BaseChunker, Callable[[], BaseChunker]]] = {}

    @classmethod
    def register(cls, key: str):
        """Decorator to register a Chunker class or factory for a given language / strategy key."""
        def decorator(subclass):
            cls._registry[key.strip().lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def register_instance(cls, key: str, instance_or_cls: Union[Type[BaseChunker], BaseChunker, Callable[[], BaseChunker]]) -> None:
        """Register a chunker instance, class, or factory callable."""
        cls._registry[key.strip().lower()] = instance_or_cls

    @classmethod
    def get(cls, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Retrieve a registered chunker by language code or strategy name."""
        return cls._registry.get(key.strip().lower(), default)

    @classmethod
    def has(cls, key: str) -> bool:
        """Check whether a strategy is registered."""
        return key.strip().lower() in cls._registry

    @classmethod
    def list_registered(cls) -> List[str]:
        """Return list of all registered strategy/language keys."""
        return sorted(list(cls._registry.keys()))

    @classmethod
    def clear(cls) -> None:
        """Clear registry (mostly for test isolation)."""
        cls._registry.clear()
