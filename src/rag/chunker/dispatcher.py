import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union

from src.rag.chunker.base import BaseChunker
from src.rag.chunker.models import Chunk
from src.rag.chunker.registry import ChunkerRegistry

logger = logging.getLogger(__name__)


class ChunkerDispatcher:
    """Language & Strategy Dispatcher adhering to the Open/Closed Principle (OCP).
    
    Responsibilities:
    1. Inspects document metadata (doc['source']['language'] or 'language_category') or explicit parameter.
    2. Automatically dispatches the document to the corresponding language-specific chunking strategy.
    3. Supports dynamic runtime registration of new languages/strategies without modifying dispatcher code.
    4. Provides safe fallback to default language ('vi') if an unrecognized language code is supplied.
    """

    def __init__(self, default_language: str = "vi"):
        self.default_language = default_language.lower()
        self._instances: Dict[str, BaseChunker] = {}

    @classmethod
    def register(cls, lang_code: str):
        """Decorator to register a chunker class with ChunkerRegistry."""
        return ChunkerRegistry.register(lang_code)

    def register_chunker(
        self,
        lang_code: str,
        chunker_or_cls: Union[Type[BaseChunker], BaseChunker, Callable[[], BaseChunker]],
    ) -> None:
        """Register a chunker strategy for a language code in the registry."""
        ChunkerRegistry.register_instance(lang_code, chunker_or_cls)
        # Clear cached instance if already initialized
        self._instances.pop(lang_code.lower(), None)

    def _resolve_chunker(self, lang_code: str) -> BaseChunker:
        """Resolve or instantiate a chunker strategy for the requested language code."""
        key = lang_code.lower()
        if key in self._instances:
            return self._instances[key]

        target_cls = ChunkerRegistry.get(key)
        if target_cls is None:
            logger.warning(
                f"No chunker registered for language '{lang_code}'. "
                f"Falling back to default '{self.default_language}'."
            )
            key = self.default_language
            target_cls = ChunkerRegistry.get(key)
            if target_cls is None:
                raise RuntimeError(
                    f"Default chunker for '{self.default_language}' is not registered! "
                    f"Available: {ChunkerRegistry.list_registered()}"
                )

        if isinstance(target_cls, BaseChunker):
            instance = target_cls
        elif isinstance(target_cls, type):
            instance = target_cls()
        elif callable(target_cls):
            instance = target_cls()
        else:
            raise TypeError(f"Registered chunker for '{key}' is neither a class nor an instance: {target_cls}")

        self._instances[key] = instance
        return instance

    def detect_language(self, doc: Dict[str, Any], explicit_lang: Optional[str] = None) -> str:
        """Extract language identifier from parameters or document metadata and normalize (e.g. vi-VN -> vi)."""
        lang = explicit_lang
        if not lang or not str(lang).strip():
            source = doc.get("source", {})
            lang = source.get("language") or source.get("language_category")

        if lang and isinstance(lang, str) and lang.strip():
            cleaned = lang.strip().lower()
            # Normalize regional tag e.g. 'vi-vn' -> 'vi', 'en-us' -> 'en', 'zh-cn' -> 'zh'
            base_lang = cleaned.split("-")[0].split("_")[0]
            if ChunkerRegistry.has(cleaned):
                return cleaned
            if ChunkerRegistry.has(base_lang):
                return base_lang
            return base_lang

        return self.default_language

    def dispatch(self, doc: Dict[str, Any], language: Optional[str] = None, **kwargs) -> List[Chunk]:
        """Dispatch document to appropriate language chunker and return produced Chunks."""
        target_lang = self.detect_language(doc, explicit_lang=language)
        chunker = self._resolve_chunker(target_lang)
        chunks = chunker.chunk(doc, **kwargs)
        return chunks

    def chunk(self, doc: Dict[str, Any], language: Optional[str] = None, **kwargs) -> List[Chunk]:
        """Convenience alias for dispatch()."""
        return self.dispatch(doc, language=language, **kwargs)


# Global default instance
default_dispatcher = ChunkerDispatcher(default_language="vi")
