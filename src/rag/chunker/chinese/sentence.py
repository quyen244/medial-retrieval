import re
from typing import List
from src.rag.chunker.base import BaseSentenceSplitter


class ChineseSentenceSplitter(BaseSentenceSplitter):
    """Sentence boundary detector tailored for Chinese medical text.

    Chinese orthography already separates the sentence-ending period (。) from the half-width
    Latin period (.) used in decimal numbers, drug dosages and inline Latin abbreviations
    (e.g. "99.5%", "IFNα-2a"). Because the two are visually distinct characters, this splitter
    only ever treats the full-width terminators 。！？；… (and their combinations, optionally
    followed by a closing quote/bracket) as sentence boundaries - unlike Vietnamese or English,
    no abbreviation list or decimal-protection step is needed: a half-width '.' simply never
    triggers a split.

    Guarantees:
    - Never splits on a half-width '.' (decimals, Latin abbreviations, dosages are inert).
    - Splits on [。！？；…]+ optionally followed by closing quotes/brackets (”’』）).
    """

    TERMINATOR = re.compile(r'([。！？；…]+[”’』）)]*)')

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences safely."""
        if not text or not text.strip():
            return []

        normalized = re.sub(r'\.{3,}', '…', text)

        parts = self.TERMINATOR.split(normalized)

        sentences: List[str] = []
        current = ""
        for i, p in enumerate(parts):
            current += p
            if i % 2 == 1:  # captured terminator group
                cleaned = current.strip()
                if cleaned:
                    sentences.append(cleaned)
                current = ""
        if current.strip():
            sentences.append(current.strip())

        return sentences
