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
                self._finalize_sentence(current, sentences)
                current = ""
        if current:
            self._finalize_sentence(current, sentences)

        return sentences

    @staticmethod
    def _finalize_sentence(current: str, sentences: List[str]) -> None:
        """Strip an accumulated sentence while preserving a meaningful '\\n' boundary.

        By the time text reaches this splitter, any surviving single '\\n' represents a
        blank-line-separated paragraph/list-item boundary that was collapsed earlier in the
        pipeline - it is never meaningless soft-wrap. A plain `.strip()` discards it outright,
        which silently fuses unrelated sentences together once call sites rejoin the returned
        sentences with "".join(). The regex split used above means this boundary newline
        surfaces as *leading* whitespace on the segment immediately following a terminator
        (e.g. "...营养素。\\n②一般..." splits into "...营养素。" then "\\n②一般..."), so it must be
        reattached to the END of the previously emitted sentence - not the current one - to
        reproduce the original separator when sentences are later joined with "".join().
        A newline stripped off the trailing edge (the rarer case, e.g. the final leftover
        segment at end-of-text) is instead appended to the current sentence itself.
        """
        cleaned = current.strip()
        if not cleaned:
            return

        leading_ws = current[:len(current) - len(current.lstrip())]
        if '\n' in leading_ws and sentences and not sentences[-1].endswith('\n'):
            sentences[-1] += '\n'

        trailing_ws = current[len(current.rstrip()):]
        if '\n' in trailing_ws:
            cleaned += '\n'

        sentences.append(cleaned)
