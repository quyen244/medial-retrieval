import re
from typing import Any, Dict, Tuple
from src.rag.chunker.base import BasePreprocessor


# CommonMark's escapable-punctuation set: characters the HTML->Markdown extractor backslash-
# escapes to prevent them from being reparsed as markdown syntax (e.g. "10\~几十" so "~" is not
# misread as strikethrough, "1\." so it is not misread as an ordered-list marker). These escapes
# are an upstream conversion artifact, never part of the rendered page text, so they must be
# undone before the text reaches ground truth / embeddings.
_ESCAPABLE_PUNCTUATION = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
_BACKSLASH_ESCAPE_RE = re.compile(r'\\([' + re.escape(_ESCAPABLE_PUNCTUATION) + r'])')

# Backslash-escaped punctuation must be hidden from the emphasis-stripping regexes below, not
# just unescaped after the fact. Two consecutive escapes such as "a\*b\*c" (a literal '*' used
# as e.g. a multiplication/placeholder sign, not markdown) would otherwise still be seen by the
# italic regex as a real "*...*" pair once the surrounding backslashes are out of the way -
# whether the unescape happens before stripping (which destroys the literal asterisks outright,
# e.g. "\*text\*" -> "text") or after it (which leaves stray leftover backslashes behind, e.g.
# "\*text\*" -> "\text\" - itself a visible backslash leaking into the output, the same defect
# class this fix exists to remove). Swapping each "\<punct>" for a private-use-area placeholder
# before the emphasis regexes run makes the escaped punctuation invisible to them entirely (no
# bare '*'/'_' character for the regex to pair up), then restoring the placeholders afterwards
# reproduces the literal punctuation exactly, with no residual backslash and no false stripping.
_PLACEHOLDER_BASE = 0xE000
_ESCAPE_TO_PLACEHOLDER = {ch: chr(_PLACEHOLDER_BASE + i) for i, ch in enumerate(_ESCAPABLE_PUNCTUATION)}
_PLACEHOLDER_TO_ESCAPE = {placeholder: ch for ch, placeholder in _ESCAPE_TO_PLACEHOLDER.items()}
_PLACEHOLDER_RE = re.compile('[' + ''.join(_ESCAPE_TO_PLACEHOLDER.values()) + ']')


def _protect_backslash_escapes(line: str) -> str:
    """Replace each "\\<punct>" escape with a private-use placeholder for the punctuation char."""
    return _BACKSLASH_ESCAPE_RE.sub(lambda m: _ESCAPE_TO_PLACEHOLDER[m.group(1)], line)


def _restore_backslash_escapes(line: str) -> str:
    """Replace placeholders back with the literal punctuation character they stand for."""
    return _PLACEHOLDER_RE.sub(lambda m: _PLACEHOLDER_TO_ESCAPE[m.group(0)], line)


def clean_markdown_syntax(text: str) -> str:
    """Filter all markdown decorators (**bold**, *italic*, __bold__, _italic_, #, ##, footnote
    markup) without losing any underlying text, numbers, punctuation or Chinese characters.

    Unlike Latin scripts, CJK text has no whitespace between words, so emphasis markers are
    always directly adjacent to Han characters. Word-boundary lookarounds (\\w) therefore never
    fire for CJK and are deliberately not used here (unlike the Vietnamese cleaner).
    """
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        l = line.strip()
        if not l:
            cleaned_lines.append("")
            continue

        # 1. Strip markdown header hashes at the beginning of the line (#, ##, ###, etc.)
        l = re.sub(r'^\s*#{1,6}\s*', '', l)

        # 2. Strip bullet point asterisks at start of line (* item)
        l = re.sub(r'^\s*\*\s+', '', l)

        # 3. Strip footnote/reference markup: <sup>[1]</sup> (with literal escaped brackets
        # \[ \] as produced by markdown-escaping), and bare superscript tags.
        l = re.sub(r'<sup>\s*\\?\[\d+\\?\]\s*</sup>', '', l)
        l = re.sub(r'</?sup>', '', l)

        # 4. Strip wiki-style back-reference arrow used in reference lists (e.g. "1. ↑ Smith...")
        l = re.sub(r'↑\s*', '', l)

        # 4.5. Hide backslash-escaped punctuation from the emphasis-stripping regexes below (see
        # module-level comment on _protect_backslash_escapes for why this must happen here, not
        # as a plain post-hoc un-escape).
        l = _protect_backslash_escapes(l)

        # 5. Strip bold and italic formatting pairs repeatedly
        prev = None
        while prev != l:
            prev = l
            # Bold-italic (***text*** or ___text___)
            l = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', l)
            l = re.sub(r'___(.+?)___', r'\1', l)
            # Bold (**text** or __text__)
            l = re.sub(r'\*\*(.+?)\*\*', r'\1', l)
            l = re.sub(r'__(.+?)__', r'\1', l)
            # Italic (*text* or _text_) - no \w boundary guard: CJK text has no whitespace
            # separating markers from surrounding Han characters, so such guards never match.
            l = re.sub(r'\*(.+?)\*', r'\1', l)
            l = re.sub(r'_(.+?)_', r'\1', l)

        # 6. Remove any stray/unmatched border markers or stray ** / __
        l = re.sub(r'^\s*[*_#]+\s*', '', l)
        l = re.sub(r'\s*[*_#]+\s*$', '', l)
        l = re.sub(r'\*\*+', '', l)
        l = re.sub(r'__+', '', l)

        # 7. Restore the backslash-escaped punctuation hidden in step 4.5, now that the
        # emphasis-stripping regexes have had no chance to misinterpret it. Must run AFTER
        # steps 5-6, never before: restoring first would turn an intentionally-escaped literal
        # "\*text\*" into bare "*text*" before the emphasis regexes run, causing them to misread
        # it as real italic markup and strip it as formatting instead of leaving it as literal
        # text (or, with a plain post-hoc un-escape instead of a placeholder, leave stray
        # leftover backslashes behind - the same "visible backslash in output" defect being fixed).
        l = _restore_backslash_escapes(l)

        # Normalize internal spacing (ASCII whitespace only; CJK has no word-spacing to collapse)
        l = re.sub(r'[ \t]+', ' ', l)
        cleaned_lines.append(l.strip())

    result = "\n".join(cleaned_lines)
    # Collapse 3+ newlines to standard double newline paragraph separation
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


class ChineseMarkdownCleaner(BasePreprocessor):
    """Preprocessor that cleans markdown formatting syntax and footnote markup while preserving
    all textual content, tailored for CJK (no word-boundary markdown emphasis)."""

    def clean(self, text: str) -> str:
        """Clean markdown syntax from input string."""
        return clean_markdown_syntax(text)

    def process(self, text: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Process text, removing markdown syntax artifacts while returning statistics."""
        cleaned = clean_markdown_syntax(text)
        stats = {
            "original_length": len(text),
            "cleaned_length": len(cleaned),
            "chars_removed": len(text) - len(cleaned),
        }
        return cleaned, stats
