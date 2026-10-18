import re
from typing import Any, Dict, Tuple
from src.rag.chunker.base import BasePreprocessor


def clean_markdown_syntax(text: str) -> str:
    """Filter all markdown decorators (**bold**, *italic*, __bold__, _italic_, #, ##, etc.)
    without losing any underlying text, numbers, punctuation or Vietnamese accents.
    
    Ensures that tokenization is not contaminated with markdown syntax artifacts,
    enabling perfect LCS (Longest Common Subsequence) ground truth matching.
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

        # 3. Strip bold and italic formatting pairs repeatedly
        prev = None
        while prev != l:
            prev = l
            # Bold-italic (***text*** or ___text___)
            l = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', l)
            l = re.sub(r'___(.+?)___', r'\1', l)
            # Bold (**text** or __text__)
            l = re.sub(r'\*\*(.+?)\*\*', r'\1', l)
            l = re.sub(r'__(.+?)__', r'\1', l)
            # Italic (*text* or _text_)
            l = re.sub(r'(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)', r'\1', l)
            l = re.sub(r'(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)', r'\1', l)

        # 4. Remove any stray/unmatched border markers or stray ** / __
        l = re.sub(r'^\s*[*_#]+\s*', '', l)
        l = re.sub(r'\s*[*_#]+\s*$', '', l)
        l = re.sub(r'\*\*+', '', l)
        l = re.sub(r'__+', '', l)

        # Normalize internal spacing
        l = re.sub(r'[ \t]+', ' ', l)
        cleaned_lines.append(l.strip())

    result = "\n".join(cleaned_lines)
    # Collapse 3+ newlines to standard double newline paragraph separation
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


class VietnameseMarkdownCleaner(BasePreprocessor):
    """Preprocessor that cleans markdown formatting syntax while preserving all textual content."""

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
