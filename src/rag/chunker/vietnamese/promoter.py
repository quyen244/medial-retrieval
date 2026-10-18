import re
from typing import Any, Dict, List, Tuple
from src.rag.chunker.base import BasePreprocessor


class VietnameseHeadingPromoter(BasePreprocessor):
    """Preprocessor for Vietnamese articles that detects pseudo-headings in markdown
    (e.g., bold numbered lists or Roman numerals styled as headings) and promotes them
    to true markdown headings (## ...), while filtering false positives.
    
    Guarantees:
    - Matches numbers: **1. ...** or **1) ...**
    - Matches Roman numerals: **I. ...** or **I) ...**
    - Converts match to '## <content>' while retaining numbering for semantic clarity.
    - False-positive filters:
      * Must stand alone as its own paragraph (bounded by blank lines / text boundaries).
      * Max line length < 100 characters.
      * No intermediate periods in the sentence body (only the period directly attached to the numeral is allowed).
    - Safe execution even if 0 headings are found.
    - Returns transformation statistics (headings before/after and promoted count).
    """

    NUMERIC_PATTERN = re.compile(r'^\*\*\s*(\d+)[\.\)]\s+(.+?)\*\*\s*$')
    ROMAN_PATTERN = re.compile(r'^\*\*\s*([IVXLC]+)[\.\)]\s+(.+?)\*\*\s*$')
    EXISTING_HEADING_PATTERN = re.compile(r'^#{1,6}\s+.*$', re.MULTILINE)

    def __init__(self, max_length: int = 100, heading_level: str = "##"):
        self.max_length = max_length
        self.heading_level = heading_level

    def count_headings(self, text: str) -> int:
        """Count markdown headings (# to ######) in text."""
        return len(self.EXISTING_HEADING_PATTERN.findall(text))

    def process(self, text: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Promote pseudo-headings to standard markdown headings with false-positive filtering."""
        if not text:
            return "", {"headings_before": 0, "promoted_count": 0, "headings_after": 0, "promoted_items": []}

        # Normalize line breaks
        normalized = text.replace('\r\n', '\n')
        
        headings_before = self.count_headings(normalized)

        # Split document by paragraph boundaries (\n\s*\n) while capturing delimiters to preserve layout
        tokens = re.split(r'(\n\s*\n)', normalized)

        promoted_items: List[str] = []
        new_tokens: List[str] = []

        for token in tokens:
            stripped = token.strip()

            # Must be a non-empty paragraph without internal newlines
            if not stripped or '\n' in stripped:
                new_tokens.append(token)
                continue

            # Filter: length must be < max_length (default 100)
            if len(stripped) >= self.max_length:
                new_tokens.append(token)
                continue

            # Check numeric match
            m_num = self.NUMERIC_PATTERN.match(stripped)
            m_rom = self.ROMAN_PATTERN.match(stripped)
            match = m_num or m_rom

            if match:
                marker = match.group(1)   # e.g., '1' or 'IV'
                body = match.group(2).strip()

                # Determine delimiter used in source (. or ))
                # We inspect original token to preserve '.' vs ')'
                # E.g. **1. Tiêu đề** -> '1.' ; **1) Tiêu đề** -> '1)'
                delimiter = "."
                if f"{marker})" in stripped:
                    delimiter = ")"

                # Filter: check for intermediate periods in the sentence body
                # Trailing period is allowed (e.g. **1. Tiêu đề.**) and stripped for heading format
                body_check = body[:-1] if body.endswith('.') else body
                if '.' in body_check:
                    # Intermediate period detected (e.g., abbreviations or full sentences -> false positive)
                    new_tokens.append(token)
                    continue

                clean_body = body[:-1].strip() if body.endswith('.') else body
                promoted_text = f"{self.heading_level} {marker}{delimiter} {clean_body}"
                new_tokens.append(promoted_text)
                promoted_items.append(promoted_text)
            else:
                new_tokens.append(token)

        result_text = "".join(new_tokens)
        headings_after = self.count_headings(result_text)

        stats = {
            "headings_before": headings_before,
            "promoted_count": len(promoted_items),
            "headings_after": headings_after,
            "promoted_items": promoted_items,
        }

        return result_text, stats
