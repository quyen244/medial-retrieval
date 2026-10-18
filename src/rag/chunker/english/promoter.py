import re
from typing import Any, Dict, List, Tuple
from src.rag.chunker.base import BasePreprocessor


class EnglishHeadingPromoter(BasePreprocessor):
    """Preprocessor for English articles that detects pseudo-headings in markdown
    (e.g., bold numbered/lettered/roman-numeral lines styled as headings) and promotes them
    to true markdown headings (## ...), while filtering false positives.

    PubMed abstracts are already well-structured (# Title, ## Abstract, ### Background/Methods/
    Results/Conclusion, ## Citation) so this rarely fires on the current corpus - it exists for
    robustness/symmetry with the Vietnamese/Chinese promoters and in case less-structured
    English content is added later (e.g. case reports with bold-numbered findings lists).

    Guarantees:
    - Matches numbers: **1. ...** or **1) ...**; letters: **A. ...** or **A) ...**;
      Roman numerals: **I. ...** or **I) ...**.
    - Converts match to '## <content>' while retaining numbering for semantic clarity.
    - False-positive filters:
      * Must stand alone as its own paragraph (bounded by blank lines / text boundaries).
      * Max line length < 100 characters.
      * No intermediate period in the sentence body (only the period directly attached to the
        marker is allowed).
    - Safe execution even if 0 headings are found.
    """

    NUMERIC_PATTERN = re.compile(r'^\*\*\s*(\d+)[\.\)]\s+(.+?)\*\*\s*$')
    ROMAN_PATTERN = re.compile(r'^\*\*\s*([IVXLC]+)[\.\)]\s+(.+?)\*\*\s*$')
    LETTER_PATTERN = re.compile(r'^\*\*\s*([A-Za-z])[\.\)]\s+(.+?)\*\*\s*$')
    EXISTING_HEADING_PATTERN = re.compile(r'^#{1,6}\s+.*$', re.MULTILINE)

    def __init__(self, max_length: int = 100, heading_level: str = "##"):
        self.max_length = max_length
        self.heading_level = heading_level

    def count_headings(self, text: str) -> int:
        """Count markdown headings (# to ######) in text."""
        return len(self.EXISTING_HEADING_PATTERN.findall(text))

    def normalize_existing_headings(self, text: str) -> str:
        """Strip redundant bold/italic wrappers inside existing markdown headings (e.g. ## **Title** -> ## Title)."""
        def clean_heading_line(match):
            hashes = match.group(1)
            content = match.group(2).strip()
            cleaned_content = re.sub(r'^\s*[*_]+\s*', '', content)
            cleaned_content = re.sub(r'\s*[*_]+\s*$', '', cleaned_content)
            cleaned_content = re.sub(r'\*\*(.+?)\*\*', r'\1', cleaned_content)
            cleaned_content = re.sub(r'__(.+?)__', r'\1', cleaned_content)
            return f"{hashes} {cleaned_content.strip()}"

        return re.sub(r'^(#{1,6})\s+(.+?)$', clean_heading_line, text, flags=re.MULTILINE)

    def process(self, text: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Promote pseudo-headings to standard markdown headings with false-positive filtering."""
        if not text:
            return "", {"headings_before": 0, "promoted_count": 0, "headings_after": 0, "promoted_items": []}

        normalized = text.replace('\r\n', '\n')
        normalized = self.normalize_existing_headings(normalized)

        headings_before = self.count_headings(normalized)

        tokens = re.split(r'(\n\s*\n)', normalized)

        promoted_items: List[str] = []
        new_tokens: List[str] = []

        for token in tokens:
            stripped = token.strip()

            if not stripped or '\n' in stripped:
                new_tokens.append(token)
                continue

            if len(stripped) >= self.max_length:
                new_tokens.append(token)
                continue

            m_num = self.NUMERIC_PATTERN.match(stripped)
            m_rom = self.ROMAN_PATTERN.match(stripped)
            m_let = self.LETTER_PATTERN.match(stripped)
            match = m_num or m_rom or m_let

            if match:
                marker = match.group(1)
                body = match.group(2).strip()

                delimiter = "."
                if f"{marker})" in stripped:
                    delimiter = ")"

                body_check = body[:-1] if body.endswith('.') else body
                if '.' in body_check:
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
