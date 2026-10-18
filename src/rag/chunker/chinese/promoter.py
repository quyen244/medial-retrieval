import re
from typing import Any, Dict, List, Optional, Tuple
from src.rag.chunker.base import BasePreprocessor


class ChineseHeadingPromoter(BasePreprocessor):
    """Preprocessor for Chinese medical articles that detects pseudo-headings and promotes
    them to true markdown headings (#### ...), while filtering false positives.

    Chinese enumeration markers (unlike Vietnamese bold-numbered paragraphs) are frequently
    plain text, not bold-wrapped: '一、', '(一)', '1、', '①' etc. are used interchangeably as
    both real sub-section headings (standalone paragraph, short label, body follows separately)
    and as inline list-item numbering fused with its own body text on the same line/paragraph
    (e.g. '1.营养不良学说 实验动物中缺乏叶酸...'). The two cases are told apart the same way
    Vietnamese tells a real heading from an inline bold emphasis: standalone paragraph, short
    length, and no sentence-level punctuation (，。！？) inside the body - fused list items
    always carry such punctuation or run well past the length cap.

    Promotes to '####' (h4, the deepest level) rather than '##' because Chinese source
    documents (zhongyibaodian.net, a-hospital.com) already carry real '##'/'###' structure;
    promoting pseudo-headings to a shallower level would invert the real hierarchy.

    Guarantees:
    - Matches arabic/Chinese-numeral/circled-numeral markers, plain or bold-wrapped.
    - False-positive filters: standalone paragraph, length < max_length, no 、。，！？ punctuation
      inside the body (after the trailing '：' is stripped), decimal numbers protected.
    - Safe execution even if 0 headings are found.
    """

    CN_DIGITS = "一二三四五六七八九十百千"
    CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"

    PATTERNS = [
        re.compile(r'^(\d{1,3})(?!\d)[\.、]\s*(.+)$'),                      # 1. / 1、
        re.compile(r'^[\(（](\d{1,3})[\)）]\s*(.+)$'),                       # (1) / （1）
        re.compile(rf'^([{CN_DIGITS}]{{1,4}})[、\.]\s*(.+)$'),               # 一、/ 一.
        re.compile(rf'^[\(（]([{CN_DIGITS}]{{1,4}})[\)）]\s*(.+)$'),         # (一) / （一）
        re.compile(rf'^([{CIRCLED}])\s*(.+)$'),                             # ①
    ]

    EXISTING_HEADING_PATTERN = re.compile(r'^#{1,6}\s+.*$', re.MULTILINE)
    # Semicolon (；) is included because Chinese uses it to coordinate parallel items in an
    # enumerated list (e.g. "...不宜驾驶车辆及操作机器；"); a body ending in one is itself one
    # coordinate list item, not a sub-heading, even when it has no comma/period of its own.
    DISQUALIFYING_PUNCT = "，。！？；,;"

    def __init__(self, max_length: int = 30, heading_level: str = "####"):
        self.max_length = max_length
        self.heading_level = heading_level

    def count_headings(self, text: str) -> int:
        """Count markdown headings (# to ######) in text."""
        return len(self.EXISTING_HEADING_PATTERN.findall(text))

    def normalize_existing_headings(self, text: str) -> str:
        """Strip redundant bold/italic wrappers inside existing markdown headings."""
        def clean_heading_line(match):
            hashes = match.group(1)
            content = match.group(2).strip()
            cleaned_content = re.sub(r'^\s*[*_]+\s*', '', content)
            cleaned_content = re.sub(r'\s*[*_]+\s*$', '', cleaned_content)
            cleaned_content = re.sub(r'\*\*(.+?)\*\*', r'\1', cleaned_content)
            cleaned_content = re.sub(r'__(.+?)__', r'\1', cleaned_content)
            return f"{hashes} {cleaned_content.strip()}"

        return re.sub(r'^(#{1,6})\s+(.+?)$', clean_heading_line, text, flags=re.MULTILINE)

    def _match_marker(self, inner: str) -> Optional[Tuple[str, str]]:
        """Try marker patterns against a candidate paragraph, return (marker, body) or None."""
        for pat in self.PATTERNS:
            m = pat.match(inner)
            if m:
                return m.group(1), m.group(2).strip()
        return None

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

            is_bold = stripped.startswith('**') and stripped.endswith('**') and len(stripped) > 4
            inner = stripped[2:-2].strip() if is_bold else stripped

            match = self._match_marker(inner)
            if not match:
                new_tokens.append(token)
                continue

            marker, body = match
            body_check = re.sub(r'[：:]\s*$', '', body).strip()

            if not body_check or any(ch in self.DISQUALIFYING_PUNCT for ch in body_check):
                new_tokens.append(token)
                continue

            promoted_text = f"{self.heading_level} {inner}"
            new_tokens.append(promoted_text)
            promoted_items.append(promoted_text)

        result_text = "".join(new_tokens)
        headings_after = self.count_headings(result_text)

        stats = {
            "headings_before": headings_before,
            "promoted_count": len(promoted_items),
            "headings_after": headings_after,
            "promoted_items": promoted_items,
        }

        return result_text, stats
