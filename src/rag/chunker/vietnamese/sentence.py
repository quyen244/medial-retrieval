import re
from typing import List
from src.rag.chunker.base import BaseSentenceSplitter


class VietnameseSentenceSplitter(BaseSentenceSplitter):
    """Robust sentence boundary detector tailored for Vietnamese medical and journalistic texts.
    
    Protects:
    - Academic, medical & honorary titles: PGS.TS., GS.TS., ThS.BS., PGS., GS., TS., ThS., BS., CN., DS., KTV., etc.
    - Administrative units: TP., TT., Q., H., P., TX., TW., etc.
    - Ministries & bodies: BYT., BXD., UBND., HĐND., etc.
    - Decimals, currencies & measurement numbers: e.g. 1.5, 3.14, 1.000, 100.000
    - Ellipsis & abbreviations: ..., v.v., vv., etc.
    
    Guarantees:
    - Never splits across protected abbreviations or decimal values.
    - Splits on [.?!;…] followed by whitespace or line boundary, preserving intact semantic units.
    """

    COMMON_ABBREVIATIONS = [
        "PGS.TS.", "PGS.TS", "GS.TS.", "GS.TS", "ThS.BS.", "ThS.BS", "BS.CKI", "BS.CKII",
        "PGS.", "GS.", "TS.", "ThS.", "BS.", "CN.", "DS.", "KTV.",
        "TP.", "TT.", "Q.", "H.", "P.", "TX.", "TW.",
        "BYT.", "BXD.", "BGD.", "UBND.", "HĐND.",
        "v.v.", "v..v", "v.v", "vv."
    ]

    def __init__(self, custom_abbreviations: List[str] = None):
        abbrs = list(self.COMMON_ABBREVIATIONS)
        if custom_abbreviations:
            abbrs.extend(custom_abbreviations)
        # Sort descending by length so longer abbreviations match first
        self.abbrs = sorted(list(set(abbrs)), key=len, reverse=True)

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences safely."""
        if not text or not text.strip():
            return []

        protected = text

        # 1. Protect abbreviations
        for i, abbr in enumerate(self.abbrs):
            token = f"§ABBR_{i}§"
            protected = protected.replace(abbr, token)

        # 2. Protect decimals (e.g., 3.14 or 1.000)
        protected = re.sub(r'(\d+)\.(\d+)', r'\1§DOT§\2', protected)

        # 3. Protect numbered lists / headings (e.g., "1. ", "2. ", "10. ", "IV. ")
        protected = re.sub(r'(^|\n|\b)(\d+)\.\s+', r'\1\2§NUMDOT§ ', protected)
        protected = re.sub(r'(^|\n|\b)([IVXLCDM]+)\.\s+', r'\1\2§ROMANDOT§ ', protected)

        # 4. Standardize multi-dots (...) into unicode ellipsis (…) so it serves as a delimiter
        protected = re.sub(r'\.{3,}', '…', protected)

        # 5. Split by sentence terminators: [.?!;…] followed by whitespace or line break
        # Regex captures delimiters to preserve them with the preceding sentence
        parts = re.split(r'([.?!;…]+(?:\s+|$))', protected)

        raw_sentences: List[str] = []
        current = ""
        for p in parts:
            current += p
            if re.search(r'[.?!;…]+(?:\s+|$)', p):
                cleaned = current.strip()
                if cleaned:
                    raw_sentences.append(cleaned)
                current = ""
        if current.strip():
            raw_sentences.append(current.strip())

        # 6. Restore placeholders in all sentences
        restored_sentences: List[str] = []
        for s in raw_sentences:
            s = s.replace("§DOT§", ".")
            s = s.replace("§NUMDOT§", ".")
            s = s.replace("§ROMANDOT§", ".")
            s = s.replace("…", "...")
            for i, abbr in enumerate(self.abbrs):
                token = f"§ABBR_{i}§"
                s = s.replace(token, abbr)
            restored_sentences.append(s)

        return restored_sentences
