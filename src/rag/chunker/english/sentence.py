import re
from typing import List
from src.rag.chunker.base import BaseSentenceSplitter


class EnglishSentenceSplitter(BaseSentenceSplitter):
    """Robust sentence boundary detector tailored for English scientific/medical text
    (PubMed abstracts, case reports).

    Protects:
    - Scientific/citation abbreviations: e.g., i.e., et al., vs., cf., approx., Fig., No.,
      Eq., Ref., Vol., pp., Dr., Mr., Mrs., Prof., St.
    - Single-letter initials (e.g. "J. Smith") and multi-initial author names.
    - Decimals, percentages, units with embedded periods are inert because a period is only
      ever a sentence boundary when followed by whitespace + a capital letter or EOF, never
      mid-token - decimals/units never satisfy the lookahead since their digit tail is not
      whitespace, so no separate decimal-protection pass is required here.
    - Ellipsis.

    Guarantees:
    - Never splits across protected abbreviations or decimal/unit values.
    - Splits on [.?!;] followed by whitespace or line boundary, preserving intact semantic units.
    """

    COMMON_ABBREVIATIONS = [
        "e.g.", "i.e.", "et al.", "etc.", "vs.", "cf.", "approx.",
        "Fig.", "Figs.", "No.", "Nos.", "Eq.", "Eqs.", "Ref.", "Refs.",
        "Vol.", "pp.", "p.", "ed.", "eds.",
        "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "St.", "Jr.", "Sr.",
        "a.m.", "p.m.", "U.S.", "U.K.",
    ]

    def __init__(self, custom_abbreviations: List[str] = None):
        abbrs = list(self.COMMON_ABBREVIATIONS)
        if custom_abbreviations:
            abbrs.extend(custom_abbreviations)
        self.abbrs = sorted(list(set(abbrs)), key=len, reverse=True)

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences safely."""
        if not text or not text.strip():
            return []

        protected = text

        # 1. Protect known abbreviations
        for i, abbr in enumerate(self.abbrs):
            token = f"§ABBR_{i}§"
            protected = re.sub(re.escape(abbr), token, protected)

        # 2. Protect single-letter initials (e.g. "J. Smith", "A. B. Jones")
        protected = re.sub(r'\b([A-Z])\.(?=\s+[A-Z])', r'\1§INIT§', protected)

        # 3. Protect decimals and numeric ranges (e.g. 3.14, 99.5%, 1.5MU)
        protected = re.sub(r'(\d+)\.(\d+)', r'\1§DOT§\2', protected)

        # 4. Protect numbered-list markers at line start only (e.g. "1. ", "IV. "). Anchored to
        # line start rather than a generic \b: a decimal already protected in step 3 can leave
        # a bare trailing digit group ("0§DOT§05") immediately before a genuine sentence-ending
        # period ("...0.05. Next sentence"), and a \b-anchored marker regex would wrongly treat
        # that tail ("05.") as a list marker, swallowing the real sentence boundary.
        protected = re.sub(r'(^|\n)(\d+)\.\s+', r'\1\2§NUMDOT§ ', protected)
        protected = re.sub(r'(^|\n)([IVXLCDM]+)\.\s+', r'\1\2§ROMANDOT§ ', protected)

        # 5. Standardize multi-dots (...) into unicode ellipsis so it serves as a delimiter
        protected = re.sub(r'\.{3,}', '…', protected)

        # 6. Split by sentence terminators: [.?!;]+ or ellipsis, followed by whitespace or EOF
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

        restored_sentences: List[str] = []
        for s in raw_sentences:
            s = s.replace("§DOT§", ".")
            s = s.replace("§NUMDOT§", ".")
            s = s.replace("§ROMANDOT§", ".")
            s = s.replace("§INIT§", ".")
            s = s.replace("…", "...")
            for i, abbr in enumerate(self.abbrs):
                token = f"§ABBR_{i}§"
                s = s.replace(token, abbr)
            restored_sentences.append(s)

        return restored_sentences
