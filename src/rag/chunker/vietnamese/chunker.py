from typing import Any, Dict, List, Optional
from langchain_text_splitters import MarkdownHeaderTextSplitter

from src.rag.chunker.base import BaseChunker
from src.rag.chunker.models import Chunk
from src.rag.chunker.registry import ChunkerRegistry
from src.rag.chunker.vietnamese.promoter import VietnameseHeadingPromoter
from src.rag.chunker.vietnamese.sentence import VietnameseSentenceSplitter


@ChunkerRegistry.register("vi")
@ChunkerRegistry.register("vietnamese")
class VietnameseHierarchicalChunker(BaseChunker):
    """Two-tier hierarchical chunker tailored for Vietnamese medical & news documents.
    
    Tier 1 (Section Level):
      - Promotes pseudo-headings (**1. ...**, **I. ...**) into markdown headings (## ...).
      - Splits document into semantic sections via MarkdownHeaderTextSplitter with strip_headers=False,
        retaining heading markers within content to optimize LCS matching and answer spans.
      - If 0 headings are found, the entire document constitutes a single section.
      
    Tier 2 (Paragraph Level & Greedy Merge):
      - Within each section, splits by '\\n\\n' into paragraph units.
      - Greedy merges sequential paragraphs towards target size (800 - 1500 chars).
      - Fallback: if a single paragraph exceeds max_chunk_size, safely splits at sentence boundaries
        using VietnameseSentenceSplitter without breaking words, abbreviations or decimals.
      - Sentence-level overlap: ~10-15% of chunk length is carried over between consecutive chunks.
      - Retains comprehensive metadata: doc_id, domain, url, title, heading_path, chunk_id,
        chunk_index, char_length, word_count, promoted_headings_count, etc.
    """

    DEFAULT_HEADERS = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]

    def __init__(
        self,
        min_chunk_size: int = 600,
        target_chunk_size: int = 1000,
        max_chunk_size: int = 1500,
        overlap_ratio: float = 0.12,
        headers_to_split: Optional[List[tuple]] = None,
        promoter: Optional[VietnameseHeadingPromoter] = None,
        sentence_splitter: Optional[VietnameseSentenceSplitter] = None,
    ):
        self.min_chunk_size = min_chunk_size
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_ratio = overlap_ratio
        self.headers_to_split = headers_to_split or self.DEFAULT_HEADERS
        self.promoter = promoter or VietnameseHeadingPromoter()
        self.sentence_splitter = sentence_splitter or VietnameseSentenceSplitter()

        self.section_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split,
            strip_headers=False,
        )

    def _build_heading_path(self, header_metadata: Dict[str, Any]) -> str:
        """Construct hierarchical breadcrumb path from header metadata (e.g. H1 > H2)."""
        levels = ["h1", "h2", "h3", "h4", "h5", "h6"]
        path_items = [str(header_metadata[lvl]).strip() for lvl in levels if lvl in header_metadata and header_metadata[lvl]]
        return " > ".join(path_items)

    def _extract_sentence_overlap(self, chunk_text: str) -> str:
        """Extract trailing sentences representing roughly 10-15% of the chunk length."""
        sentences = self.sentence_splitter.split_sentences(chunk_text)
        if len(sentences) <= 1:
            return ""

        target_overlap_len = int(len(chunk_text) * self.overlap_ratio)
        overlap_sentences: List[str] = []
        accumulated_len = 0

        # Traverse sentences in reverse
        for s in reversed(sentences):
            # Do not consume the entire chunk as overlap
            if len(overlap_sentences) + 1 >= len(sentences):
                break
            overlap_sentences.insert(0, s)
            accumulated_len += len(s) + 1
            if accumulated_len >= target_overlap_len:
                break

        return " ".join(overlap_sentences).strip()

    def _slice_oversized_unit(self, text: str) -> List[str]:
        """Fallback slice for an oversized sentence or token that exceeds max_chunk_size."""
        words = text.split()
        if len(words) > 1:
            chunks: List[str] = []
            curr_words: List[str] = []
            curr_len = 0
            for w in words:
                w_len = len(w)
                candidate = curr_len + (1 if curr_words else 0) + w_len
                if candidate <= self.max_chunk_size:
                    curr_words.append(w)
                    curr_len = candidate
                else:
                    if curr_words:
                        chunks.append(" ".join(curr_words))
                    if w_len > self.max_chunk_size:
                        # Hard slice continuous sequence without spaces
                        for i in range(0, w_len, self.max_chunk_size):
                            chunks.append(w[i:i + self.max_chunk_size])
                        curr_words = []
                        curr_len = 0
                    else:
                        curr_words = [w]
                        curr_len = w_len
            if curr_words:
                chunks.append(" ".join(curr_words))
            return chunks
        else:
            # Single gigantic token without spaces (e.g. base64 corrupted blob)
            return [text[i:i + self.max_chunk_size] for i in range(0, len(text), self.max_chunk_size)]

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """Split an oversized paragraph into units, each guaranteed <= max_chunk_size."""
        sentences = self.sentence_splitter.split_sentences(paragraph)
        if not sentences:
            return self._slice_oversized_unit(paragraph)

        groups: List[str] = []
        current_group: List[str] = []
        current_len = 0

        for s in sentences:
            s_len = len(s)
            # If a single sentence exceeds max_chunk_size, safely split by word / char boundaries
            if s_len > self.max_chunk_size:
                if current_group:
                    groups.append(" ".join(current_group))
                    current_group = []
                    current_len = 0
                sliced_parts = self._slice_oversized_unit(s)
                groups.extend(sliced_parts)
                continue

            candidate_len = current_len + (1 if current_group else 0) + s_len
            if candidate_len <= self.max_chunk_size:
                current_group.append(s)
                current_len = candidate_len
            else:
                if current_group:
                    groups.append(" ".join(current_group))
                current_group = [s]
                current_len = s_len

        if current_group:
            groups.append(" ".join(current_group))

        return groups

    def _merge_paragraphs_to_chunks(self, paragraphs: List[str]) -> List[str]:
        """Greedy merge paragraphs into chunks adhering to target and max size limits with sentence overlap."""
        if not paragraphs:
            return []

        # Flatten any oversized paragraphs via sentence-boundary fallback
        units: List[str] = []
        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue
            if len(p_clean) > self.max_chunk_size:
                units.extend(self._split_long_paragraph(p_clean))
            else:
                units.append(p_clean)

        if not units:
            return []

        chunks: List[str] = []
        current_units: List[str] = []
        current_len = 0

        for unit in units:
            unit_len = len(unit)
            candidate_len = current_len + (2 if current_units else 0) + unit_len

            if candidate_len <= self.max_chunk_size:
                current_units.append(unit)
                current_len = candidate_len
            else:
                # Max chunk size would be exceeded. Emit current buffer
                if current_units:
                    chunk_text = "\n\n".join(current_units).strip()
                    chunks.append(chunk_text)

                    # Compute sentence-level overlap from the emitted chunk
                    overlap_prefix = self._extract_sentence_overlap(chunk_text)

                    if overlap_prefix:
                        # Check if overlap + unit fits within max_chunk_size
                        if len(overlap_prefix) + 2 + unit_len <= self.max_chunk_size:
                            current_units = [overlap_prefix, unit]
                            current_len = len(overlap_prefix) + 2 + unit_len
                        else:
                            # If overlap would overflow with unit, omit overlap
                            current_units = [unit]
                            current_len = unit_len
                    else:
                        current_units = [unit]
                        current_len = unit_len
                else:
                    current_units = [unit]
                    current_len = unit_len

        # Emit leftover buffer
        if current_units:
            leftover_text = "\n\n".join(current_units).strip()
            # If leftover only mirrors the overlap of previous chunk, avoid redundant duplicate
            if not (chunks and leftover_text == self._extract_sentence_overlap(chunks[-1])):
                chunks.append(leftover_text)

        return chunks

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> List[Chunk]:
        """Chunk raw text with optional initial metadata."""
        doc = {
            "source": metadata or {},
            "content": {"markdown": text},
        }
        return self.chunk(doc, **kwargs)

    def chunk(self, doc: Dict[str, Any], **kwargs) -> List[Chunk]:
        """Execute the complete two-tier chunking workflow on an ingested document."""
        source = doc.get("source", {})
        doc_id = source.get("doc_id", "unknown_doc")
        domain = source.get("domain", "")
        url = source.get("final_url") or source.get("requested_url", "")
        title = source.get("title", "")
        raw_markdown = doc.get("content", {}).get("markdown", "")

        if not raw_markdown or not raw_markdown.strip():
            return []

        # Step 1: Preprocessing & Heading Promotion
        promoted_markdown, promo_stats = self.promoter.process(raw_markdown)
        promoted_headings_count = promo_stats.get("promoted_count", 0)

        # Step 2: Tier 1 - Section Level Split
        sections = self.section_splitter.split_text(promoted_markdown)
        if not sections:
            # Fallback if splitter produces empty result
            from langchain_core.documents import Document
            sections = [Document(page_content=promoted_markdown, metadata={})]

        # Step 3: Tier 2 - Paragraph Level & Greedy Merge across sections
        result_chunks: List[Chunk] = []
        global_chunk_index = 0

        for sec in sections:
            sec_content = sec.page_content.strip()
            if not sec_content:
                continue

            heading_path = self._build_heading_path(sec.metadata)
            paragraphs = sec_content.split("\n\n")

            merged_texts = self._merge_paragraphs_to_chunks(paragraphs)

            for text_chunk in merged_texts:
                clean_chunk = text_chunk.strip()
                if not clean_chunk:
                    continue

                chunk_id = f"{doc_id}_c{global_chunk_index:03d}"
                char_length = len(clean_chunk)
                word_count = len(clean_chunk.split())

                chunk_obj = Chunk(
                    chunk_id=chunk_id,
                    chunk_index=global_chunk_index,
                    content=clean_chunk,
                    char_length=char_length,
                    doc_id=doc_id,
                    domain=domain,
                    url=url,
                    title=title,
                    heading_path=heading_path,
                    language="vi",
                    metadata={
                        **sec.metadata,
                        "word_count": word_count,
                        "promoted_headings_count": promoted_headings_count,
                        "headings_before": promo_stats.get("headings_before", 0),
                        "headings_after": promo_stats.get("headings_after", 0),
                    }
                )
                result_chunks.append(chunk_obj)
                global_chunk_index += 1

        return result_chunks
