import re
from typing import Any, Dict, List, Optional
from langchain_text_splitters import MarkdownHeaderTextSplitter

from src.rag.chunker.base import BaseChunker
from src.rag.chunker.models import Chunk
from src.rag.chunker.registry import ChunkerRegistry
from src.rag.chunker.vietnamese.cleaner import VietnameseMarkdownCleaner, clean_markdown_syntax
from src.rag.chunker.vietnamese.promoter import VietnameseHeadingPromoter
from src.rag.chunker.vietnamese.sentence import VietnameseSentenceSplitter


@ChunkerRegistry.register("vi")
@ChunkerRegistry.register("vietnamese")
class VietnameseHierarchicalChunker(BaseChunker):
    """Two-tier hierarchical chunker tailored for Vietnamese medical & news documents.
    
    Features:
      1. Clean Markdown Syntax: Strips markdown noise (**, *, __, _, #, ##) from chunk
         content without losing words or numbers, ensuring high LCS matching with ground truth.
      2. Heading-to-Content Binding: Tightly binds headings (< 120 chars) with their immediate
         content paragraphs to eliminate orphan headings at chunk ends.
      3. Cross-Section Greedy Merge: Merges consecutive small sections/items across boundaries
         until the target chunk size (800 - 1500 chars / ~200 - 300 words) is reached.
      4. Fallback Sentence Splitting: Gracefully splits oversized blocks (> 1500 chars)
         at sentence boundaries using VietnameseSentenceSplitter without breaking abbreviations.
      5. Sentence-Level Overlap: Carries over 10-15% trailing body sentences between consecutive chunks.
      6. Dual Object/Dict API: Returns Chunk instances supporting both object access (chunk.content)
         and dictionary indexing (chunk['content'], dict(chunk)).
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
        target_chunk_size: int = 800,
        max_chunk_size: int = 1000,
        overlap_ratio: float = 0.15,
        headers_to_split: Optional[List[tuple]] = None,
        promoter: Optional[VietnameseHeadingPromoter] = None,
        cleaner: Optional[VietnameseMarkdownCleaner] = None,
        sentence_splitter: Optional[VietnameseSentenceSplitter] = None,
    ):
        self.min_chunk_size = min_chunk_size
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_ratio = overlap_ratio
        self.headers_to_split = headers_to_split or self.DEFAULT_HEADERS
        self.promoter = promoter or VietnameseHeadingPromoter()
        self.cleaner = cleaner or VietnameseMarkdownCleaner()
        self.sentence_splitter = sentence_splitter or VietnameseSentenceSplitter()

        self.section_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split,
            strip_headers=False,
        )
    def get_params(self):
        return {
            'min_chunk_size' : self.min_chunk_size,
            'target_chunk_size' : self.target_chunk_size,
            'max_chunk_size' : self.max_chunk_size,
            'overlap_ratio' : self.overlap_ratio, 
            'headers_to_split' : self.DEFAULT_HEADERS,
            'promoter' : self.promoter,
            'cleaner' : self.cleaner ,
            'sentence_splitter' : self.sentence_splitter,
            'section_splitter' : self.section_splitter
        }
        
    def _build_composite_heading_path(self, headings: List[str], doc_title: str = "") -> str:
        """Construct hierarchical breadcrumb path from collected headings and document title."""
        clean_title = clean_markdown_syntax(doc_title).strip()
        filtered_headings = []
        for h in headings:
            h_clean = clean_markdown_syntax(h).strip()
            if h_clean and h_clean != clean_title and h_clean not in filtered_headings:
                filtered_headings.append(h_clean)

        if not filtered_headings:
            return clean_title

        if clean_title:
            return f"{clean_title} > {' | '.join(filtered_headings)}"
        return " | ".join(filtered_headings)

    def _extract_sentence_overlap(self, chunk_text: str) -> str:
        """Extract trailing sentences representing roughly 10-15% of chunk length from body text."""
        paras = [p.strip() for p in chunk_text.split("\n\n") if p.strip()]
        if not paras:
            return ""

        # Extract only from the last body paragraph to never overlap a standalone heading
        last_para = paras[-1]
        sentences = self.sentence_splitter.split_sentences(last_para)
        if not sentences:
            return ""

        target_overlap_len = int(len(chunk_text) * self.overlap_ratio)
        overlap_sentences: List[str] = []
        accumulated_len = 0

        # Traverse sentences in reverse
        for s in reversed(sentences):
            # Do not consume the entire last paragraph as overlap if multiple sentences exist
            if len(overlap_sentences) >= len(sentences):
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

    def _split_oversized_bound_unit(self, heading: str, body: str) -> List[str]:
        """Safely split an oversized bound heading+body unit while preserving the heading on the first slice."""
        clean_h = clean_markdown_syntax(heading).strip()
        clean_b = clean_markdown_syntax(body).strip()

        if not clean_b:
            if len(clean_h) > self.max_chunk_size:
                return self._slice_oversized_unit(clean_h)
            return [clean_h]

        sentences = self.sentence_splitter.split_sentences(clean_b)
        if not sentences:
            sentences = self._slice_oversized_unit(clean_b)

        result_units: List[str] = []
        first_group_sentences: List[str] = []
        current_len = len(clean_h) + 2

        # First slice: heading + as many sentences as fit within max_chunk_size
        idx = 0
        while idx < len(sentences):
            s = sentences[idx]
            s_len = len(s)
            candidate_len = current_len + (1 if first_group_sentences else 0) + s_len
            if candidate_len <= self.max_chunk_size:
                first_group_sentences.append(s)
                current_len = candidate_len
                idx += 1
            else:
                break

        if first_group_sentences:
            first_unit = f"{clean_h}\n\n{' '.join(first_group_sentences)}"
            result_units.append(first_unit)
        else:
            # Heading alone is close to max_chunk_size
            result_units.append(clean_h)

        # Remaining sentences split as regular paragraphs
        remaining_text = " ".join(sentences[idx:])
        if remaining_text.strip():
            if len(remaining_text) > self.max_chunk_size:
                result_units.extend(self._split_long_paragraph(remaining_text))
            else:
                result_units.append(remaining_text)

        return result_units

    def _build_semantic_units(self, sections: List[Any]) -> List[Dict[str, Any]]:
        """Transform sections into bound semantic units.
        
        Guarantees Heading-to-Content Binding:
        - When a section has a heading (< 120 chars), it is bound with its first content paragraph.
        - Oversized units are sliced with sentence fallback, keeping heading bound to initial slice.
        - Text is cleaned of all markdown syntax artifacts.
        """
        raw_units: List[Dict[str, Any]] = []

        for sec in sections:
            raw_content = sec.page_content.strip()
            if not raw_content:
                continue

            lines = raw_content.split("\n")
            first_line = lines[0].strip()

            # Check if section starts with a markdown heading #...
            has_markdown_heading = bool(re.match(r"^#{1,6}\s+", first_line))

            if has_markdown_heading:
                heading_clean = clean_markdown_syntax(first_line).strip()
                body_raw = "\n".join(lines[1:]).strip()
                body_paras = [p.strip() for p in re.split(r'\n\s*\n', body_raw) if p.strip()]

                if body_paras:
                    # Bind heading tightly to the first body paragraph
                    first_body = clean_markdown_syntax(body_paras[0])
                    bound_text = f"{heading_clean}\n\n{first_body}"

                    if len(bound_text) > self.max_chunk_size:
                        sub_units = self._split_oversized_bound_unit(heading_clean, first_body)
                        for su in sub_units:
                            raw_units.append({"text": su, "heading": heading_clean, "metadata": sec.metadata})
                    else:
                        raw_units.append({"text": bound_text, "heading": heading_clean, "metadata": sec.metadata})

                    # Subsequent body paragraphs under this heading
                    for p in body_paras[1:]:
                        clean_p = clean_markdown_syntax(p)
                        if len(clean_p) > self.max_chunk_size:
                            for su in self._split_long_paragraph(clean_p):
                                raw_units.append({"text": su, "heading": heading_clean, "metadata": sec.metadata})
                        else:
                            raw_units.append({"text": clean_p, "heading": heading_clean, "metadata": sec.metadata})
                else:
                    # Section only contained the heading line (no body in this section)
                    raw_units.append({"text": heading_clean, "heading": heading_clean, "metadata": sec.metadata})
            else:
                # Section without a leading markdown heading (e.g. intro section or doc without headings)
                sec_heading = ""
                for h_level in ["h4", "h3", "h2", "h1"]:
                    if h_level in sec.metadata and sec.metadata[h_level]:
                        sec_heading = clean_markdown_syntax(str(sec.metadata[h_level])).strip()
                        break

                paras = [p.strip() for p in re.split(r'\n\s*\n', raw_content) if p.strip()]
                for p in paras:
                    clean_p = clean_markdown_syntax(p)
                    if len(clean_p) > self.max_chunk_size:
                        for su in self._split_long_paragraph(clean_p):
                            raw_units.append({"text": su, "heading": sec_heading, "metadata": sec.metadata})
                    else:
                        raw_units.append({"text": clean_p, "heading": sec_heading, "metadata": sec.metadata})

        # Post-process: If an isolated heading unit is immediately followed by a content unit, bind them!
        consolidated_units: List[Dict[str, Any]] = []
        i = 0
        while i < len(raw_units):
            curr = raw_units[i]
            # Check if current unit is a short heading standing alone
            is_lone_heading = (
                len(curr["text"]) < 120
                and "\n\n" not in curr["text"]
                and (curr["text"] == curr.get("heading") or not curr["text"].endswith((".", "!", "?")))
            )
            if is_lone_heading and i + 1 < len(raw_units):
                nxt = raw_units[i + 1]
                combined_text = f"{curr['text']}\n\n{nxt['text']}"
                if len(combined_text) <= self.max_chunk_size:
                    consolidated_units.append({
                        "text": combined_text,
                        "heading": curr.get("heading") or nxt.get("heading"),
                        "metadata": {**curr["metadata"], **nxt["metadata"]},
                    })
                    i += 2
                    continue
            consolidated_units.append(curr)
            i += 1

        return consolidated_units

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> List[Chunk]:
        """Chunk raw text with optional initial metadata."""
        doc = {
            "source": metadata or {},
            "content": {"markdown": text},
        }
        return self.chunk(doc, **kwargs)

    def chunk(self, doc: Dict[str, Any], **kwargs) -> List[Chunk]:
        """Execute the hierarchical chunking workflow on an ingested document."""
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
            from langchain_core.documents import Document
            sections = [Document(page_content=promoted_markdown, metadata={})]

        # Step 3: Build Semantic Units with Heading-to-Content Binding
        units = self._build_semantic_units(sections)
        if not units:
            return []

        # Step 4: Tier 2 - Cross-Section Greedy Merge adhering to target size (800 - 1500 chars)
        result_chunks: List[Chunk] = []
        global_chunk_index = 0

        curr_units: List[str] = []
        curr_len = 0
        curr_headings: List[str] = []
        curr_metadata: Dict[str, Any] = {}

        for u in units:
            u_text = u["text"].strip()
            if not u_text:
                continue

            u_len = len(u_text)
            candidate_len = curr_len + (2 if curr_units else 0) + u_len

            if candidate_len <= self.max_chunk_size:
                curr_units.append(u_text)
                curr_len = candidate_len
                h_name = u.get("heading", "").strip()
                if h_name and h_name not in curr_headings:
                    curr_headings.append(h_name)
                curr_metadata.update(u.get("metadata", {}))
            else:
                # Max chunk size would be exceeded. Emit current buffer as a chunk!
                if curr_units:
                    chunk_text = "\n\n".join(curr_units).strip()
                    chunk_id = f"{doc_id}_c{global_chunk_index:03d}"
                    char_length = len(chunk_text)
                    word_count = len(chunk_text.split())
                    heading_path = self._build_composite_heading_path(curr_headings, title)

                    chunk_obj = Chunk(
                        chunk_id=chunk_id,
                        chunk_index=global_chunk_index,
                        content=chunk_text,
                        char_length=char_length,
                        doc_id=doc_id,
                        domain=domain,
                        url=url,
                        title=title,
                        heading_path=heading_path,
                        language="vi",
                        metadata={
                            **curr_metadata,
                            "headings": list(curr_headings),
                            "word_count": word_count,
                            "promoted_headings_count": promoted_headings_count,
                            "headings_before": promo_stats.get("headings_before", 0),
                            "headings_after": promo_stats.get("headings_after", 0),
                        }
                    )
                    result_chunks.append(chunk_obj)
                    global_chunk_index += 1

                    # Compute sentence-level overlap from the emitted chunk
                    overlap_prefix = self._extract_sentence_overlap(chunk_text)

                    if overlap_prefix and len(overlap_prefix) + 2 + u_len <= self.max_chunk_size:
                        curr_units = [overlap_prefix, u_text]
                        curr_len = len(overlap_prefix) + 2 + u_len
                    else:
                        curr_units = [u_text]
                        curr_len = u_len

                    h_name = u.get("heading", "").strip()
                    curr_headings = [h_name] if h_name else []
                    curr_metadata = dict(u.get("metadata", {}))
                else:
                    curr_units = [u_text]
                    curr_len = u_len
                    h_name = u.get("heading", "").strip()
                    curr_headings = [h_name] if h_name else []
                    curr_metadata = dict(u.get("metadata", {}))

        # Emit leftover buffer
        if curr_units:
            leftover_text = "\n\n".join(curr_units).strip()
            is_redundant_overlap = (
                result_chunks
                and leftover_text == self._extract_sentence_overlap(result_chunks[-1].content)
            )
            if not is_redundant_overlap and leftover_text:
                chunk_id = f"{doc_id}_c{global_chunk_index:03d}"
                char_length = len(leftover_text)
                word_count = len(leftover_text.split())
                heading_path = self._build_composite_heading_path(curr_headings, title)

                chunk_obj = Chunk(
                    chunk_id=chunk_id,
                    chunk_index=global_chunk_index,
                    content=leftover_text,
                    char_length=char_length,
                    doc_id=doc_id,
                    domain=domain,
                    url=url,
                    title=title,
                    heading_path=heading_path,
                    language="vi",
                    metadata={
                        **curr_metadata,
                        "headings": list(curr_headings),
                        "word_count": word_count,
                        "promoted_headings_count": promoted_headings_count,
                        "headings_before": promo_stats.get("headings_before", 0),
                        "headings_after": promo_stats.get("headings_after", 0),
                    }
                )
                result_chunks.append(chunk_obj)

        return result_chunks
