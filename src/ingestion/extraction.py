"""Turn fetched HTML into the stored document shape.

The stored shape is JSON wrapping Markdown. The JSON envelope carries
provenance - where the text came from, when, and how it was produced - which
is what makes dedup, quality review and incremental reindexing possible. The
Markdown body carries the article itself, in the form a header-aware chunker
can split.

Main-content detection is delegated to trafilatura, which scores nodes by text
density and link density. The previous hand-rolled approach looked for
semantic HTML5 tags and fell back to <body> when it found none, which on sites
that use <div class="header"> meant keeping the whole page chrome.
"""

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

import trafilatura


SCHEMA_VERSION = "2.0"
CLEANING_VERSION = "clean-v002"

# Shorter than this is almost never a usable article.
DEFAULT_MIN_CHARS = 300

_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_HTML_LANG_RE = re.compile(r"<html[^>]*\blang=[\"']([^\"']+)[\"']", re.IGNORECASE)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def count_headings(markdown: str) -> int:
    """Number of real Markdown headings.

    Worth tracking per document: many sites mark their headings with <strong>
    instead of <h2>, and those pages reach the chunker with nothing to split
    on. The chunker uses this to choose between header-aware and recursive
    splitting.
    """
    return len(_HEADING_RE.findall(markdown))


def detect_language(html: str) -> str | None:
    """Language from the <html lang> attribute, which trafilatura does not report."""
    match = _HTML_LANG_RE.search(html[:4000])
    return match.group(1).strip() if match else None


def extract_markdown(html: str, favor_precision: bool = False) -> str:
    """Main-content Markdown, with navigation, footer and related links removed.

    favor_precision trims more aggressively - it drops more boilerplate but
    also risks dropping real paragraphs on sites with unusual markup.
    """
    markdown = trafilatura.extract(
        html,
        output_format="markdown",
        include_tables=True,
        include_comments=False,
        favor_precision=favor_precision,
    )
    return (markdown or "").strip()


def extract_metadata(html: str) -> dict[str, Any]:
    """Title, description and publication date, as far as the page declares them."""
    metadata = trafilatura.extract_metadata(html)
    if metadata is None:
        return {"title": "", "description": None, "published_date": None}

    fields = metadata.as_dict()
    return {
        "title": (fields.get("title") or "").strip(),
        "description": (fields.get("description") or "").strip() or None,
        "published_date": fields.get("date") or None,
    }


def build_document(
    html: str,
    doc_id: int,
    domain: str,
    requested_url: str,
    final_url: str,
    status_code: int,
    min_chars: int = DEFAULT_MIN_CHARS,
    favor_precision: bool = False,
) -> dict[str, Any]:
    """Assemble the JSON envelope around the extracted Markdown."""
    markdown = extract_markdown(html, favor_precision=favor_precision)
    metadata = extract_metadata(html)
    heading_count = count_headings(markdown)

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "doc_id": doc_id,
            "domain": domain,
            "requested_url": requested_url,
            "final_url": final_url,
            "http_status": status_code,
            "language": detect_language(html),
            "title": metadata["title"],
            "description": metadata["description"],
            "published_date": metadata["published_date"],
        },
        "content": {
            "markdown": markdown,
            "char_count": len(markdown),
            "heading_count": heading_count,
            "sha256": sha256_text(markdown),
        },
        "ingestion": {
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "extractor": f"trafilatura-{trafilatura.__version__}",
            "cleaning_version": CLEANING_VERSION,
            "chunking_version": None,
            "embedding_model": None,
        },
        "quality": {
            "is_empty": not markdown,
            "too_short": len(markdown) < min_chars,
            "has_title": bool(metadata["title"]),
            "has_headings": heading_count > 0,
        },
    }
