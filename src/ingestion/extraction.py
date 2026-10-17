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

import asyncio
import hashlib
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from datetime import datetime, timezone
from typing import Any

import trafilatura

from src.ingestion.categories import language_of


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


def assemble_document(
    doc_id: int,
    domain: str,
    requested_url: str,
    final_url: str,
    status_code: int,
    markdown: str,
    title: str = "",
    description: str | None = None,
    published_date: str | None = None,
    language: str | None = None,
    extractor: str = "unknown",
    min_chars: int = DEFAULT_MIN_CHARS,
) -> dict[str, Any]:
    """Wrap already-extracted Markdown in the stored JSON envelope.

    Site crawlers that do not go through HTML - PubMed reads the E-utilities
    XML API, for instance - produce their Markdown their own way and call
    this directly, so every document keeps the same shape.
    """
    heading_count = count_headings(markdown)

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "doc_id": doc_id,
            "domain": domain,
            "requested_url": requested_url,
            "final_url": final_url,
            "http_status": status_code,
            # Declared by the page, often missing or wrong.
            "language": language,
            # Decided by the domain, which is always right. This is the field
            # to filter on when retrieving across languages.
            "language_category": language_of(domain),
            "title": title,
            "description": description,
            "published_date": published_date,
        },
        "content": {
            "markdown": markdown,
            "char_count": len(markdown),
            "heading_count": heading_count,
            "sha256": sha256_text(markdown),
        },
        "ingestion": {
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "extractor": extractor,
            "cleaning_version": CLEANING_VERSION,
            "chunking_version": None,
            "embedding_model": None,
        },
        "quality": {
            "is_empty": not markdown,
            "too_short": len(markdown) < min_chars,
            "has_title": bool(title),
            "has_headings": heading_count > 0,
        },
    }


# --------------------------------------------------------------------------
# Running extraction off the event loop
# --------------------------------------------------------------------------
#
# Extraction is pure CPU - trafilatura parses the HTML and walks the tree -
# and it costs about 70 ms per document. Run inline, it blocks the event loop
# for that whole time, so no request can start or finish while it works. That
# capped the crawler at roughly 9 documents per second no matter how many
# workers or how generous the per-host limits were.
#
# Threads do not help: lxml holds the GIL through parsing, and a thread pool
# measured *slower* than sequential (0.63-0.76x). Processes do, because each
# gets its own interpreter: 19.7 doc/s sequential against 100.3 doc/s across
# 12 processes, a 5x speedup even after paying to ship the HTML between
# processes.

_pool: ProcessPoolExecutor | None = None


def start_extraction_pool(workers: int) -> None:
    """Open the pool extraction runs in. Call once per run."""
    global _pool
    if _pool is None and workers > 0:
        _pool = ProcessPoolExecutor(max_workers=workers)


def shutdown_extraction_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=True)
        _pool = None


async def build_document_async(**kwargs: Any) -> dict[str, Any]:
    """Build a document without blocking the event loop.

    Falls back to building inline when no pool is open, which keeps tests and
    one-off scripts working without any setup.
    """
    if _pool is None:
        return build_document(**kwargs)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_pool, partial(build_document, **kwargs))


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
    """Extract Markdown from an HTML page and wrap it in the JSON envelope."""
    markdown = extract_markdown(html, favor_precision=favor_precision)
    metadata = extract_metadata(html)

    return assemble_document(
        doc_id=doc_id,
        domain=domain,
        requested_url=requested_url,
        final_url=final_url,
        status_code=status_code,
        markdown=markdown,
        title=metadata["title"],
        description=metadata["description"],
        published_date=metadata["published_date"],
        language=detect_language(html),
        extractor=f"trafilatura-{trafilatura.__version__}",
        min_chars=min_chars,
    )
