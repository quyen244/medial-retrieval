"""baotayninh.vn: repair the broken document before extracting.

The site serves 155 KB of perfectly good article HTML wrapped in a document
that closes itself immediately:

    <!DOCTYPE html>
    <html lang="" class="desktop header-fixed layout_style_1">

    </html>

    <head>
    ... 155 KB of real page ...

The first `</html>` lands at byte 76. Every HTML parser - html.parser, lxml,
trafilatura's own - treats everything after it as trailing junk, so the parsed
document has no <body>, no <p>, and no title, and extraction returns nothing
from a page that is entirely present.

Fetching is not the problem here: the response is HTTP 200 with the full
article, under any header set. Deleting that one stray tag is the whole fix.

Measured over four articles: 0 characters before, 1,660-7,906 after.
"""

import httpx

from src.ingestion.extraction import build_document_async
from src.ingestion.fetching import (
    BROWSER_HEADERS, FetchError, classify_failure, fetch_html,
)
from src.ingestion.sites.base import CrawlConfig, CrawlOutcome, Sample, SiteCrawler


CLOSING_HTML = "</html>"
OPENING_HEAD = "<head"


def repair_premature_close(html: str) -> str:
    """Drop a `</html>` that appears before the document's own <head>.

    Deliberately narrow: it only fires when the closing tag precedes <head>,
    which cannot happen in a well-formed document, so a correct page passes
    through untouched.
    """
    close_at = html.find(CLOSING_HTML)
    head_at = html.find(OPENING_HEAD)
    if close_at == -1 or head_at == -1 or close_at > head_at:
        return html
    return html[:close_at] + html[close_at + len(CLOSING_HTML):]


class BaoTayNinhCrawler(SiteCrawler):
    name = "baotayninh"
    domains = ("baotayninh.vn",)

    async def crawl_one(
        self, client: httpx.AsyncClient, sample: Sample, config: CrawlConfig
    ) -> CrawlOutcome:
        try:
            html, final_url, status = await fetch_html(
                client,
                sample.url,
                config.retries,
                policy=config.policy,
                headers=BROWSER_HEADERS,
            )
        except (FetchError, httpx.HTTPError) as exc:
            return CrawlOutcome.failed(
                sample,
                reason=classify_failure(exc),
                detail=f"{type(exc).__name__}: {exc}",
            )

        document = await build_document_async(
            html=repair_premature_close(html),
            doc_id=sample.doc_id,
            domain=sample.domain,
            requested_url=sample.url,
            final_url=final_url,
            status_code=status,
            min_chars=config.min_chars,
            favor_precision=config.favor_precision,
        )

        if document["quality"]["is_empty"]:
            return CrawlOutcome.failed(
                sample,
                reason="empty_extraction",
                detail=f"HTTP {status}, {len(html)} bytes, nothing extracted "
                       "even after repairing the document",
            )

        return CrawlOutcome(sample=sample, document=document)
