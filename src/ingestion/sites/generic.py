"""Default crawler: fetch HTML, let trafilatura find the article.

This handles every domain that serves readable HTML without a challenge.
Measured across the 20 largest domains, 12 of them (42.7% of the corpus) come
out clean this way, so this is the path most documents take.
"""

import httpx

from src.ingestion.extraction import build_document_async
from src.ingestion.fetching import FetchError, classify_failure, fetch_html
from src.ingestion.sites.base import CrawlConfig, CrawlOutcome, Sample, SiteCrawler


class GenericCrawler(SiteCrawler):
    name = "generic"
    domains = ()  # the fallback: claims no domain, serves whatever is left

    async def crawl_one(
        self, client: httpx.AsyncClient, sample: Sample, config: CrawlConfig
    ) -> CrawlOutcome:
        try:
            html, final_url, status = await fetch_html(
                client, sample.url, config.retries, policy=config.policy
            )
        except (FetchError, httpx.HTTPError) as exc:
            return CrawlOutcome.failed(
                sample,
                reason=classify_failure(exc),
                detail=f"{type(exc).__name__}: {exc}",
            )

        document = await build_document_async(
            html=html,
            doc_id=sample.doc_id,
            domain=sample.domain,
            requested_url=sample.url,
            final_url=final_url,
            status_code=status,
            min_chars=config.min_chars,
            favor_precision=config.favor_precision,
        )

        if document["quality"]["is_empty"]:
            # HTTP said yes but there is no text. Usually a challenge page
            # rather than a genuinely empty article, so it is worth recording
            # for a retry rather than storing.
            return CrawlOutcome.failed(
                sample,
                reason="empty_extraction",
                detail=f"HTTP {status}, {len(html)} bytes of HTML, no text extracted",
            )

        return CrawlOutcome(sample=sample, document=document)
