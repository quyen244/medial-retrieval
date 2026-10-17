"""Vietnamese provincial news sites that demand a full browser header set.

baohaiphong.vn, baonghean.vn, baodanang.vn, hanoimoi.vn and
mega.vietnamplus.vn all sit behind the same kind of edge rule. It does not inspect the User-Agent alone - sending a
browser UA with a short header list still gets 403. What it checks for is the
header set a real Chrome sends: Sec-Fetch-*, sec-ch-ua and
Upgrade-Insecure-Requests.

Measured, one article per site:

    bot header set      403, 284 bytes ("Your request has been blocked")
    full browser set    200, 72-123 KB of real article HTML

hanoimoi.vn and mega.vietnamplus.vn were found later to answer the same way:
403 to the bot header set, 200 with 5,276 and 7,366 characters of extractable
Markdown to the browser set.

Beyond the headers these sites are ordinary, so extraction is the generic
trafilatura path.
"""

import httpx

from src.ingestion.extraction import build_document_async
from src.ingestion.fetching import (
    BROWSER_HEADERS, FetchError, classify_failure, fetch_html,
)
from src.ingestion.sites.base import CrawlConfig, CrawlOutcome, Sample, SiteCrawler


class VietnameseNewsCrawler(SiteCrawler):
    name = "vnnews"
    domains = (
        "baohaiphong.vn",
        "baonghean.vn",
        "baodanang.vn",
        "hanoimoi.vn",
        "mega.vietnamplus.vn",
    )

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
            return CrawlOutcome.failed(
                sample,
                reason="empty_extraction",
                detail=f"HTTP {status}, {len(html)} bytes, no text extracted",
            )

        return CrawlOutcome(sample=sample, document=document)
