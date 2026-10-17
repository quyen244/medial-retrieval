"""laodong.vn: answer the JavaScript cookie challenge, then read normally.

The site replies to an unknown client with HTTP 200 and a 177-byte body that
contains no article at all:

    <html><body><script>document.cookie="D1N=1e4880..."+"; expires=...";
    window.location.reload(true);</script></body></html>

A browser runs that script, stores the cookie and reloads; a plain HTTP client
sees a successful but empty page. The challenge hands us the cookie value in
the markup, so answering it is a matter of reading the value, storing it, and
asking again - no JavaScript engine required.

Measured: the retry turns 177 bytes into ~104 KB of HTML and ~2,100 characters
of Markdown. Once the cookie is in the client's jar it is reused for every
later request to the site, so the challenge is normally paid only once.
"""

import re

import httpx

from src.ingestion.extraction import build_document_async
from src.ingestion.fetching import (
    FetchError, classify_failure, fetch_html, polite_get, validate_response,
)
from src.ingestion.sites.base import CrawlConfig, CrawlOutcome, Sample, SiteCrawler


# document.cookie="NAME=VALUE"+"; expires=..."
COOKIE_CHALLENGE_RE = re.compile(
    r'document\.cookie\s*=\s*["\']([A-Za-z0-9_-]+)=([^"\';]+)'
)

# The challenge page is tiny; a real article never is.
CHALLENGE_MAX_BYTES = 2_000


class LaodongCrawler(SiteCrawler):
    name = "laodong"
    domains = ("laodong.vn",)

    async def crawl_one(
        self, client: httpx.AsyncClient, sample: Sample, config: CrawlConfig
    ) -> CrawlOutcome:
        try:
            html, final_url, status = await self._fetch_past_challenge(
                client, sample.url, config, 
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
                reason="challenge_unsolved",
                detail=f"still empty after answering the challenge ({len(html)} bytes)",
            )

        return CrawlOutcome(sample=sample, document=document)

    async def _fetch_past_challenge(
        self, client: httpx.AsyncClient, url: str, config: CrawlConfig
    ) -> tuple[str, str, int]:
        """Fetch the article, answering the cookie challenge if one is served."""
        html, final_url, status = await fetch_html(
            client, url, config.retries, policy=config.policy
        )

        challenge = self._read_challenge(html)
        if challenge is None:
            return html, final_url, status

        name, value = challenge
        client.cookies.set(name, value, domain=".laodong.vn")

        # The retry goes through the same gate, so answering a challenge
        # never doubles the rate the site sees.
        response = await polite_get(client, url, config.policy)
        html = validate_response(response)
        return html, str(response.url), response.status_code

    @staticmethod
    def _read_challenge(html: str) -> tuple[str, str] | None:
        """Return the (name, value) cookie the page is asking us to set."""
        if len(html) > CHALLENGE_MAX_BYTES:
            return None
        match = COOKIE_CHALLENGE_RE.search(html)
        return (match.group(1), match.group(2)) if match else None
