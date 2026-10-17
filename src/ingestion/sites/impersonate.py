"""Sites behind Cloudflare that a matching TLS fingerprint gets through.

Cloudflare does not only read headers - it fingerprints the TLS handshake
(JA3) and the HTTP/2 settings. httpx presents Python's fingerprint, which no
browser ever sends, so these sites answer 403 no matter how perfect the header
set is. curl_cffi replays a real Chrome handshake, and the same request then
succeeds.

This replaced a Playwright crawler that drove headless Chromium. Measured on
the same articles:

    Playwright    4.54 s/page, 4/6 succeeded, ~150 MB per page
    curl_cffi     0.4-0.9 s/page, 8/8 succeeded, no browser at all

That is roughly a tenfold speedup and a better success rate, and it takes
nhathuoclongchau.com.vn - 79,598 documents - from the slowest domain in the
corpus to an ordinary one.

A harvested cf_clearance cookie does *not* transfer to httpx: the cookie is
bound to the fingerprint that earned it, so replaying it from a different TLS
stack still returns 403. Impersonation has to happen at the connection level,
which is what this module does.

Not a universal key: zysjonline.com stays 403 under every impersonation
profile tried, and remains in blocked.py.
"""

import asyncio

from curl_cffi.requests import AsyncSession

from src.ingestion.extraction import build_document_async, classify_empty_extraction
from src.ingestion.fetching import detect_block_page
from src.ingestion.sites.base import CrawlConfig, CrawlOutcome, Sample, SiteCrawler


# Chrome build whose handshake is replayed. Worth revisiting when Cloudflare
# starts rejecting it; chrome124 and chrome120 worked identically when tested.
IMPERSONATE_PROFILE = "chrome131"

REQUEST_TIMEOUT = 30


class ImpersonateCrawler(SiteCrawler):
    """Fetches over a connection that looks like Chrome, then extracts as usual."""

    name = "impersonate"
    domains = (
        "nhathuoclongchau.com.vn",
        "tamanhhospital.vn",
    )

    def __init__(self, profile: str = IMPERSONATE_PROFILE):
        self.profile = profile
        self._session: AsyncSession | None = None
        self._startup = asyncio.Lock()

    async def _ensure_session(self) -> AsyncSession:
        """One session per run, so connections are pooled like the httpx client."""
        if self._session is not None:
            return self._session
        async with self._startup:
            if self._session is None:
                self._session = AsyncSession(impersonate=self.profile)
            return self._session

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def crawl_one(
        self, client, sample: Sample, config: CrawlConfig
    ) -> CrawlOutcome:
        session = await self._ensure_session()

        last_error = ""
        for attempt in range(config.retries + 1):
            async with config.policy.request_slot(sample.domain):
                try:
                    response = await session.get(sample.url, timeout=REQUEST_TIMEOUT)
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    response = None

            if response is not None:
                if response.status_code == 200:
                    marker = detect_block_page(response.text)
                    if marker is None:
                        return await self._build(sample, response, config)
                    last_error = f"anti-bot interstitial, matched {marker!r}"
                else:
                    last_error = f"HTTP {response.status_code}"

            if attempt < config.retries:
                await asyncio.sleep(min(2**attempt, 8))

        return CrawlOutcome.failed(
            sample, reason="impersonate_failed", detail=last_error
        )

    async def _build(
        self, sample: Sample, response, config: CrawlConfig
    ) -> CrawlOutcome:
        document = await build_document_async(
            html=response.text,
            doc_id=sample.doc_id,
            domain=sample.domain,
            requested_url=sample.url,
            final_url=str(response.url),
            status_code=response.status_code,
            min_chars=config.min_chars,
            favor_precision=config.favor_precision,
        )

        if document["quality"]["is_empty"]:
            return CrawlOutcome.failed(
                sample,
                reason=classify_empty_extraction(response.text),
                detail=f"HTTP 200, {len(response.text)} bytes, no text extracted",
            )

        return CrawlOutcome(sample=sample, document=document)
