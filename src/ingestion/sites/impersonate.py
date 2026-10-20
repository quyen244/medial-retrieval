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

nhathuoclongchau.com.vn turned out to rate-limit by session, not by IP: a
single AsyncSession reused for the whole run got 429'd after roughly 500-600
requests, while a brand new session against the same IP, same minute, same
URLs, sailed through with clean 200s every time. One shared session for a run
of 80,000 requests looks nothing like real traffic - every genuine visitor
is one browser tab with its own cookies, not one client making tens of
thousands of sequential requests. Rotating to a fresh session well before the
observed failure threshold keeps every session looking like an ordinary
visit instead of the thing that gets flagged.
"""

import asyncio

from curl_cffi.requests import AsyncSession

from src.ingestion.extraction import build_document_async, classify_empty_extraction
from src.ingestion.fetching import detect_block_page
from src.ingestion.sites.base import CrawlConfig, CrawlOutcome, Sample, SiteCrawler


# Chrome build whose handshake is replayed. Worth revisiting when Cloudflare
# starts rejecting it; chrome124 and chrome120 worked identically when tested.
IMPERSONATE_PROFILE = "chrome131"

# Observed failure point for one session was ~500-600 requests. Rotating at
# half that leaves margin without re-handshaking so often it adds overhead.
ROTATE_SESSION_AFTER = 250

REQUEST_TIMEOUT = 30

# Captured verbatim when present on a non-200 response. retry-after is the one
# that matters most - it is the server naming its own cooldown, which beats
# any pacing number we guess from the outside. The rest are context for
# whichever rate-limit scheme the site happens to use.
RATE_LIMIT_HEADERS = ("retry-after", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset")


class ImpersonateCrawler(SiteCrawler):
    """Fetches over a connection that looks like Chrome, then extracts as usual."""

    name = "impersonate"
    domains = (
        "nhathuoclongchau.com.vn",
        "tamanhhospital.vn",
    )

    def __init__(self, profile: str = IMPERSONATE_PROFILE):
        self.profile = profile
        self._sessions: dict[str, AsyncSession] = {}
        self._request_counts: dict[str, int] = {}
        self._startup = asyncio.Lock()

    async def _ensure_session(self, domain: str, force_new: bool = False) -> AsyncSession:
        """This domain's current session, rotating to a fresh one when it is
        either stale (ROTATE_SESSION_AFTER requests old) or just got a
        blocked/rate-limited response (force_new) - see module docstring for
        why reuse past that point gets flagged.
        """
        stale = force_new or self._request_counts.get(domain, 0) >= ROTATE_SESSION_AFTER
        if domain in self._sessions and not stale:
            return self._sessions[domain]
        async with self._startup:
            stale = force_new or self._request_counts.get(domain, 0) >= ROTATE_SESSION_AFTER
            if domain not in self._sessions or stale:
                old = self._sessions.get(domain)
                if old is not None:
                    await old.close()
                self._sessions[domain] = AsyncSession(impersonate=self.profile)
                self._request_counts[domain] = 0
            return self._sessions[domain]

    async def aclose(self) -> None:
        for session in self._sessions.values():
            await session.close()
        self._sessions.clear()
        self._request_counts.clear()

    async def crawl_one(
        self, client, sample: Sample, config: CrawlConfig
    ) -> CrawlOutcome:
        domain = sample.domain
        session = await self._ensure_session(domain)

        last_error = ""
        for attempt in range(config.retries + 1):
            self._request_counts[domain] = self._request_counts.get(domain, 0) + 1
            async with config.policy.request_slot(domain):
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
                    headers = {
                        h: response.headers.get(h)
                        for h in RATE_LIMIT_HEADERS
                        if response.headers.get(h)
                    }
                    suffix = f" headers={headers}" if headers else ""
                    last_error = (
                        f"HTTP {response.status_code} "
                        f"(attempt {attempt + 1}/{config.retries + 1}){suffix}"
                    )
                    if response.status_code in (403, 429):
                        # This session is the thing that got flagged, not
                        # this URL - retrying through it again would just
                        # confirm the same block. Burn it for a clean one.
                        session = await self._ensure_session(domain, force_new=True)

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
