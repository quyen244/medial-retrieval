"""Sites that are known to refuse this client, recorded rather than retried.

Each of these was investigated and the reason identified. None of them fail
for a reason more headers or more patience would fix, so attempting them is
pure cost: at three retries each, the three domains here would spend roughly
800,000 requests to learn what is already written down below.

A domain stays here until the thing that blocks it changes - a proxy with a
different geography, a headless browser that can answer a JS challenge, or
the site coming back online. When that happens, give it a real crawler and
replay its backlog from data/raw_documents/_failures/<domain>.jsonl.
"""

import httpx

from src.ingestion.sites.base import CrawlConfig, CrawlOutcome, Sample, SiteCrawler


# domain -> (short reason code, what was observed, what would be needed)
KNOWN_BLOCKS: dict[str, tuple[str, str, str]] = {
    "www.120ask.com": (
        "origin_down",
        "nginx returns HTTP 521 with an empty body for every path, including "
        "the site root, over both http and https. The mobile host m.120ask.com "
        "is alive, but it indexes articles by an opaque id "
        "(/nk/post/0hc4fltyawkim7pf.html) and has no route that accepts the "
        "numeric question ids this corpus stores. The Wayback Machine has no "
        "snapshot of the sampled URLs",
        "the origin to come back, or a mapping from numeric question id to the "
        "mobile site's opaque id - no such mapping was found",
    ),
    "zysjonline.com": (
        "http_403",
        "403 on every sampled article under both header sets; headless "
        "Chromium stays on Cloudflare's interstitial ('Chờ một chút...') for "
        "25 seconds without clearing it",
        "a headful browser, a residential IP, or the site's permission",
    ),
}


class BlockedSiteCrawler(SiteCrawler):
    """Fails immediately, with the investigated reason attached.

    Returning a failure rather than raising keeps these documents flowing into
    the normal ledger, so the backlog stays complete and a later run can pick
    them up without re-deriving why they were skipped.
    """

    name = "blocked"
    domains = tuple(KNOWN_BLOCKS)

    async def crawl_one(
        self, client: httpx.AsyncClient, sample: Sample, config: CrawlConfig
    ) -> CrawlOutcome:
        reason, observed, needed = KNOWN_BLOCKS.get(
            sample.domain,
            ("blocked", "no detail recorded", "investigation"),
        )
        return CrawlOutcome.failed(
            sample,
            reason=reason,
            detail=f"known block: {observed}. Needs: {needed}",
        )
