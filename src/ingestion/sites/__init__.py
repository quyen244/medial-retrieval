"""Dispatcher: pick the crawler that owns a URL.

Registering a new site is one import and one entry in SITE_CRAWLERS. Anything
not registered falls through to the generic HTML crawler, so a domain only
needs code once it has been measured to need it.
"""

from urllib.parse import urlparse

from src.ingestion.sites.base import (
    CrawlConfig, CrawlOutcome, Sample, SiteCrawler,
)
from src.ingestion.sites.baotayninh import BaoTayNinhCrawler
from src.ingestion.sites.blocked import BlockedSiteCrawler
from src.ingestion.sites.impersonate import ImpersonateCrawler
from src.ingestion.sites.excluded import ExcludedSiteCrawler
from src.ingestion.sites.generic import GenericCrawler
from src.ingestion.sites.laodong import LaodongCrawler
from src.ingestion.sites.pubmed import PubMedCrawler
from src.ingestion.sites.vnnews import VietnameseNewsCrawler


__all__ = [
    "CrawlConfig", "CrawlOutcome", "Sample", "SiteCrawler",
    "crawler_for", "crawler_for_domain", "registered_crawlers",
    "describe_routing", "close_all",
]


# One instance per site, created once and reused: a crawler may hold state
# that must be shared across calls, such as PubMed's rate limiter.
SITE_CRAWLERS: tuple[SiteCrawler, ...] = (
    PubMedCrawler(),
    LaodongCrawler(),
    VietnameseNewsCrawler(),
    BaoTayNinhCrawler(),
    ImpersonateCrawler(),
    BlockedSiteCrawler(),
    ExcludedSiteCrawler(),
)

FALLBACK_CRAWLER: SiteCrawler = GenericCrawler()

# domain -> crawler, built once from each crawler's declared domains.
_BY_DOMAIN: dict[str, SiteCrawler] = {
    domain: crawler
    for crawler in SITE_CRAWLERS
    for domain in crawler.domains
}


def crawler_for_domain(domain: str) -> SiteCrawler:
    """The crawler that owns a domain, or the generic one."""
    domain = domain.lower()
    if domain in _BY_DOMAIN:
        return _BY_DOMAIN[domain]

    # A registered domain also owns its subdomains: a crawler for "laodong.vn"
    # handles "amp.laodong.vn" unless something more specific is registered.
    for registered, crawler in _BY_DOMAIN.items():
        if domain.endswith("." + registered):
            return crawler

    return FALLBACK_CRAWLER


def crawler_for(url: str) -> SiteCrawler:
    """The crawler that owns a URL."""
    return crawler_for_domain(urlparse(url).netloc.split("@")[-1].split(":")[0])


def registered_crawlers() -> tuple[SiteCrawler, ...]:
    return SITE_CRAWLERS + (FALLBACK_CRAWLER,)


async def close_all() -> None:
    """Release resources held by any crawler, once a run is over."""
    for crawler in registered_crawlers():
        await crawler.aclose()


def describe_routing() -> str:
    """Human-readable routing table, for startup logs and debugging."""
    lines = []
    for crawler in SITE_CRAWLERS:
        mode = "bulk" if crawler.prefers_bulk else "per-url"
        lines.append(f"  {crawler.name:10s} [{mode:7s}] {', '.join(crawler.domains)}")
    lines.append(f"  {FALLBACK_CRAWLER.name:10s} [per-url] everything else")
    return "\n".join(lines)
