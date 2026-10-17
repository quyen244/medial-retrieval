"""Sources deliberately dropped from the corpus.

These are not failures and they are not a backlog. They were looked at, and a
decision was made not to pursue them, so they are skipped without a request
and without an entry in the retry ledger. Keeping that decision in code -
rather than as something everyone has to remember - means a later run does
not quietly start attempting them again.

Removing a domain from here puts it straight back into the crawl.
"""

import httpx

from src.ingestion.sites.base import CrawlConfig, CrawlOutcome, Sample, SiteCrawler


# domain -> why it was dropped
EXCLUDED_DOMAINS: dict[str, str] = {
    # Behind an Alibaba ESA rule that only an IP inside China would clear.
    # Standing up a China proxy is not worth 2.2% of the corpus.
    "www.familydoctor.com.cn": "needs a China-based IP; not worth the proxy",
    "baidianfeng.familydoctor.com.cn": "needs a China-based IP; not worth the proxy",
    "qy.familydoctor.com.cn": "needs a China-based IP; not worth the proxy",
    "ask.familydoctor.com.cn": "needs a China-based IP; not worth the proxy",
    "www.familydoctor.cn": "same publisher as familydoctor.com.cn",

    # Origin returns 521 for everything; 734 documents between them.
    "article.iiyi.com": "origin down, too few documents to chase",
    "bingli.iiyi.com": "origin down, too few documents to chase",

    # A retired staging host. The articles may exist on a production domain,
    # but that domain is not in this corpus.
    "pmc-ecm-healthblog.beta.pharmacity.io": "retired staging host",
}


class ExcludedSiteCrawler(SiteCrawler):
    """Reports a domain as out of scope without touching the network."""

    name = "excluded"
    domains = tuple(EXCLUDED_DOMAINS)

    async def crawl_one(
        self, client: httpx.AsyncClient, sample: Sample, config: CrawlConfig
    ) -> CrawlOutcome:
        return CrawlOutcome.failed(
            sample,
            reason="excluded",
            detail=EXCLUDED_DOMAINS.get(sample.domain, "dropped from the corpus"),
        )
