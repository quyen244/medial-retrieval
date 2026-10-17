"""pubmed.ncbi.nlm.nih.gov: read the E-utilities API, never the web pages.

Scraping the website does not work and should not be attempted. PubMed answers
unknown clients with HTTP 203 and a "Cookies must be enabled" interstitial, and
sustained page requests get the client's IP blocked outright - which is what
happened while measuring this corpus:

    Your access to the NCBI website has been temporarily blocked due to a
    possible misuse/abuse situation [...] learning how to better use
    E-utilities for more efficient work such that your work does not impact
    the ability of other researchers to also use our site.

E-utilities is the supported path for exactly this use. It matters that it is
used the way NCBI asks:

* One request carries up to BATCH_SIZE PMIDs, so 240,000 documents cost about
  1,200 requests rather than 240,000. This is the single most important
  difference between being a good citizen and being blocked again.
* Requests are rate limited - 3/second anonymously, 10/second with an API key.
* Every request identifies the caller via `tool` and `email`.

Set NCBI_API_KEY to raise the rate limit. A key is free from
https://account.ncbi.nlm.nih.gov/settings/ and is strongly recommended for a
crawl of this size.
"""

import asyncio
import os
import re
import time
import xml.etree.ElementTree as ET

import httpx

from src.ingestion.extraction import assemble_document
from src.ingestion.politeness import PolitenessPolicy
from src.ingestion.sites.base import CrawlConfig, CrawlOutcome, Sample, SiteCrawler


EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# NCBI accepts far more, but a few hundred keeps one failure cheap to retry.
BATCH_SIZE = 200

# NCBI's documented ceilings. Exceeding them is what gets clients blocked.
REQUESTS_PER_SECOND_ANONYMOUS = 3
REQUESTS_PER_SECOND_WITH_KEY = 10

TOOL_NAME = "MedicalKBResearch"
CONTACT_EMAIL = os.environ.get("NCBI_CONTACT_EMAIL", "airaceuit2005@gmail.com")

PMID_RE = re.compile(r"/(\d+)/?(?:[?#].*)?$")


def extract_pmid(url: str) -> str | None:
    """PubMed article URLs end in the PMID: .../pubmed/26439093/"""
    match = PMID_RE.search(url)
    return match.group(1) if match else None


class RateLimiter:
    """Allow at most `per_second` operations per second, across all callers."""

    def __init__(self, per_second: int):
        self._min_interval = 1.0 / per_second
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = self._next_allowed - now
            if delay > 0:
                await asyncio.sleep(delay)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval


class PubMedCrawler(SiteCrawler):
    name = "pubmed"
    domains = ("pubmed.ncbi.nlm.nih.gov",)

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key if api_key is not None else os.environ.get("NCBI_API_KEY")
        rate = (
            REQUESTS_PER_SECOND_WITH_KEY if self.api_key
            else REQUESTS_PER_SECOND_ANONYMOUS
        )
        self._limiter = RateLimiter(rate)

    @property
    def prefers_bulk(self) -> bool:
        """One request serves hundreds of documents, so never split the work."""
        return True

    async def crawl_one(
        self, client: httpx.AsyncClient, sample: Sample, config: CrawlConfig
    ) -> CrawlOutcome:
        outcomes = await self.crawl_many(client, [sample], config)
        return outcomes[0]

    async def crawl_many(
        self,
        client: httpx.AsyncClient,
        samples: list[Sample],
        config: CrawlConfig,
    ) -> list[CrawlOutcome]:
        outcomes: list[CrawlOutcome] = []
        fetchable: list[tuple[str, Sample]] = []

        for sample in samples:
            pmid = extract_pmid(sample.url)
            if pmid is None:
                outcomes.append(CrawlOutcome.failed(
                    sample, reason="no_pmid", detail=f"no PMID in {sample.url}"
                ))
            else:
                fetchable.append((pmid, sample))

        for start in range(0, len(fetchable), BATCH_SIZE):
            chunk = fetchable[start : start + BATCH_SIZE]
            outcomes.extend(await self._fetch_chunk(client, chunk, config))

        return outcomes

    async def _fetch_chunk(
        self,
        client: httpx.AsyncClient,
        chunk: list[tuple[str, Sample]],
        config: CrawlConfig,
    ) -> list[CrawlOutcome]:
        """One efetch call covering every PMID in the chunk."""
        by_pmid = {pmid: sample for pmid, sample in chunk}

        try:
            xml = await self._request(
                client, list(by_pmid), config.retries, config.policy
            )
        except (httpx.HTTPError, ValueError) as exc:
            return [
                CrawlOutcome.failed(
                    sample, reason="efetch_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                for sample in by_pmid.values()
            ]

        articles = parse_article_set(xml)

        outcomes = []
        for pmid, sample in by_pmid.items():
            article = articles.get(pmid)
            if article is None:
                outcomes.append(CrawlOutcome.failed(
                    sample, reason="pmid_not_returned",
                    detail=f"PMID {pmid} missing from the efetch response",
                ))
                continue

            markdown = render_markdown(article)
            if not markdown:
                outcomes.append(CrawlOutcome.failed(
                    sample, reason="no_abstract",
                    detail=f"PMID {pmid} has a record but no abstract text",
                ))
                continue

            outcomes.append(CrawlOutcome(
                sample=sample,
                document=assemble_document(
                    doc_id=sample.doc_id,
                    domain=sample.domain,
                    requested_url=sample.url,
                    final_url=sample.url,
                    status_code=200,
                    markdown=markdown,
                    title=article["title"],
                    description=None,
                    published_date=article["year"],
                    language=article["language"],
                    extractor="ncbi-eutils-efetch",
                    min_chars=config.min_chars,
                ),
            ))

        return outcomes

    async def _request(
        self,
        client: httpx.AsyncClient,
        pmids: list[str],
        retries: int,
        policy: PolitenessPolicy,
    ) -> str:
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "tool": TOOL_NAME,
            "email": CONTACT_EMAIL,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            await self._limiter.wait()
            try:
                # Two gates on purpose: the limiter above enforces NCBI's
                # documented per-second ceiling, the slot below is the
                # pipeline-wide guarantee that no host is hit concurrently.
                async with policy.request_slot("eutils.ncbi.nlm.nih.gov"):
                    response = await client.post(EFETCH_URL, data=params)
                response.raise_for_status()
                text = response.text
                if "<PubmedArticleSet" not in text:
                    # The abuse page is HTML, not the XML we asked for.
                    raise ValueError(
                        "efetch did not return a PubmedArticleSet "
                        "(NCBI may have blocked this client)"
                    )
                return text
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt < retries:
                    await asyncio.sleep(min(2**attempt, 8))

        raise ValueError(f"efetch failed after {retries + 1} attempts: {last_error}")


# --------------------------------------------------------------------------
# XML -> Markdown
# --------------------------------------------------------------------------

def _text_of(node: ET.Element | None) -> str:
    """All text inside a node, including nested markup such as <i> or <sup>."""
    if node is None:
        return ""
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()


def parse_article_set(xml: str) -> dict[str, dict]:
    """Parse an efetch response into {pmid: article fields}."""
    root = ET.fromstring(xml)
    articles: dict[str, dict] = {}

    for entry in root.findall(".//PubmedArticle"):
        citation = entry.find("MedlineCitation")
        if citation is None:
            continue

        pmid = _text_of(citation.find("PMID"))
        if not pmid:
            continue

        article = citation.find("Article")
        if article is None:
            continue

        # Abstract sections keep their labels (BACKGROUND, METHODS, ...)
        # because they are real structure the chunker can use.
        sections = [
            {
                "label": (part.get("Label") or "").strip(),
                "text": _text_of(part),
            }
            for part in article.findall(".//Abstract/AbstractText")
        ]

        authors = [
            " ".join(filter(None, [
                _text_of(author.find("ForeName")),
                _text_of(author.find("LastName")),
            ])).strip()
            for author in article.findall(".//AuthorList/Author")
        ]

        articles[pmid] = {
            "pmid": pmid,
            "title": _text_of(article.find("ArticleTitle")),
            "journal": _text_of(article.find(".//Journal/Title")),
            "year": _text_of(article.find(".//Journal/JournalIssue/PubDate/Year")) or None,
            "language": _text_of(article.find("Language")) or None,
            "authors": [name for name in authors if name],
            "sections": [s for s in sections if s["text"]],
            "keywords": [
                _text_of(kw) for kw in citation.findall(".//KeywordList/Keyword")
            ],
        }

    return articles


def render_markdown(article: dict) -> str:
    """Render one article record as Markdown with real headings.

    Unlike the scraped sites, PubMed records carry genuine structure, so the
    output has `#` headings a header-aware chunker can split on.
    """
    if not article["sections"]:
        return ""

    lines: list[str] = []
    if article["title"]:
        lines.append(f"# {article['title']}")
        lines.append("")

    lines.append("## Abstract")
    lines.append("")
    for section in article["sections"]:
        if section["label"]:
            lines.append(f"### {section['label'].title()}")
            lines.append("")
        lines.append(section["text"])
        lines.append("")

    citation_bits = [bit for bit in (article["journal"], article["year"]) if bit]
    if citation_bits or article["authors"]:
        lines.append("## Citation")
        lines.append("")
        if article["authors"]:
            shown = ", ".join(article["authors"][:10])
            if len(article["authors"]) > 10:
                shown += ", et al."
            lines.append(shown)
        if citation_bits:
            lines.append(". ".join(citation_bits) + ".")
        lines.append(f"PMID: {article['pmid']}")
        lines.append("")

    if article["keywords"]:
        lines.append("## Keywords")
        lines.append("")
        lines.append(", ".join(kw for kw in article["keywords"] if kw))

    return "\n".join(lines).strip()
