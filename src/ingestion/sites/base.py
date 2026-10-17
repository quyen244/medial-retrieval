"""The contract every site crawler implements.

One domain, one crawler. The generic crawler handles every site that serves
readable HTML; a site gets its own crawler only when the generic path has been
measured to fail on it, and that crawler encodes exactly what that site needs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx

from src.ingestion.extraction import DEFAULT_MIN_CHARS
from src.ingestion.politeness import PolitenessPolicy


@dataclass(frozen=True)
class Sample:
    """One row of the corpus: the document to crawl."""
    doc_id: int
    url: str
    domain: str


@dataclass(frozen=True)
class CrawlConfig:
    """Settings a crawler may consult. Shared by every site."""
    retries: int = 3
    min_chars: int = DEFAULT_MIN_CHARS
    favor_precision: bool = False
    # Every crawler must route its requests through this, so that no site
    # receives more load than its limits allow.
    policy: PolitenessPolicy = field(default_factory=PolitenessPolicy)


@dataclass
class CrawlOutcome:
    """What a crawler produced for one sample.

    Exactly one of `document` or `reason` is meaningful: a document means the
    article was retrieved, a reason means it was not and says why in a label
    that groups across samples.
    """
    sample: Sample
    document: dict | None = None
    reason: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.document is not None

    @classmethod
    def failed(cls, sample: Sample, reason: str, detail: str = "") -> "CrawlOutcome":
        return cls(sample=sample, reason=reason, detail=detail)


class SiteCrawler(ABC):
    """Base class for per-domain crawlers.

    Subclasses declare the domains they own and implement `crawl_one`. A site
    whose API rewards fetching many documents per request - PubMed's
    E-utilities accepts hundreds of IDs at a time - overrides `crawl_many`
    instead of making one request per URL.
    """

    name: str = "base"
    domains: tuple[str, ...] = ()

    @abstractmethod
    async def crawl_one(
        self, client: httpx.AsyncClient, sample: Sample, config: CrawlConfig
    ) -> CrawlOutcome:
        """Retrieve one document. Must not raise."""

    async def crawl_many(
        self,
        client: httpx.AsyncClient,
        samples: list[Sample],
        config: CrawlConfig,
    ) -> list[CrawlOutcome]:
        """Retrieve several documents.

        The default is one request per sample; the orchestrator supplies the
        concurrency around it. Override when the site offers a bulk endpoint.
        """
        return [await self.crawl_one(client, sample, config) for sample in samples]

    async def aclose(self) -> None:
        """Release anything the crawler holds open.

        Default is a no-op; a crawler that keeps a browser or a connection
        alive across documents overrides this, and the orchestrator calls it
        once the run is finished.
        """

    @property
    def prefers_bulk(self) -> bool:
        """True when crawl_many must be called as a whole rather than per URL.

        The orchestrator uses this to decide whether to spread a domain's
        samples across its worker pool or hand them over in one go.
        """
        return False
