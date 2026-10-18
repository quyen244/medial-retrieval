"""Crawl the ViBioMIR corpus into per-domain JSON documents.

Run:
    python -m src.ingestion.crawler --concurrency 16
    python -m src.ingestion.crawler --routes      # show which crawler owns what

Each URL is routed to the crawler that owns its domain (see src/ingestion/sites),
so a site that needs special handling gets it without complicating the rest.
This file owns only the orchestration: how the work is ordered and paced,
where results are stored, and what is written down when something fails.

Pacing lives entirely in the per-host policy (see politeness.py). There is no
global batch barrier: documents flow through a pool of workers that share one
round-robin stream, so one slow host cannot stall the documents queued behind
it. An earlier batch-based design took 3.6 seconds for a batch of 32 when 8 of
them shared a host, against 0.08 seconds when they did not.
"""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion import sites, storage
from src.ingestion.categories import language_of
from src.ingestion.circuit import CircuitBreaker
from src.ingestion.extraction import (
    DEFAULT_MIN_CHARS, shutdown_extraction_pool, start_extraction_pool,
)
from src.ingestion.fetching import build_client
from src.ingestion.politeness import HostLimits, PolitenessPolicy
from src.ingestion.sites import CrawlConfig, CrawlOutcome, Sample, SiteCrawler
from src.ingestion.utils import get_dataset


# Outcomes of processing one sample. Only OK means a usable article was stored.
OK = "ok"
TRIPPED = "tripped"     # host stopped answering; not attempted
SKIPPED = "skipped"        # already on disk from an earlier run
TOO_SHORT = "too_short"    # stored, but probably not a real article
FAILED = "failed"          # nothing usable retrieved; worth trying again later
DEAD = "dead"              # confirmed permanently gone; not stored, never retried
EXCLUDED = "excluded"      # the source was deliberately dropped

# Reasons that mean the document itself will never succeed, regardless of how
# many more times it is fetched: a dead link, a wrong Content-Type, or a page
# that announces its own deletion (see extraction.classify_empty_extraction).
# Letting these flow into the ordinary retry path would mean every future
# crawl invocation re-fetches a corpse. 93 documents hit this in one run
# before the distinction existed.
PERMANENT_REASONS = frozenset({"http_404", "not_html", "deleted_article"})

# Statuses that leave the document still owing us a crawl. EXCLUDED and DEAD
# are not: one is a deliberate policy decision, the other a confirmed fact
# about the document, and retrying either wastes a request for a known answer.
RETRYABLE = (FAILED, TRIPPED)


@dataclass(frozen=True)
class Result:
    """What happened to one sample, for logging and the final summary."""
    status: str
    sample: Sample
    crawler: str = ""
    char_count: int = 0
    reason: str = ""    # short, groupable label
    detail: str = ""    # full message, for debugging one case

    @property
    def needs_retry_later(self) -> bool:
        return self.status in RETRYABLE

    @property
    def belongs_in_ledger(self) -> bool:
        """Worth writing down, whether or not it will be retried.

        DEAD documents are not retried, but they still belong in the ledger:
        it is the record a human reads to see why a domain's count never
        reaches 100%, and it is what lets a later run recognize the same
        dead link without fetching it again.
        """
        return self.status in RETRYABLE or self.status == DEAD

    def to_log_line(self) -> str:
        payload = {
            "status": self.status,
            "doc_id": self.sample.doc_id,
            "domain": self.sample.domain,
            "crawler": self.crawler,
        }
        if self.char_count:
            payload["char_count"] = self.char_count
        if self.reason:
            payload["reason"] = self.reason
        if self.detail:
            payload["detail"] = self.detail
        # ensure_ascii: a Windows console cannot encode Vietnamese titles.
        return json.dumps(payload, ensure_ascii=True)

    def to_failure_record(self) -> dict:
        return {
            "doc_id": self.sample.doc_id,
            "domain": self.sample.domain,
            "url": self.sample.url,
            "crawler": self.crawler,
            "status": self.status,
            "reason": self.reason,
            "detail": self.detail,
            "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
        }


def store(outcome: CrawlOutcome, crawler_name: str, out_dir: Path) -> Result:
    """Write a successful outcome to disk and describe what happened."""
    if not outcome.ok:
        if outcome.reason == "excluded":
            status = EXCLUDED
        elif outcome.reason in PERMANENT_REASONS:
            status = DEAD
        else:
            status = FAILED
        return Result(
            status, outcome.sample, crawler=crawler_name,
            reason=outcome.reason, detail=outcome.detail,
        )

    document = outcome.document
    path = storage.document_path(
        out_dir, outcome.sample.domain, outcome.sample.doc_id
    )
    storage.save_document(path, document)

    return Result(
        TOO_SHORT if document["quality"]["too_short"] else OK,
        outcome.sample,
        crawler=crawler_name,
        char_count=document["content"]["char_count"],
    )


def interleave_by_domain(samples: list[Sample]) -> list[Sample]:
    """Reorder so consecutive samples belong to different domains.

    The corpus is stored grouped by domain: a thousand consecutive rows are
    usually one site, and pubmed alone occupies 240,000 rows in a block. Left
    in that order, every worker in the pool is pointed at the same server at
    the same time, and the per-host limit throttles the entire run to one
    site's pace.

    Round-robin across domains fixes both halves of that: each host sees a
    thin, spaced stream, while the pool stays busy because the hosts in flight
    are all different.
    """
    buckets: dict[str, deque[Sample]] = {}
    for sample in samples:
        buckets.setdefault(sample.domain, deque()).append(sample)

    queues = list(buckets.values())
    ordered: list[Sample] = []
    while queues:
        for queue in queues:
            ordered.append(queue.popleft())
        queues = [queue for queue in queues if queue]
    return ordered


def load_samples(
    limit: int | None,
    interleave: bool = True,
    language: str | None = None,
) -> list[Sample]:
    """Load the corpus rows to crawl.

    `language` narrows the run to one language group, so the three groups can
    be crawled separately - useful because they have different owners,
    different blockers and very different sizes.
    """
    data = get_dataset("corpus")["train"]
    rows = data[:]

    samples = [
        Sample(doc_id, url, storage.domain_of(url))
        for doc_id, url in zip(rows["id"], rows["url"])
    ]
    if language:
        samples = [s for s in samples if language_of(s.domain) == language]

    if interleave:
        samples = interleave_by_domain(samples)

    # Limit last, and after interleaving: the corpus is grouped by domain, so
    # taking the first N rows of it would sample one site rather than the run.
    # Applied after the round-robin, --limit 400 spreads over every domain,
    # which is what makes a short run a useful rehearsal for a long one.
    return samples[:limit] if limit is not None else samples


class Reporter:
    """Collects results: counts them, logs the problems, buffers the backlog.

    The ledger is flushed periodically rather than per document, so a run that
    dies partway through still leaves behind nearly everything it learned
    without paying for a file write per URL.
    """

    def __init__(self, out_dir: Path, total: int, flush_every: int, verbose: bool):
        self.out_dir = out_dir
        self.total = total
        self.flush_every = flush_every
        self.verbose = verbose
        self.tally: Counter = Counter()
        self.done = 0
        self._pending: list[dict] = []

    def record(self, result: Result) -> None:
        self.tally[result.status] += 1
        self.done += 1

        # Successes are counted, not printed: a 1.2M-row run would otherwise
        # bury the problems under a million good lines.
        if self.verbose or result.status in (FAILED, TOO_SHORT):
            print(result.to_log_line())

        if result.belongs_in_ledger:
            self._pending.append(result.to_failure_record())

        if self.done % self.flush_every == 0:
            self.flush()
            self.print_progress()

    def flush(self) -> None:
        if self._pending:
            storage.record_failures(self.out_dir, self._pending)
            self._pending.clear()

    def print_progress(self) -> None:
        print(
            f"[progress] {self.done}/{self.total} "
            + " ".join(f"{name}={count}" for name, count in sorted(self.tally.items())),
            file=sys.stderr,
        )


def partition_by_mode(
    samples: list[Sample],
) -> tuple[dict[SiteCrawler, list[Sample]], list[Sample]]:
    """Split the work into bulk jobs and a per-URL stream.

    A bulk crawler turns many documents into few requests and paces itself, so
    it is handed its whole share at once. Everything else is crawled one URL
    at a time by the worker pool.
    """
    bulk: dict[SiteCrawler, list[Sample]] = {}
    stream: list[Sample] = []

    for sample in samples:
        crawler = sites.crawler_for_domain(sample.domain)
        if crawler.prefers_bulk:
            bulk.setdefault(crawler, []).append(sample)
        else:
            stream.append(sample)

    return bulk, stream


async def crawl(samples: list[Sample], args: argparse.Namespace) -> Counter:
    """Crawl every sample, then report what happened."""
    out_dir = Path(args.out)
    policy = PolitenessPolicy(
        HostLimits(
            max_concurrent=args.per_host_concurrency,
            min_interval=args.per_host_delay,
        )
    )
    config = CrawlConfig(
        retries=args.retries,
        min_chars=args.min_chars,
        favor_precision=args.favor_precision,
        policy=policy,
    )
    active_hosts = set()
    for domain in {sample.domain for sample in samples}:
        active_hosts.update(sites.crawler_for_domain(domain).request_hosts)
    print("Per-host limits:\n" + policy.describe(active_hosts), file=sys.stderr)

    reporter = Reporter(out_dir, len(samples), args.report_every, args.verbose)
    breaker = CircuitBreaker(args.circuit_threshold)
    known_dead = load_known_dead(out_dir, samples)
    if known_dead:
        print(f"Known dead from earlier runs: {len(known_dead)} documents "
              f"will be skipped without a request", file=sys.stderr)
    bulk_jobs, stream = partition_by_mode(samples)
    cursor = iter(stream)   # one shared position, consumed by every worker

    try:
        # The connection pool is sized to the workers; see build_client.
        async with build_client(args.timeout, args.concurrency) as client:
            await asyncio.gather(
                *(
                    _run_bulk(
                        client, crawler, group, config, out_dir, reporter, known_dead
                    )
                    for crawler, group in bulk_jobs.items()
                ),
                *(
                    _worker(
                        client, cursor, config, out_dir, reporter, breaker, known_dead
                    )
                    for _ in range(args.concurrency)
                ),
            )
    finally:
        reporter.flush()
        # Crawlers may hold a session open; the run owns shutting it down.
        await sites.close_all()
        shutdown_extraction_pool()

    reporter.print_progress()
    if breaker.tripped_hosts:
        print("[circuit] hosts dropped this run: "
              + ", ".join(sorted(breaker.tripped_hosts)), file=sys.stderr)
    return reporter.tally


def load_known_dead(
    out_dir: Path, samples: list[Sample]
) -> dict[str, set[int]]:
    """Doc_ids already confirmed dead in an earlier run, per domain.

    Read once at startup rather than per document: the failure ledger for a
    domain is a handful of JSONL lines even on a large crawl, so loading it
    up front is cheap and turns every later lookup into a set membership
    check instead of a file read.
    """
    domains = {sample.domain for sample in samples}
    dead: dict[str, set[int]] = {}
    for domain in domains:
        records = storage.load_failures(out_dir, domain)
        confirmed = {
            doc_id for doc_id, record in records.items()
            if record.get("reason") in PERMANENT_REASONS
        }
        if confirmed:
            dead[domain] = confirmed
    return dead


async def _run_bulk(
    client,
    crawler: SiteCrawler,
    group: list[Sample],
    config: CrawlConfig,
    out_dir: Path,
    reporter: Reporter,
    known_dead: dict[str, set[int]],
) -> None:
    """Hand a bulk crawler its whole share in one call."""
    todo = [
        s for s in group
        if not _already_handled(out_dir, s, reporter, known_dead)
    ]
    if not todo:
        return
    batch_size = crawler.bulk_batch_size or len(todo)
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        for outcome in await crawler.crawl_many(client, batch, config):
            reporter.record(store(outcome, crawler.name, out_dir))


async def _worker(
    client,
    stream: Iterator[Sample],
    config: CrawlConfig,
    out_dir: Path,
    reporter: Reporter,
    breaker: CircuitBreaker,
    known_dead: dict[str, set[int]],
) -> None:
    """Pull samples one at a time until the stream is exhausted.

    Every worker shares one iterator. Advancing it never awaits, so on a
    single-threaded event loop no two workers can take the same sample.

    There is deliberately no batch barrier here. A worker waiting on a slow
    host holds up only itself, and because the stream is ordered round-robin
    across domains, the next sample almost always belongs to a different host.
    """
    for sample in stream:
        if _already_handled(out_dir, sample, reporter, known_dead):
            continue

        if not storage.is_http_url(sample.url):
            reporter.record(Result(
                FAILED, sample, reason="bad_url", detail="not an http(s) URL"
            ))
            continue

        if breaker.is_open(sample.domain):
            reporter.record(Result(
                TRIPPED, sample, reason="host_stopped_answering",
                detail="circuit open after repeated failures; left for a later run",
            ))
            continue

        crawler = sites.crawler_for_domain(sample.domain)
        outcome = await crawler.crawl_one(client, sample, config)

        if outcome.ok:
            breaker.record_success(sample.domain)
        elif outcome.reason != "excluded" and breaker.record_failure(sample.domain):
            print(
                f"[circuit] {sample.domain} stopped answering after "
                f"{breaker.threshold} consecutive failures - skipping the rest",
                file=sys.stderr,
            )

        reporter.record(store(outcome, crawler.name, out_dir))


def _already_handled(
    out_dir: Path,
    sample: Sample,
    reporter: Reporter,
    known_dead: dict[str, set[int]],
) -> bool:
    """True when this sample needs no request: already stored, or known dead."""
    if storage.document_path(out_dir, sample.domain, sample.doc_id).exists():
        reporter.record(Result(SKIPPED, sample))
        return True
    if sample.doc_id in known_dead.get(sample.domain, ()):
        reporter.record(Result(
            DEAD, sample, reason="confirmed_dead",
            detail="matched a permanent failure from an earlier run",
        ))
        return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl corpus URLs into per-domain JSON documents."
    )
    parser.add_argument("--out", default="data/raw_documents",
                        help="Output directory; documents land in <out>/<domain>/")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=150,
                        help="Worker count: how many documents are in flight at once")
    parser.add_argument("--extract-workers", type=int,
                        default=max(2, (os.cpu_count() or 4) - 2),
                        help="Processes used to parse HTML off the event loop")
    parser.add_argument("--circuit-threshold", type=int, default=25,
                        help="Consecutive failures before a host is dropped for the run")
    parser.add_argument("--report-every", type=int, default=500,
                        help="Documents between progress lines and ledger flushes")
    parser.add_argument("--min_chars", type=int, default=DEFAULT_MIN_CHARS,
                        help="Below this a document is stored but flagged too_short")
    parser.add_argument("--favor_precision", action="store_true",
                        help="Trim boilerplate harder, at the risk of losing text")
    parser.add_argument("--per-host-concurrency", type=int, default=6,
                        help="Max simultaneous requests to any single host")
    parser.add_argument("--per-host-delay", type=float, default=0.2,
                        help="Minimum seconds between requests to the same host")
    parser.add_argument("--no-interleave", action="store_true",
                        help="Keep corpus order instead of rotating across domains")
    parser.add_argument("--language", choices=["vi", "zh", "en"],
                        help="Crawl only one language group")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N rows (for testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Log every document, not just the problems")
    parser.add_argument("--routes", action="store_true",
                        help="Print the domain routing table and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.routes:
        print("Domain routing:")
        print(sites.describe_routing())
        return 0

    try:
        samples = load_samples(
            args.limit,
            interleave=not args.no_interleave,
            language=args.language,
        )
        tally = asyncio.run(crawl(samples, args))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("[summary] " + " ".join(f"{k}={v}" for k, v in sorted(tally.items())),
          file=sys.stderr)
    return 0 if tally[FAILED] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
