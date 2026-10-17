"""Separate a client-side ceiling from a server-side one.

The crawl settles around 10 documents per second and nothing in the pipeline
explains it: bandwidth peaks at 41 Mbps, extraction queues for about a second,
saving costs 2 ms, and loosening the per-host limits buys 5%.

This asks the one question that tells the two apart. Drive a *single* fast
host as hard as the policy allows. If one host alone goes far past 10 doc/s,
the client can clearly do more and the multi-host number is the sites' doing.
If one host also stops near 10, the limit is here, not out there.

Run:
    python -m scripts.probe_ceiling --domain suckhoedoisong.vn --count 800
"""

import argparse
import asyncio
import json
import statistics
import time

from src.ingestion import sites
from src.ingestion.extraction import (
    shutdown_extraction_pool, start_extraction_pool,
)
from src.ingestion.fetching import build_client
from src.ingestion.politeness import HostLimits, PolitenessPolicy
from src.ingestion.sites import CrawlConfig, Sample
from src.ingestion.utils import get_dataset


def load_urls(domain: str, count: int) -> list[Sample]:
    """Corpus rows for one domain."""
    from src.ingestion import storage

    rows = get_dataset("corpus")["train"][:]
    out = []
    for doc_id, url in zip(rows["id"], rows["url"]):
        if storage.domain_of(url) == domain:
            out.append(Sample(doc_id, url, domain))
            if len(out) >= count:
                break
    return out


async def run(domain: str, count: int, concurrency: int,
              host_concurrency: int, delay: float, extract_workers: int) -> None:
    samples = load_urls(domain, count)
    start_extraction_pool(extract_workers)

    policy = PolitenessPolicy(
        HostLimits(max_concurrent=host_concurrency, min_interval=delay),
        overrides={domain: HostLimits(host_concurrency, delay)},
    )
    config = CrawlConfig(retries=0, policy=policy)
    crawler = sites.crawler_for_domain(domain)
    semaphore = asyncio.Semaphore(concurrency)

    latencies: list[float] = []
    ok = 0
    failed = 0

    async def one(sample: Sample, client) -> None:
        nonlocal ok, failed
        async with semaphore:
            started = time.monotonic()
            outcome = await crawler.crawl_one(client, sample, config)
            latencies.append(time.monotonic() - started)
            if outcome.ok:
                ok += 1
            else:
                failed += 1

    async with build_client(30, concurrency) as client:
        started = time.monotonic()
        await asyncio.gather(*(one(s, client) for s in samples))
        wall = time.monotonic() - started

    await sites.close_all()
    shutdown_extraction_pool()

    mean = statistics.mean(latencies)
    ordered = sorted(latencies)
    print(f"\n  {domain}  ({crawler.name} crawler)")
    print(f"  {ok} ok, {failed} failed, {wall:.1f}s -> "
          f"[bold]{len(samples)/wall:.1f} doc/s[/bold]".replace("[bold]", "").replace("[/bold]", ""))
    print(f"  latency  mean={mean:.2f}s  p50={ordered[len(ordered)//2]:.2f}s  "
          f"p90={ordered[int(len(ordered)*0.9)]:.2f}s  max={ordered[-1]:.2f}s")
    print(f"  slots busy on average: {sum(latencies)/wall:.0f} of {concurrency}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="suckhoedoisong.vn")
    parser.add_argument("--count", type=int, default=800)
    parser.add_argument("--concurrency", type=int, default=150)
    parser.add_argument("--host-concurrency", type=int, default=16)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--extract-workers", type=int, default=10)
    args = parser.parse_args()

    asyncio.run(run(args.domain, args.count, args.concurrency,
                    args.host_concurrency, args.delay, args.extract_workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
