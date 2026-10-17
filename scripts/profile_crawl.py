"""Find which stage of the crawl is the bottleneck.

Runs the real pipeline over a slice of the corpus and reports how much wall
time each stage accounts for, so tuning aims at the stage that actually costs
something rather than the one that looks expensive.

Run:
    python -m scripts.profile_crawl --limit 1200 --concurrency 320
"""

import argparse
import asyncio
import collections
import time
from pathlib import Path

import src.ingestion.extraction as extraction
from src.ingestion import sites, storage
from src.ingestion.crawler import load_samples, store
from src.ingestion.fetching import build_client
from src.ingestion.politeness import PolitenessPolicy
from src.ingestion.sites import CrawlConfig


timings: dict[str, float] = collections.defaultdict(float)
counts: dict[str, int] = collections.defaultdict(int)


def instrument() -> None:
    """Wrap the two stages that are not network, measuring each."""
    real_build = extraction.build_document_async

    async def timed_build(**kwargs):
        started = time.monotonic()
        try:
            return await real_build(**kwargs)
        finally:
            timings["extract"] += time.monotonic() - started
            counts["extract"] += 1

    extraction.build_document_async = timed_build
    for name in ("generic", "laodong", "vnnews", "baotayninh", "impersonate"):
        module = __import__(f"src.ingestion.sites.{name}", fromlist=["x"])
        if hasattr(module, "build_document_async"):
            module.build_document_async = timed_build

    real_save = storage.save_document

    def timed_save(path, document):
        started = time.monotonic()
        try:
            return real_save(path, document)
        finally:
            timings["save"] += time.monotonic() - started
            counts["save"] += 1

    storage.save_document = timed_save


async def run(limit: int, concurrency: int, extract_workers: int) -> None:
    samples = load_samples(limit, interleave=True, language="vi")
    extraction.start_extraction_pool(extract_workers)

    config = CrawlConfig(retries=1, policy=PolitenessPolicy())
    out_dir = Path("data/_profile")
    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    inflight = {"n": 0, "peak": 0}
    samples_log = []

    async def gauge():
        while True:
            await asyncio.sleep(2)
            samples_log.append(inflight["n"])

    async def one(sample, client):
        nonlocal done
        async with semaphore:
            inflight["n"] += 1
            inflight["peak"] = max(inflight["peak"], inflight["n"])
            crawler = sites.crawler_for_domain(sample.domain)
            started = time.monotonic()
            outcome = await crawler.crawl_one(client, sample, config)
            timings["crawl_one"] += time.monotonic() - started
            store(outcome, crawler.name, out_dir)
            inflight["n"] -= 1
            done += 1

    started = time.monotonic()
    g = asyncio.create_task(gauge())
    async with build_client(30, concurrency) as client:
        await asyncio.gather(*(one(s, client) for s in samples))
    wall = time.monotonic() - started
    g.cancel()

    await sites.close_all()
    extraction.shutdown_extraction_pool()

    fetch = timings["crawl_one"] - timings["extract"]
    print(f"\n  {done} documents in {wall:.1f}s = {done/wall:.1f} doc/s "
          f"(concurrency {concurrency}, extract workers {extract_workers})\n")
    print(f"  {'stage':12s} {'total s':>10s} {'per doc':>10s}")
    print("  " + "-" * 34)
    print(f"  {'fetch+gate':12s} {fetch:10.1f} {fetch/max(done,1)*1000:9.0f}ms")
    print(f"  {'extract':12s} {timings['extract']:10.1f} "
          f"{timings['extract']/max(counts['extract'],1)*1000:9.0f}ms")
    print(f"  {'save':12s} {timings['save']:10.1f} "
          f"{timings['save']/max(counts['save'],1)*1000:9.0f}ms")
    print(f"\n  concurrency actually used: "
          f"{timings['crawl_one']/wall:.0f} of {concurrency} slots busy on average")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--concurrency", type=int, default=320)
    parser.add_argument("--extract-workers", type=int, default=6)
    args = parser.parse_args()

    instrument()
    asyncio.run(run(args.limit, args.concurrency, args.extract_workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
