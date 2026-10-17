"""Measure extraction the way the crawler actually calls it.

An earlier benchmark used ProcessPoolExecutor.map with a chunksize, which
hands each worker a batch and amortises the cost of shipping data between
processes. The crawler does not do that: every document is submitted on its
own through run_in_executor, so it pays the full pickle-and-pipe cost per
document, for an HTML body that is 141 KB at the median.

This measures the submit-one-at-a-time path, which is the one that matters.

Run:
    python -m scripts.bench_executor
"""

import asyncio
import json
import time

import httpx

from src.ingestion import extraction
from src.ingestion.fetching import BROWSER_HEADERS


SAMPLE_DOMAINS = [
    "suckhoecongdongonline.vn",
    "suckhoedoisong.vn",
    "thanhnien.vn",
    "www.vinmec.com",
]
DOCS = 120


def fetch_samples() -> list[str]:
    pools = {
        entry["domain"]: entry["pool"]
        for entry in json.load(
            open("data/measure_samples.json", encoding="utf-8")
        )["domains"]
    }
    htmls = []
    with httpx.Client(
        headers=BROWSER_HEADERS, follow_redirects=True, timeout=30
    ) as client:
        for domain in SAMPLE_DOMAINS:
            try:
                htmls.append(client.get(pools[domain][0]["url"]).text)
            except Exception:
                pass
    return htmls


async def measure(htmls: list[str], workers: int) -> float:
    """Throughput when every document is submitted on its own."""
    extraction.shutdown_extraction_pool()
    extraction.start_extraction_pool(workers)

    work = (htmls * ((DOCS // len(htmls)) + 1))[:DOCS]
    kwargs = dict(
        doc_id=1, domain="x.vn", requested_url="u", final_url="u",
        status_code=200,
    )
    # Warm the workers so process startup is not counted.
    await asyncio.gather(*(
        extraction.build_document_async(html=h, **kwargs) for h in htmls
    ))

    started = time.perf_counter()
    await asyncio.gather(*(
        extraction.build_document_async(html=h, **kwargs) for h in work
    ))
    elapsed = time.perf_counter() - started

    extraction.shutdown_extraction_pool()
    return len(work) / elapsed


async def main() -> int:
    htmls = fetch_samples()
    sizes = [len(h) // 1024 for h in htmls]
    print(f"{DOCS} documents, sizes {sizes} KB, submitted one at a time\n")

    started = time.perf_counter()
    work = (htmls * ((DOCS // len(htmls)) + 1))[:DOCS]
    for html in work:
        extraction.build_document(
            html, 1, "x.vn", "u", "u", 200
        )
    inline = len(work) / (time.perf_counter() - started)
    print(f"  inline (no pool)      {inline:6.1f} doc/s")

    for workers in (4, 10, 14):
        rate = await measure(htmls, workers)
        print(f"  run_in_executor x{workers:<3d} {rate:6.1f} doc/s   "
              f"({rate/inline:.2f}x vs inline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
