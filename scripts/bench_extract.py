"""Measure how fast documents can be extracted, and with what parallelism.

Extraction is pure CPU: trafilatura parses the HTML and walks the tree. That
work happens inside the crawler's event loop, so while it runs, no request can
start or finish. If it is the binding constraint, no amount of concurrency or
per-host tuning moves the run time.

Run:
    python -m scripts.bench_extract
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count

import httpx

from src.ingestion.extraction import build_document
from src.ingestion.fetching import BROWSER_HEADERS


SAMPLE_DOMAINS = [
    "suckhoecongdongonline.vn",
    "suckhoedoisong.vn",
    "thanhnien.vn",
    "www.vinmec.com",
]
REPEATS = 24


def extract_one(html: str) -> int:
    """Top level so a spawned worker process can import it."""
    return build_document(html, 1, "x.vn", "u", "u", 200)["content"]["char_count"]


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


def main() -> int:
    htmls = fetch_samples()
    work = htmls * REPEATS
    print(f"CPU cores: {cpu_count()} | {len(work)} documents | "
          f"sizes {[len(h) // 1024 for h in htmls]} KB\n")

    start = time.perf_counter()
    for html in work:
        extract_one(html)
    sequential = time.perf_counter() - start
    print(f"  sequential          {sequential:6.2f}s -> "
          f"{len(work)/sequential:6.1f} doc/s   (what the crawler does today)")

    for workers in (4, 8):
        with ThreadPoolExecutor(workers) as pool:
            start = time.perf_counter()
            list(pool.map(extract_one, work))
            elapsed = time.perf_counter() - start
        print(f"  threads  x{workers:<2d}        {elapsed:6.2f}s -> "
              f"{len(work)/elapsed:6.1f} doc/s   ({sequential/elapsed:.2f}x)")

    for workers in (4, 8, 12):
        with ProcessPoolExecutor(workers) as pool:
            list(pool.map(extract_one, work[:workers]))   # warm the workers
            start = time.perf_counter()
            list(pool.map(extract_one, work, chunksize=4))
            elapsed = time.perf_counter() - start
        print(f"  processes x{workers:<2d}       {elapsed:6.2f}s -> "
              f"{len(work)/elapsed:6.1f} doc/s   ({sequential/elapsed:.2f}x)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
