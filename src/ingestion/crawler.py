"""Crawl the ViBioMIR corpus into per-domain JSON documents.

Run:
    python -m src.ingestion.crawler --concurrency 8 --sleep 1

The pipeline is deliberately split across modules: fetching.py decides whether
a response is real content, extraction.py turns HTML into the stored shape,
storage.py decides where it lands. This file only wires them together and
reports what happened.
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.ingestion import storage
from src.ingestion.extraction import DEFAULT_MIN_CHARS, build_document
from src.ingestion.fetching import (
    FetchError, build_client, classify_failure, fetch_html,
)
from src.ingestion.utils import get_dataset


# Outcomes of processing one sample. Only OK means a usable article was stored.
OK = "ok"
SKIPPED = "skipped"          # already on disk from an earlier run
TOO_SHORT = "too_short"      # stored, but almost certainly not an article
EMPTY = "empty"              # extractor found no content; not stored
FAILED = "failed"            # never got usable HTML; not stored


@dataclass(frozen=True)
class Sample:
    """One row of the corpus."""
    doc_id: int
    url: str

    @property
    def domain(self) -> str:
        return storage.domain_of(self.url)


@dataclass(frozen=True)
class Result:
    """What happened to one sample, for logging and the final summary."""
    status: str
    sample: Sample
    char_count: int = 0
    reason: str = ""    # short, groupable label
    detail: str = ""    # full message, for debugging one case

    @property
    def needs_retry_later(self) -> bool:
        """Nothing usable was stored, so this document still owes us a crawl."""
        return self.status in (FAILED, EMPTY)

    def to_log_line(self) -> str:
        payload = {
            "status": self.status,
            "doc_id": self.sample.doc_id,
            "domain": self.sample.domain,
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
            "status": self.status,
            "reason": self.reason,
            "detail": self.detail,
            "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
        }


@dataclass(frozen=True)
class Options:
    """Everything the pipeline needs that is not the sample itself."""
    out_dir: Path
    retries: int
    min_chars: int
    favor_precision: bool


async def process_sample(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    sample: Sample,
    options: Options,
) -> Result:
    """Fetch, extract and store one article. Never raises."""
    if not storage.is_http_url(sample.url):
        return Result(FAILED, sample, reason="bad_url", detail="not an http(s) URL")

    path = storage.document_path(options.out_dir, sample.domain, sample.doc_id)
    if path.exists():
        return Result(SKIPPED, sample)

    async with semaphore:
        try:
            html, final_url, status_code = await fetch_html(
                client, sample.url, options.retries
            )
        except (FetchError, httpx.HTTPError) as exc:
            return Result(
                FAILED, sample,
                reason=classify_failure(exc),
                detail=f"{type(exc).__name__}: {exc}",
            )

    document = build_document(
        html=html,
        doc_id=sample.doc_id,
        domain=sample.domain,
        requested_url=sample.url,
        final_url=final_url,
        status_code=status_code,
        min_chars=options.min_chars,
        favor_precision=options.favor_precision,
    )

    chars = document["content"]["char_count"]
    if document["quality"]["is_empty"]:
        # A 200 with no extractable text is usually a JS challenge rather than
        # a genuinely empty article, so it is worth recording for a retry.
        return Result(
            EMPTY, sample,
            reason="empty_extraction",
            detail=f"HTTP {status_code}, {len(html)} bytes of HTML, no text extracted",
        )

    storage.save_document(path, document)
    status = TOO_SHORT if document["quality"]["too_short"] else OK
    return Result(status, sample, char_count=chars)


def iter_samples(limit: int | None) -> list[Sample]:
    """Load the corpus rows to crawl, in corpus order."""
    data = get_dataset("corpus")["train"]
    total = len(data) if limit is None else min(limit, len(data))
    rows = data[:total]
    return [Sample(doc_id, url) for doc_id, url in zip(rows["id"], rows["url"])]


async def crawl(samples: list[Sample], args: argparse.Namespace) -> Counter:
    """Run the whole corpus in batches, reporting as it goes."""
    options = Options(
        out_dir=Path(args.out),
        retries=args.retries,
        min_chars=args.min_chars,
        favor_precision=args.favor_precision,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    tally: Counter = Counter()

    async with build_client(args.timeout) as client:
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]

            results = await asyncio.gather(
                *(
                    process_sample(client, semaphore, sample, options)
                    for sample in batch
                )
            )

            for result in results:
                tally[result.status] += 1
                # Successes are counted, not printed: a 1.2M-row run would
                # otherwise bury the problems under a million good lines.
                if args.verbose or result.status in (FAILED, EMPTY, TOO_SHORT):
                    print(result.to_log_line())

            # Write the backlog as we go, so a run that dies partway through
            # still leaves behind what the per-domain crawlers will need.
            storage.record_failures(
                options.out_dir,
                [r.to_failure_record() for r in results if r.needs_retry_later],
            )

            done = start + len(batch)
            print(
                f"[progress] {done}/{len(samples)} "
                + " ".join(f"{name}={count}" for name, count in sorted(tally.items())),
                file=sys.stderr,
            )

            if start + args.batch_size < len(samples):
                await asyncio.sleep(args.sleep)

    return tally


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl corpus URLs into per-domain JSON documents."
    )
    parser.add_argument("--out", default="data/raw_documents",
                        help="Output directory; documents land in <out>/<domain>/")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=200,
                        help="Samples dispatched before pausing for --sleep")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Max requests in flight at once")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds to pause between batches")
    parser.add_argument("--min_chars", type=int, default=DEFAULT_MIN_CHARS,
                        help="Below this a document is stored but flagged too_short")
    parser.add_argument("--favor_precision", action="store_true",
                        help="Trim boilerplate harder, at the risk of losing text")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N rows (for testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Log every document, not just the problems")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        samples = iter_samples(args.limit)
        tally = crawl_sync(samples, args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("[summary] " + " ".join(f"{k}={v}" for k, v in sorted(tally.items())),
          file=sys.stderr)
    return 0 if tally[FAILED] == 0 and tally[EMPTY] == 0 else 2


def crawl_sync(samples: list[Sample], args: argparse.Namespace) -> Counter:
    return asyncio.run(crawl(samples, args))


if __name__ == "__main__":
    raise SystemExit(main())
