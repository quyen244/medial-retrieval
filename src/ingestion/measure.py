"""Measure extraction quality per domain, before committing to a crawl.

Run:
    python -m src.ingestion.measure --domains 20 --per-domain 10

Samples a handful of real articles from each of the largest domains, runs the
same fetch and extraction path the crawler uses, and reports where the generic
approach holds up and where it does not. The point is to find the domains that
need their own handling *before* spending 1.2M requests finding out.

Nothing is written to the document store; the raw per-page measurements are
saved as JSON so a later run can be compared against this one.
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TextColumn, TimeElapsedColumn,
)
from rich.table import Table

from src.ingestion import sites, storage
from src.ingestion.extraction import (
    DEFAULT_MIN_CHARS, count_headings, extract_markdown, extract_metadata,
)
from src.ingestion.fetching import (
    FetchError, build_client, classify_failure, fetch_html,
)
from src.ingestion.politeness import HostLimits, PolitenessPolicy
from src.ingestion.utils import get_dataset


SAMPLE_CACHE = Path("data/measure_samples.json")
REPORT_PATH = Path("data/extraction_report.json")

# Verdicts, worst first. Each names the single thing wrong with a domain.
BLOCKED = "BLOCKED"            # nothing fetchable at all
UNSTABLE = "UNSTABLE"          # some requests fail
THIN = "THIN"                  # fetches fine, extracts too little
HEALTHY = "HEALTHY"

VERDICT_STYLE = {
    BLOCKED: "bold red",
    UNSTABLE: "bold yellow",
    THIN: "bold magenta",
    HEALTHY: "bold green",
}

console = Console()


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    doc_id: int
    url: str
    domain: str


# Scanning 1.2M rows takes about a minute, so one scan caches a pool large
# enough to serve any reasonable --per-domain without rescanning.
SAMPLE_POOL_PER_DOMAIN = 50


def _scan_corpus(seed: int) -> list[dict]:
    """One pass over the corpus: a random pool of rows for every domain.

    Rows are sampled randomly rather than taken from the front, because the
    corpus is grouped by domain and consecutive rows tend to come from the
    same section of a site.
    """
    console.print("[dim]Scanning corpus to group rows by domain (one-off)...[/dim]")
    rows = get_dataset("corpus")["train"][:]

    by_domain: dict[str, list[tuple[int, str]]] = {}
    for doc_id, url in zip(rows["id"], rows["url"]):
        by_domain.setdefault(storage.domain_of(url), []).append((doc_id, url))

    rng = random.Random(seed)
    pools = [
        {
            "domain": domain,
            "count": len(entries),
            "pool": [
                {"doc_id": doc_id, "url": url}
                for doc_id, url in rng.sample(
                    entries, min(SAMPLE_POOL_PER_DOMAIN, len(entries))
                )
            ],
        }
        for domain, entries in by_domain.items()
    ]
    pools.sort(key=lambda row: row["count"], reverse=True)
    return pools


def select_samples(top_domains: int, per_domain: int, seed: int) -> list[Sample]:
    """Pick `per_domain` random articles from each of the largest domains."""
    pools: list[dict] | None = None

    if SAMPLE_CACHE.exists():
        cached = json.loads(SAMPLE_CACHE.read_text(encoding="utf-8"))
        if cached.get("seed") == seed and cached.get("pool", 0) >= per_domain:
            pools = cached["domains"]

    if pools is None:
        pools = _scan_corpus(seed)
        SAMPLE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SAMPLE_CACHE.write_text(
            json.dumps(
                {"seed": seed, "pool": SAMPLE_POOL_PER_DOMAIN, "domains": pools},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return [
        Sample(row["doc_id"], row["url"], entry["domain"])
        for entry in pools[:top_domains]
        for row in entry["pool"][:per_domain]
    ]


# --------------------------------------------------------------------------
# Probing one page
# --------------------------------------------------------------------------

@dataclass
class PageProbe:
    """Everything one article told us about its domain."""
    doc_id: int
    domain: str
    url: str
    ok: bool
    failure: str = ""          # exception class name, when ok is False
    http_status: int = 0
    html_bytes: int = 0
    chars: int = 0             # markdown length, default settings
    chars_precise: int = 0     # markdown length with favor_precision
    headings: int = 0
    has_title: bool = False


async def probe_page(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    sample: Sample,
    retries: int,
    policy: PolitenessPolicy,
) -> PageProbe:
    """Fetch and extract one article without storing it. Never raises."""
    async with semaphore:
        try:
            html, _final_url, status = await fetch_html(
                client, sample.url, retries, policy=policy
            )
        except (FetchError, httpx.HTTPError) as exc:
            return PageProbe(
                doc_id=sample.doc_id, domain=sample.domain, url=sample.url,
                ok=False, failure=classify_failure(exc),
            )

    markdown = extract_markdown(html)
    precise = extract_markdown(html, favor_precision=True)
    metadata = extract_metadata(html)

    return PageProbe(
        doc_id=sample.doc_id,
        domain=sample.domain,
        url=sample.url,
        ok=True,
        http_status=status,
        html_bytes=len(html),
        chars=len(markdown),
        chars_precise=len(precise),
        headings=count_headings(markdown),
        has_title=bool(metadata["title"]),
    )


async def probe_all(
    samples: list[Sample], concurrency: int, timeout: float, retries: int
) -> list[PageProbe]:
    """Probe every sample, interleaved across domains to stay polite."""
    shuffled = list(samples)
    random.Random(0).shuffle(shuffled)

    semaphore = asyncio.Semaphore(concurrency)
    # The measurement run is a crawl too: one host must not receive ten
    # back-to-back requests just because it is being evaluated.
    policy = PolitenessPolicy(HostLimits(max_concurrent=1, min_interval=1.0))
    probes: list[PageProbe] = []

    columns = (
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[note]}"),
        TimeElapsedColumn(),
    )

    with Progress(*columns, console=console) as progress:
        task = progress.add_task("Probing articles", total=len(shuffled), note="")

        async with build_client(timeout) as client:
            pending = [
                asyncio.create_task(
                    probe_page(client, semaphore, sample, retries, policy)
                )
                for sample in shuffled
            ]
            for coro in asyncio.as_completed(pending):
                probe = await coro
                probes.append(probe)
                failed = sum(1 for p in probes if not p.ok)
                progress.update(task, advance=1, note=f"{failed} failed")

    return probes


# --------------------------------------------------------------------------
# Aggregating
# --------------------------------------------------------------------------

@dataclass
class DomainReport:
    domain: str
    corpus_pct: float
    sampled: int
    ok: int
    failures: dict[str, int] = field(default_factory=dict)
    median_chars: int = 0
    median_chars_precise: int = 0
    too_short: int = 0
    with_headings: int = 0
    with_title: int = 0

    @property
    def ok_rate(self) -> float:
        return self.ok / self.sampled if self.sampled else 0.0

    @property
    def too_short_rate(self) -> float:
        return self.too_short / self.ok if self.ok else 0.0

    @property
    def heading_rate(self) -> float:
        return self.with_headings / self.ok if self.ok else 0.0

    @property
    def extracted_nothing(self) -> bool:
        """Requests succeed but every page yields no text.

        This is a block wearing a success status - typically a JS cookie
        challenge served under HTTP 200 - not a weak extractor.
        """
        return self.ok > 0 and self.median_chars == 0

    @property
    def verdict(self) -> str:
        if self.ok == 0 or self.extracted_nothing:
            return BLOCKED
        if self.ok_rate < 0.8:
            return UNSTABLE
        if self.too_short_rate > 0.3:
            return THIN
        return HEALTHY

    @property
    def note(self) -> str:
        """The one sentence a reviewer needs for this domain."""
        if self.ok == 0:
            reason = max(self.failures, key=self.failures.get) if self.failures else "?"
            return f"every request failed: {reason}"
        if self.extracted_nothing:
            return "HTTP 200 but empty body - JS challenge, not an extraction problem"
        if self.ok_rate < 0.8:
            reason = max(self.failures, key=self.failures.get) if self.failures else "?"
            return f"{self.sampled - self.ok}/{self.sampled} failed: {reason}"
        if self.too_short_rate > 0.3:
            return f"{self.too_short}/{self.ok} extractions under the minimum"
        if self.heading_rate == 0:
            return "clean, but no markdown headings - needs recursive chunking"
        return "clean extraction, headings present"


def summarize(
    probes: list[PageProbe], corpus_pct: dict[str, float], min_chars: int
) -> list[DomainReport]:
    """Collapse per-page probes into one row per domain, worst first."""
    by_domain: dict[str, list[PageProbe]] = {}
    for probe in probes:
        by_domain.setdefault(probe.domain, []).append(probe)

    reports = []
    for domain, group in by_domain.items():
        good = [p for p in group if p.ok]

        failures: dict[str, int] = {}
        for probe in group:
            if not probe.ok:
                failures[probe.failure] = failures.get(probe.failure, 0) + 1

        reports.append(DomainReport(
            domain=domain,
            corpus_pct=corpus_pct.get(domain, 0.0),
            sampled=len(group),
            ok=len(good),
            failures=failures,
            median_chars=int(statistics.median([p.chars for p in good])) if good else 0,
            median_chars_precise=(
                int(statistics.median([p.chars_precise for p in good])) if good else 0
            ),
            too_short=sum(1 for p in good if p.chars < min_chars),
            with_headings=sum(1 for p in good if p.headings > 0),
            with_title=sum(1 for p in good if p.has_title),
        ))

    severity = {BLOCKED: 0, UNSTABLE: 1, THIN: 2, HEALTHY: 3}
    reports.sort(key=lambda r: (severity[r.verdict], -r.corpus_pct))
    return reports


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_table(reports: list[DomainReport], min_chars: int) -> None:
    table = Table(
        title=f"Extraction health by domain  (trafilatura, min_chars={min_chars})",
        title_style="bold",
        header_style="bold",
        expand=True,
    )
    table.add_column("Domain", overflow="fold")
    table.add_column("Corpus", justify="right")
    table.add_column("Fetched", justify="right")
    table.add_column("Median\nchars", justify="right")
    table.add_column("Precise\nmode", justify="right")
    table.add_column("Too\nshort", justify="right")
    table.add_column("Has\nheadings", justify="right")
    table.add_column("Verdict")
    table.add_column("What to know", overflow="fold")

    for report in reports:
        verdict = report.verdict
        fetched = f"{report.ok}/{report.sampled}"
        table.add_row(
            report.domain,
            f"{report.corpus_pct:.2f}%",
            f"[red]{fetched}[/red]" if report.ok_rate < 1 else fetched,
            f"{report.median_chars:,}" if report.ok else "-",
            f"{report.median_chars_precise:,}" if report.ok else "-",
            f"{report.too_short}/{report.ok}" if report.ok else "-",
            f"{report.with_headings}/{report.ok}" if report.ok else "-",
            f"[{VERDICT_STYLE[verdict]}]{verdict}[/{VERDICT_STYLE[verdict]}]",
            report.note,
        )

    console.print(table)


def render_summary(reports: list[DomainReport]) -> None:
    """Corpus-weighted view: how much of the dataset each verdict covers."""
    weight: dict[str, float] = {}
    for report in reports:
        weight[report.verdict] = weight.get(report.verdict, 0.0) + report.corpus_pct

    covered = sum(r.corpus_pct for r in reports)
    table = Table(
        title=f"Corpus coverage by verdict  ({covered:.1f}% of corpus sampled)",
        title_style="bold", header_style="bold", expand=False,
    )
    table.add_column("Verdict")
    table.add_column("Domains", justify="right")
    table.add_column("% of corpus", justify="right")

    for verdict in (BLOCKED, UNSTABLE, THIN, HEALTHY):
        if verdict not in weight:
            continue
        count = sum(1 for r in reports if r.verdict == verdict)
        table.add_row(
            f"[{VERDICT_STYLE[verdict]}]{verdict}[/{VERDICT_STYLE[verdict]}]",
            str(count),
            f"{weight[verdict]:.2f}%",
        )

    console.print(table)

    no_headings = [r for r in reports if r.ok and r.heading_rate == 0]
    if no_headings:
        share = sum(r.corpus_pct for r in no_headings)
        console.print(
            f"\n[bold]Chunking note[/bold]: {len(no_headings)} domains "
            f"({share:.1f}% of corpus) produce [bold]no markdown headings[/bold]. "
            "MarkdownHeaderTextSplitter would return one chunk per document there, "
            "so the chunker needs a recursive fallback."
        )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def load_corpus_pct() -> dict[str, float]:
    """Per-domain share of the corpus, from the earlier full-corpus count."""
    path = Path("data/domain_samples.json")
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["domain"]: row["pct"] for row in data["domains"]}


def drop_handled_domains(samples: list[Sample]) -> tuple[list[Sample], list[str]]:
    """Remove domains that already have a dedicated crawler.

    Those are solved or known-blocked, and re-probing them with the generic
    path would both waste requests and, in PubMed's case, repeat exactly the
    access pattern that got this client blocked. What is worth measuring is
    the territory no one has looked at yet.
    """
    kept, skipped = [], set()
    for sample in samples:
        crawler = sites.crawler_for_domain(sample.domain)
        if crawler is sites.FALLBACK_CRAWLER:
            kept.append(sample)
        else:
            skipped.add(f"{sample.domain} ({crawler.name})")
    return kept, sorted(skipped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure per-domain extraction quality on a sample of articles."
    )
    parser.add_argument("--domains", type=int, default=20,
                        help="How many of the largest domains to test")
    parser.add_argument("--per-domain", type=int, default=10,
                        help="Articles sampled from each domain")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-handled", action="store_true",
                        help="Also probe domains that already have their own crawler")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    samples = select_samples(args.domains, args.per_domain, args.seed)
    if not args.include_handled:
        samples, skipped = drop_handled_domains(samples)
        if skipped:
            console.print(
                f"[dim]Skipping {len(skipped)} domains that already have a "
                f"crawler: {', '.join(skipped)}[/dim]\n"
            )
    console.print(
        f"Sampling [bold]{args.per_domain}[/bold] articles from each of the "
        f"[bold]{args.domains}[/bold] largest domains "
        f"([bold]{len(samples)}[/bold] requests)\n"
    )

    probes = asyncio.run(
        probe_all(samples, args.concurrency, args.timeout, args.retries)
    )

    reports = summarize(probes, load_corpus_pct(), args.min_chars)

    console.print()
    render_table(reports, args.min_chars)
    console.print()
    render_summary(reports)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "settings": vars(args),
                "domains": [
                    {**asdict(r), "verdict": r.verdict, "note": r.note}
                    for r in reports
                ],
                "pages": [asdict(p) for p in probes],
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"\n[dim]Per-page measurements saved to {REPORT_PATH}[/dim]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
