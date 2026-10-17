"""How far along the crawl is, per domain.

Run:
    python -m src.ingestion.status
    python -m src.ingestion.status --language zh --incomplete-only

Everything here is *derived*, never stored. The documents on disk are the
truth about what was crawled, the corpus is the truth about what was expected,
and the failure ledger is the truth about what was tried and refused. A
separate status table would be a fourth copy of facts the first three already
hold, and the moment a run is interrupted - or a directory is deleted by hand -
that copy becomes a confident liar.

The cost of deriving instead of storing is one directory listing per domain,
which is seconds for a corpus of this size and only when the report is asked
for.
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.ingestion import storage
from src.ingestion.categories import language_of


console = Console()

LANGUAGE_LABEL = {"vi": "Tiếng Việt", "zh": "Tiếng Trung", "en": "Tiếng Anh"}


@dataclass
class DomainStatus:
    domain: str
    language: str
    expected: int          # rows this domain has in the corpus
    stored: int            # documents written to disk
    failed: int            # distinct doc_ids the ledger says were refused
    failure_reasons: dict[str, int]

    @property
    def attempted(self) -> int:
        return self.stored + self.failed

    @property
    def remaining(self) -> int:
        """Documents never tried, or tried before the ledger was written."""
        return max(0, self.expected - self.attempted)

    @property
    def completeness(self) -> float:
        return self.stored / self.expected if self.expected else 0.0

    @property
    def is_done(self) -> bool:
        """Nothing left to attempt - every row either landed or was refused."""
        return self.remaining == 0

    @property
    def verdict(self) -> str:
        if self.stored == 0 and self.failed:
            return "BLOCKED"
        if self.is_done and self.failed == 0:
            return "COMPLETE"
        if self.is_done:
            return "DONE (with gaps)"
        if self.attempted == 0:
            return "NOT STARTED"
        return "IN PROGRESS"


VERDICT_STYLE = {
    "COMPLETE": "bold green",
    "DONE (with gaps)": "green",
    "IN PROGRESS": "bold yellow",
    "NOT STARTED": "dim",
    "BLOCKED": "bold red",
}


def count_stored(out_dir: Path, domain: str) -> int:
    folder = out_dir / domain
    if not folder.is_dir():
        return 0
    return sum(1 for _ in folder.glob("*.json"))


def read_ledger(out_dir: Path, domain: str) -> tuple[int, dict[str, int]]:
    """Distinct failed doc_ids and a count per reason."""
    records = storage.load_failures(out_dir, domain)
    reasons = Counter(r.get("reason", "?") for r in records.values())
    return len(records), dict(reasons)


def collect(out_dir: Path, expected_counts: dict[str, int]) -> list[DomainStatus]:
    rows = []
    for domain, expected in expected_counts.items():
        failed, reasons = read_ledger(out_dir, domain)
        rows.append(DomainStatus(
            domain=domain,
            language=language_of(domain),
            expected=expected,
            stored=count_stored(out_dir, domain),
            failed=failed,
            failure_reasons=reasons,
        ))
    return rows


def load_expected() -> dict[str, int]:
    """Rows per domain in the corpus, from the earlier full-corpus count."""
    path = Path("data/domain_samples.json")
    if not path.exists():
        raise SystemExit(
            "data/domain_samples.json is missing - it holds the expected row "
            "count per domain. Regenerate it before asking for status."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["domain"]: row["count"] for row in data["domains"]}


def render(rows: list[DomainStatus], show_reasons: bool) -> None:
    for language in ("vi", "zh", "en"):
        group = [r for r in rows if r.language == language]
        if not group:
            continue
        group.sort(key=lambda r: -r.expected)

        expected = sum(r.expected for r in group)
        stored = sum(r.stored for r in group)
        failed = sum(r.failed for r in group)

        table = Table(
            title=(
                f"{LANGUAGE_LABEL[language]}  -  "
                f"{stored:,}/{expected:,} stored ({stored/expected*100:.1f}%), "
                f"{failed:,} failed"
            ),
            title_style="bold", header_style="bold", expand=True,
        )
        table.add_column("Domain", overflow="fold")
        table.add_column("Expected", justify="right")
        table.add_column("Stored", justify="right")
        table.add_column("Failed", justify="right")
        table.add_column("Remaining", justify="right")
        table.add_column("%", justify="right")
        table.add_column("Status")
        if show_reasons:
            table.add_column("Top failure", overflow="fold")

        for row in group:
            style = VERDICT_STYLE[row.verdict]
            cells = [
                row.domain,
                f"{row.expected:,}",
                f"{row.stored:,}",
                f"{row.failed:,}" if row.failed else "-",
                f"{row.remaining:,}" if row.remaining else "-",
                f"{row.completeness*100:.1f}%",
                f"[{style}]{row.verdict}[/{style}]",
            ]
            if show_reasons:
                top = max(row.failure_reasons, key=row.failure_reasons.get, default="")
                cells.append(
                    f"{top} x{row.failure_reasons[top]}" if top else "-"
                )
            table.add_row(*cells)

        console.print(table)
        console.print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report crawl completeness per domain, derived from disk."
    )
    parser.add_argument("--out", default="data/raw_documents")
    parser.add_argument("--language", choices=["vi", "zh", "en"],
                        help="Only report one language group")
    parser.add_argument("--incomplete-only", action="store_true",
                        help="Hide domains that have nothing left to attempt")
    parser.add_argument("--reasons", action="store_true",
                        help="Show the most common failure reason per domain")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable output instead of a table")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)

    rows = collect(out_dir, load_expected())
    if args.language:
        rows = [r for r in rows if r.language == args.language]
    if args.incomplete_only:
        rows = [r for r in rows if not r.is_done]

    if args.json:
        print(json.dumps([
            {
                "domain": r.domain, "language": r.language,
                "expected": r.expected, "stored": r.stored, "failed": r.failed,
                "remaining": r.remaining, "completeness": round(r.completeness, 4),
                "verdict": r.verdict, "failure_reasons": r.failure_reasons,
            }
            for r in rows
        ], ensure_ascii=True, indent=2))
        return 0

    if not rows:
        console.print("[green]Nothing left to crawl.[/green]")
        return 0

    render(rows, args.reasons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
