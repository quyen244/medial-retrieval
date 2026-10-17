"""Where a crawled document lands on disk.

Documents are grouped by domain - data/raw_documents/<domain>/<doc_id>.json -
because extraction quality is a per-domain property. Grouping this way lets
you review, re-extract or discard one site's documents without touching the
rest, and keeps any single directory from holding a million files.
"""

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def domain_of(url: str) -> str:
    """Host part of a URL, lowercased and without the port."""
    netloc = urlparse(url).netloc.lower()
    return netloc.split("@")[-1].split(":")[0]


def is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def document_path(out_dir: Path, domain: str, doc_id: int) -> Path:
    """Destination for one document. The domain is used verbatim as the folder."""
    return out_dir / domain / f"{doc_id}.json"


def save_document(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# Failure ledger
# --------------------------------------------------------------------------
#
# Whatever could not be crawled is written down rather than dropped: each
# domain that needs its own crawler later starts from its own backlog file.

FAILURES_DIRNAME = "_failures"


def failure_path(out_dir: Path, domain: str) -> Path:
    return out_dir / FAILURES_DIRNAME / f"{domain}.jsonl"


def record_failures(out_dir: Path, records: list[dict[str, Any]]) -> None:
    """Append failures as JSONL, one file per domain.

    Appending means a re-run can add a second entry for the same document.
    That is deliberate: the history shows whether a domain fails the same way
    every time. Use load_failures() to collapse it to the latest attempt.
    """
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_domain.setdefault(record["domain"], []).append(record)

    for domain, group in by_domain.items():
        path = failure_path(out_dir, domain)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in group
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(lines)


def load_failures(out_dir: Path, domain: str) -> dict[int, dict[str, Any]]:
    """Latest recorded failure per doc_id for one domain."""
    path = failure_path(out_dir, domain)
    if not path.exists():
        return {}

    latest: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record["doc_id"]] = record
    return latest
