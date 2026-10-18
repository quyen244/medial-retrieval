"""Build per-language domain-sample testsets for manual chunking inspection.

Mirrors how data/testset/chunking_domain_sample.json (Vietnamese) was built: for each domain
in the target language bucket, pick the file with the smallest numeric doc_id in that domain's
folder, in alphabetical-by-domain order, and dump the raw per-doc JSON objects unmodified.

Chinese has 34 domains in the current corpus, so "one doc per domain" gives good structural
coverage. English has exactly one domain (pubmed.ncbi.nlm.nih.gov), so "one per domain" would
yield a single, unrepresentative document; for English this script instead draws a fixed-seed
stratified sample across heading_count/char_count buckets to cover both bare abstracts and
full Background/Methods/Results/Conclusion case reports.
"""
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from src.ingestion.categories import language_of

RAW_DOCS_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw_documents"


def _domains() -> List[str]:
    return sorted(
        d for d in os.listdir(RAW_DOCS_ROOT)
        if d != "_failures" and (RAW_DOCS_ROOT / d).is_dir()
    )


def _load_doc(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_one_per_domain_sample(language: str) -> List[Dict[str, Any]]:
    """For every domain publishing in `language`, pick the file with the smallest numeric
    doc_id in that domain's folder (alphabetical-by-domain order)."""
    docs: List[Dict[str, Any]] = []
    for domain in _domains():
        if language_of(domain) != language:
            continue
        folder = RAW_DOCS_ROOT / domain
        doc_ids = []
        for fn in os.listdir(folder):
            if not fn.endswith(".json"):
                continue
            stem = fn[:-5]
            if stem.isdigit():
                doc_ids.append(int(stem))
        if not doc_ids:
            continue
        smallest = min(doc_ids)
        docs.append(_load_doc(folder / f"{smallest}.json"))
    return docs


def build_stratified_english_sample(n: int = 18, seed: int = 42) -> List[Dict[str, Any]]:
    """Stratified sample of pubmed.ncbi.nlm.nih.gov docs: bucket by heading_count (bare
    abstract vs. full structured case report) and by char_count (short vs. long), then
    draw evenly from each bucket so the dump covers structural diversity, not just volume.
    """
    folder = RAW_DOCS_ROOT / "pubmed.ncbi.nlm.nih.gov"
    files = sorted(f for f in os.listdir(folder) if f.endswith(".json"))

    rng = random.Random(seed)
    rng.shuffle(files)

    # Light-touch metadata scan (heading_count/char_count only) to bucket without loading
    # every doc's full markdown body twice.
    catalog = []
    for fn in files:
        doc = _load_doc(folder / fn)
        hc = doc["content"].get("heading_count", 0)
        cc = doc["content"].get("char_count", 0)
        catalog.append((fn, hc, cc, doc))
        if len(catalog) >= 4000:  # enough to populate every bucket without scanning all 239k
            break

    median_cc = sorted(c[2] for c in catalog)[len(catalog) // 2]

    buckets: Dict[str, List[Any]] = {
        "bare_short": [], "bare_long": [], "structured_short": [], "structured_long": [],
    }
    for fn, hc, cc, doc in catalog:
        structured = "structured" if hc >= 6 else "bare"
        length = "long" if cc >= median_cc else "short"
        buckets[f"{structured}_{length}"].append(doc)

    per_bucket = max(1, n // len(buckets))
    sample: List[Dict[str, Any]] = []
    for key, items in buckets.items():
        rng.shuffle(items)
        sample.extend(items[:per_bucket])

    # Top up to n if some bucket was short on items.
    if len(sample) < n:
        remaining = [doc for items in buckets.values() for doc in items if doc not in sample]
        rng.shuffle(remaining)
        sample.extend(remaining[: n - len(sample)])

    sample = sample[:n]
    sample.sort(key=lambda d: d["source"]["doc_id"])
    return sample


def dump(docs: List[Dict[str, Any]], out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(docs)} docs to {out}")


if __name__ == "__main__":
    zh_docs = build_one_per_domain_sample("zh")
    dump(zh_docs, str(RAW_DOCS_ROOT.parents[0] / "testset" / "chunking_domain_sample_zh.json"))

    en_docs = build_stratified_english_sample(n=18, seed=42)
    dump(en_docs, str(RAW_DOCS_ROOT.parents[0] / "testset" / "chunking_domain_sample_en.json"))
