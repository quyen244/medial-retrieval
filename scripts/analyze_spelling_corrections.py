"""Compare query dataset text before and after spelling correction."""

import argparse
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Set, TextIO, Tuple

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.ingestion.utils import get_dataset
from src.pipeline.spelling_corrector import SpellingCorrector


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "data" / "spelling_correction_analysis"
DELETED = "[deleted]"
ADDED = "[added]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run spelling correction on the ViBioMIR query train split and "
            "report changed queries and word mappings."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of queries sent to the corrector per batch (default: 16)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optionally process only the first N queries",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def get_word_changes(original: str, corrected: str) -> List[Tuple[str, str]]:
    """Return changed whitespace-delimited tokens as before/after pairs."""
    original_words = original.split()
    corrected_words = corrected.split()
    matcher = SequenceMatcher(
        None, original_words, corrected_words, autojunk=False
    )
    changes: List[Tuple[str, str]] = []

    for tag, original_start, original_end, corrected_start, corrected_end in (
        matcher.get_opcodes()
    ):
        if tag == "equal":
            continue

        original_block = original_words[original_start:original_end]
        corrected_block = corrected_words[corrected_start:corrected_end]
        paired_count = min(len(original_block), len(corrected_block))
        changes.extend(
            zip(original_block[:paired_count], corrected_block[:paired_count])
        )
        changes.extend(
            (word, DELETED) for word in original_block[paired_count:]
        )
        changes.extend(
            (ADDED, word) for word in corrected_block[paired_count:]
        )

    return changes


def write_query_report(
    report: TextIO, query_number: int, original: str, corrected: str
) -> None:
    changes = get_word_changes(original, corrected)
    report.write(f"Query #{query_number}\n")
    report.write(f"Before: {original}\n")
    report.write(f"After: {corrected}\n")
    report.write("Different words:\n")
    if changes:
        for old_word, new_word in changes:
            report.write(f"- {old_word} -> {new_word}\n")
    else:
        report.write("- No word changes (whitespace or formatting changed).\n")
    report.write("\n")


def add_to_word_map(
    word_map: Dict[str, Set[str]], original: str, corrected: str
) -> None:
    for old_word, new_word in get_word_changes(original, corrected):
        word_map.setdefault(old_word, set()).add(new_word)


def main() -> None:
    args = parse_args()
    dataset = get_dataset("query")
    rows = dataset["train"]
    queries = [row["query"] for row in rows]
    if args.limit is not None:
        queries = queries[: args.limit]

    for index, query in enumerate(queries, start=1):
        if not isinstance(query, str):
            raise TypeError(
                f"Expected query #{index} to be str, got {type(query).__name__}"
            )

    tokenizer = AutoTokenizer.from_pretrained("nluai/correction-spelling-vi")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "nluai/correction-spelling-vi", device_map="auto"
    )
    corrector = SpellingCorrector(tokenizer, model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "changed_queries.txt"
    word_map_path = args.output_dir / "word_map.txt"
    word_map: Dict[str, Set[str]] = {}
    changed_count = 0

    with report_path.open("w", encoding="utf-8") as report:
        for batch_start in range(0, len(queries), args.batch_size):
            batch = queries[batch_start : batch_start + args.batch_size]
            corrected_batch = corrector.inference(batch, verbose=False)
            if len(corrected_batch) != len(batch):
                raise RuntimeError(
                    "SpellingCorrector returned "
                    f"{len(corrected_batch)} results for {len(batch)} queries"
                )

            for offset, (original, corrected) in enumerate(
                zip(batch, corrected_batch)
            ):
                if original == corrected:
                    continue

                changed_count += 1
                query_number = batch_start + offset + 1
                write_query_report(report, query_number, original, corrected)
                add_to_word_map(word_map, original, corrected)

    with word_map_path.open("w", encoding="utf-8") as word_map_file:
        for original_word in sorted(word_map):
            for corrected_word in sorted(word_map[original_word]):
                word_map_file.write(
                    f"{original_word}: {corrected_word}\n"
                )

    print(f"Processed {len(queries)} queries; changed: {changed_count}.")
    print(f"Changed-query report: {report_path}")
    print(f"Word map: {word_map_path}")


if __name__ == "__main__":
    main()
