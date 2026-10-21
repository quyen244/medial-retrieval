"""Xuất 1.200 query thật (get_dataset('query')) ra jsonl cho run_augment.py.

run_augment.py đọc file local (id/query mỗi dòng), còn 1.200 query thật nằm
trong dataset HF AIGuruTinix/ViBioMIR (xem src/ingestion/utils.get_dataset).
Script này chỉ nối hai bên, không có logic gì khác.
"""
import json
from pathlib import Path

from src.ingestion.utils import get_dataset

OUT = Path("data/queries/real_queries.jsonl")


def main() -> int:
    ds = get_dataset("query")["train"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps({"id": row["id"], "query": row["query"]}, ensure_ascii=False) + "\n")
    print(f"-> {OUT} ({len(ds)} query)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
