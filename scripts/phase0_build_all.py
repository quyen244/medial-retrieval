"""Lắp mọi biến thể Phase 0 trong MỘT tiến trình.

Dump top-50 nặng 80 MB nên mỗi lần chạy lại script lắp bài là mất công đọc lại
từ đầu; gom các biến thể vào một lượt đọc.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.pipeline.submission import zip_submission
from scripts.phase0_assemble import DUMP, OUT_DIR, RawDocs, build, load_index, validate

VARIANTS = [
    # name,          docs, chunks, mode,   window, cap
    ("E1_docs44",      44,     40, "orig",   0,      0),
    ("E2_win2500",     44,     44, "win",    2500,   0),
    ("E3_full8k",      44,     44, "full",   0,      8000),
]


def main():
    records = [json.loads(l) for l in open(DUMP, encoding="utf-8")]
    raw_docs = RawDocs(load_index())
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, n_docs, n_chunks, mode, win, cap in VARIANTS:
        entries, stats = build(records, raw_docs, n_docs, n_chunks, mode, win, cap)
        validate(entries, len(records))
        out = os.path.join(OUT_DIR, f"{name}.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, ensure_ascii=False)
        zp = zip_submission(out)
        chars = sorted(stats["chars"])
        print(f"{name:14s} doc/câu={statistics.mean(stats['n_docs']):5.1f} "
              f"chunk/câu={statistics.mean(stats['n_chunks']):5.1f} "
              f"chars median={chars[len(chars)//2]:5d} mean={statistics.mean(chars):6.0f} "
              f"max={chars[-1]:6d} | json {os.path.getsize(out)/1e6:6.1f} MB "
              f"zip {os.path.getsize(zp)/1e6:5.1f} MB | fallback={len(stats['fallback'])}")


if __name__ == "__main__":
    main()
