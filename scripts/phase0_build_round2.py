"""Vòng 2: TÁCH RỜI số doc và số chunk.

Kết quả vòng 1 cho hai sự thật trái chiều nhau:
  - h_doc PHẲNG từ m=35.1 tới 42.7 (6.214 -> 6.212). Doc ở hạng 36-43 của pool
    hiện tại gần như không có gold, nên `F2 = 5h/(4g+m)` chỉ còn giảm theo m.
    => danh sách doc phải NGẮN LẠI.
  - Cửa sổ dài đưa h_chunk từ 1.64 lên 4.93 (h_chunk/h_doc: 0.26 -> 0.79).
    Mỗi cửa sổ thêm vào vẫn mang thêm chunk đúng.
    => danh sách chunk nên GIỮ DÀI.

Hai chiều ngược nhau, nên đây mới là lúc việc tách `relevant_docs` khỏi
`relevant_chunks` thật sự sinh lợi. Cả hai bài dùng CHUNG một chunk list
(44 cửa sổ) nên F2_chunk bằng nhau, và hiệu số FINAL giữa chúng tách ra
đúng F2_doc ở hai mốc m - giải được h_doc(12) và h_doc(25) không cần giả định.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.pipeline.submission import zip_submission
from scripts.phase0_assemble import DUMP, OUT_DIR, RawDocs, build, load_index

VARIANTS = [
    # name,        docs, chunks, mode,   win, cap
    ("N1_d25_c44",   25,     44, "full",   0, 8000),
    ("N2_d12_c44",   12,     44, "full",   0, 8000),
]


def main():
    records = [json.loads(l) for l in open(DUMP, encoding="utf-8")]
    raw_docs = RawDocs(load_index())
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, n_docs, n_chunks, mode, win, cap in VARIANTS:
        entries, stats = build(records, raw_docs, n_docs, n_chunks, mode, win, cap)

        ids = [e["id"] for e in entries]
        assert len(ids) == len(set(ids)) == 1200
        stray = sum(1 for e in entries
                    if not {c["doc_id"] for c in e["relevant_chunks"]} <= set(e["relevant_docs"]))

        out = os.path.join(OUT_DIR, f"{name}.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, ensure_ascii=False)
        zp = zip_submission(out)
        chars = sorted(stats["chars"])
        print(f"{name:12s} doc/câu={statistics.mean(stats['n_docs']):5.1f} "
              f"chunk/câu={statistics.mean(stats['n_chunks']):5.1f} "
              f"chars median={chars[len(chars)//2]:5d} mean={statistics.mean(chars):6.0f} | "
              f"zip {os.path.getsize(zp)/1e6:5.1f} MB | câu có chunk ngoài relevant_docs={stray}")


if __name__ == "__main__":
    main()
