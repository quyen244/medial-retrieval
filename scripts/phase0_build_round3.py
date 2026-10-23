"""Vòng 3: chốt m_doc tại đỉnh đo được, và thử nới trần cửa sổ.

Bảng chi tiết của E1/E3/E4 cho đường cong h_doc thật (h = P x m, không suy từ F2):

    m=25.0  -> h=6.093
    m=35.1  -> h=7.013      biên 9.11% mỗi doc ở hạng 26-35
    m=42.7  -> h=7.058      biên 0.60% mỗi doc ở hạng 36-43

Ngưỡng đáng thêm một doc là h/(4g+m) = 2.9%. Hạng 26-35 vượt xa, hạng 36-43 nằm
dưới hẳn. Đỉnh F2_doc đo được ở m=35.1 (0.1293); m=25 cho 0.1222 và m=42.7 cho
0.1253. Nên chốt m_doc=35, không nới lên 43 cũng không cắt xuống 25.

Tầng chunk thì ngược chiều: biên 4.21% mỗi chunk so với ngưỡng 2.15%, nên giữ
đủ 44. Hai chiều trái nhau chính là chỗ việc tách relevant_docs khỏi
relevant_chunks sinh lợi.

N3 chốt m_doc=35. N5 giữ nguyên N3 và chỉ nới trần cửa sổ 8.000 -> 20.000, để
tách hai khả năng còn lại của 22% chunk đang trượt (h_c/h_d = 0.78): bị cắt đuôi
ở trần 8.000 thì nới trần sẽ cứu; bị phạt vì mẫu số quá dài thì nới trần sẽ làm
tệ hơn. Hai kết quả chỉ ra hai việc khác nhau.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.pipeline.submission import zip_submission
from scripts.phase0_assemble import DUMP, OUT_DIR, RawDocs, build, load_index

VARIANTS = [
    # name,           docs, chunks, mode,   win, cap
    ("N3_d35_c44",      35,     44, "full",   0,  8000),
    ("N5_d35_cap20k",   35,     44, "full",   0, 20000),
]


def main():
    records = [json.loads(l) for l in open(DUMP, encoding="utf-8")]
    raw_docs = RawDocs(load_index())
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, n_docs, n_chunks, mode, win, cap in VARIANTS:
        entries, stats = build(records, raw_docs, n_docs, n_chunks, mode, win, cap)
        ids = [e["id"] for e in entries]
        assert len(ids) == len(set(ids)) == 1200
        assert all(isinstance(d, int) for e in entries for d in e["relevant_docs"])
        assert all(c["chunk_text"] for e in entries for c in e["relevant_chunks"])
        stray = sum(1 for e in entries
                    if not {c["doc_id"] for c in e["relevant_chunks"]} <= set(e["relevant_docs"]))

        out = os.path.join(OUT_DIR, f"{name}.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, ensure_ascii=False)
        zp = zip_submission(out)
        ch = sorted(stats["chars"])
        print(f"{name:14s} doc/câu={statistics.mean(stats['n_docs']):5.1f} "
              f"chunk/câu={statistics.mean(stats['n_chunks']):5.1f} "
              f"chars median={ch[len(ch)//2]:5d} mean={statistics.mean(ch):6.0f} "
              f"max={ch[-1]:6d} | zip {os.path.getsize(zp)/1e6:5.1f} MB | stray={stray}")


if __name__ == "__main__":
    main()
