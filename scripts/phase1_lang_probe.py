"""Bước 1.0 - thăm dò ngôn ngữ. Không cần GPU, lắp từ dump top-50 có sẵn.

Pool hiện tại có vi 48% / en 34% / zh 17%; sau rerank thành vi 59% / en 22% /
zh 18%. Nhưng KHÔNG biết 6,93 gold doc tìm được thuộc ngôn ngữ nào.

Cách đo: nộp CHỈ các doc en+zh của pool. Leaderboard trả h_enzh trực tiếp
(h = P x m). Vì recall chấm trên cùng một g_doc ~51 cho mọi bài, suy ra
h_vi ~ 6,93 - h_enzh.

Quyết định điều gì:
  - h_enzh cao  -> nguồn en/zh đang đóng góp thật, đáng đầu tư dịch query
                   sang en/zh và đáng crawl thêm tiếng Trung.
  - h_enzh ~ 0  -> gold gần như toàn tiếng Việt. Bỏ hẳn nhánh dịch thuật,
                   dồn sức vào chiều sâu nhánh vi. Tiết kiệm nhiều ngày công.

Chunk đi kèm dùng cửa sổ dài trần 8.000 như N3 (cấu hình tốt nhất đã đo), và
giới hạn trong đúng các doc được nộp để không lẫn biến.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.pipeline.submission import (submission_chunk_text, submission_doc_id,
                                         zip_submission)
from scripts.phase0_assemble import DUMP, OUT_DIR, RawDocs, doc_rank, load_index

CAP = 8000


def build(records, raw_docs, langs, n_docs):
    entries, stats = [], {"nd": [], "nc": [], "chars": []}
    for r in records:
        ranked = [(d, c) for d, (_s, c) in doc_rank(r["candidates"])
                  if c["language"] in langs][:n_docs]
        docs, chunks = [], []
        for d, c in ranked:
            raw = raw_docs.markdown(d)
            text = raw[:CAP] if raw else c["content"]
            docs.append(submission_doc_id(d))
            chunks.append({"doc_id": submission_doc_id(d),
                           "chunk_text": submission_chunk_text(text)})
        if not docs:  # không có doc nào thuộc ngôn ngữ này -> vẫn phải có entry
            stats["nd"].append(0)
            stats["nc"].append(0)
            entries.append({"id": int(r["id"]), "relevant_docs": [], "relevant_chunks": []})
            continue
        stats["nd"].append(len(docs))
        stats["nc"].append(len(chunks))
        stats["chars"].extend(len(c["chunk_text"]) for c in chunks)
        entries.append({"id": int(r["id"]), "relevant_docs": docs, "relevant_chunks": chunks})
    return entries, stats


def main():
    records = [json.loads(l) for l in open(DUMP, encoding="utf-8")]
    raw_docs = RawDocs(load_index())
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, langs, n in [("P1_enzh_only", {"en", "zh"}, 35),
                           ("P2_vi_only", {"vi"}, 35)]:
        entries, stats = build(records, raw_docs, langs, n)
        assert len({e["id"] for e in entries}) == 1200
        empty = sum(1 for e in entries if not e["relevant_docs"])
        out = os.path.join(OUT_DIR, f"{name}.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, ensure_ascii=False)
        zp = zip_submission(out)
        ch = sorted(stats["chars"])
        print(f"{name:14s} doc/câu={statistics.mean(stats['nd']):5.2f} "
              f"chunk/câu={statistics.mean(stats['nc']):5.2f} "
              f"chars median={ch[len(ch)//2]:5d} | zip {os.path.getsize(zp)/1e6:5.1f} MB | "
              f"câu rỗng={empty}")


if __name__ == "__main__":
    main()
