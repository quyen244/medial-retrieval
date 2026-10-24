"""Lắp lại bài nộp từ pool doc-level ĐÃ RERANK, đổi m_doc / m_chunk, không cần GPU.

scripts/phase1_doc_retrieve.py ghi <tên>_pool.jsonl: mỗi câu đủ top_docs doc đã
xếp theo điểm rerank. Đổi số doc nộp chỉ là cắt lại danh sách đó.

Hai tầng chấm độc lập (CLAUDE.md): DOCS_F2 chỉ phụ thuộc relevant_docs, CHUNKS_F2
chỉ phụ thuộc relevant_chunks. Nên một bài đổi cả m_doc lẫn m_chunk vẫn đọc tách
được hiệu ứng từng tầng.

    python -m scripts.assemble_doc_submission --pool data/tmp/phase1/doc_d35_c44_top60_bge_pool.jsonl \
        --m-doc 50 --m-chunk 60 --name doc_d50_c60_top60_bge
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.md_strip import strip_markdown
from scripts.phase0_assemble import RawDocs, load_index
from src.rag.pipeline.submission import submission_chunk_text, submission_doc_id, zip_submission

N_QUERIES = 1200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--m-doc", type=int, required=True)
    ap.add_argument("--m-chunk", type=int, required=True)
    ap.add_argument("--cap", type=int, default=8000, help="trần ký tự chunk_text (đỉnh đã đo ở Phase 0)")
    ap.add_argument("--no-strip", action="store_true", help="không bóc cú pháp markdown")
    ap.add_argument("--out-dir", default="data/tmp/phase1")
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(args.pool, encoding="utf-8")]
    sizes = [len(p["candidates"]) for p in pool]
    if min(sizes) < max(args.m_doc, args.m_chunk):
        print(f"CẢNH BÁO: pool nhỏ nhất {min(sizes)} doc < m yêu cầu; câu đó nộp ít hơn")
    raw_docs = RawDocs(load_index())
    prep = submission_chunk_text if args.no_strip else (lambda x: submission_chunk_text(strip_markdown(x)))

    entries, n_doc, n_chunk, chars, missing = [], [], [], [], 0
    for p in pool:
        cands = p["candidates"]   # đã xếp theo rerank_score giảm dần
        docs = [submission_doc_id(c["doc_id"]) for c in cands[: args.m_doc]]
        chunks = []
        for c in cands[: args.m_chunk]:
            raw = raw_docs.markdown(c["doc_id"])
            if not raw:
                missing += 1
                continue
            chunks.append({"doc_id": submission_doc_id(c["doc_id"]), "chunk_text": prep(raw[: args.cap])})
        entries.append({"id": int(p["id"]), "relevant_docs": docs, "relevant_chunks": chunks})
        n_doc.append(len(docs))
        n_chunk.append(len(chunks))
        chars += [len(c["chunk_text"]) for c in chunks]

    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)) == N_QUERIES, f"cần đủ {N_QUERIES} id, đang có {len(set(ids))}"
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.name}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False)
    zp = zip_submission(out)
    size = os.path.getsize(zp) / 1e6
    if size >= 100:
        os.remove(zp)
        raise SystemExit(f"{zp} nặng {size:.1f} MB, vượt giới hạn 100 MB của BTC. Giảm --m-chunk "
                         f"(~1,58 MB mỗi chunk ở trần 8.000) hoặc --cap.")
    chars.sort()
    print(f"{args.name}: m_doc TB {statistics.mean(n_doc):.1f} | m_chunk TB {statistics.mean(n_chunk):.1f} | "
          f"chunk median {chars[len(chars) // 2]} ký tự | thiếu bản gốc {missing}")
    print(f"-> {zp} ({size:.1f} MB, giới hạn 100 MB)")


if __name__ == "__main__":
    main()
