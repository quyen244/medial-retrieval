"""Dựng 4 bài nộp vòng private: 3 danh sách doc x 2 danh sách chunk, đọc markdown gốc MỘT lần.

Hai tầng chấm độc lập (CLAUDE.md), nên mỗi bài là một cặp (doc list, chunk list):
  1. ftrr d160        + chunk c100 trần 2.700 (= tầng chunk của SOTA, chép thẳng từ zip SOTA)
  2. bge  d160 (SOTA) + chunk c120 trần 2.050
  3. ftrr d160        + chunk c120 trần 2.050
  4. ens  d160        + chunk c120 trần 2.050

Doc list (cùng 300 ứng viên của pool top300, data/tmp/phase1/doc_top300_bge_pool.jsonl):
  bge   điểm bge-reranker-v2-m3 với câu gốc (= `orig` của fuse_submission, bài SOTA)
  ftrr  điểm quyenpro/re-ranker-vibio với câu sửa (scripts/kaggle_rerank_ft.py merge)
  ens   RRF có trọng số: 0,6/(60 + hạng ftrr) + 0,4/(60 + hạng bge). Gộp theo HẠNG vì hai
        model cho điểm trên hai thang khác nhau, cộng điểm thì trọng số mất nghĩa.

Chunk list c120: nối top60 rồi top300 (bỏ trùng) như fuse_submission, cửa sổ đầu bài.

    python -m scripts.build_private_submissions                 # dựng cả 4
    python -m scripts.build_private_submissions --cap 1950      # nếu c120 vượt 100 MB

Trần 2.250 cho đoạn TB 1.897 ký tự, mỗi zip c120 106,1 MB (vượt 100 MB), nên mặc định là 2.050.
"""
import argparse
import json
import os
import statistics
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.fuse_submission import POOLS, RRF_K, load_pool, rank_docs
from scripts.md_strip import strip_markdown
from scripts.phase0_assemble import RawDocs, load_index
from src.rag.pipeline.submission import submission_chunk_text, submission_doc_id, zip_submission

D = "data/tmp/phase1"
SOTA_ZIP = "data/fuse_d160_c100_top300_orig_ctop60+top300_cap2700.zip"
FTRR_POOL = POOLS["ftrr_qc"]
LIMIT = 100e6  # BTC: dưới 100 MB; lấy MB thập phân cho chắc (nhỏ hơn 100 MiB)


def ranked(pool):
    return {q: sorted(cs, key=lambda c: -c["rerank_score"]) for q, cs in pool.items()}


def doc_lists(m_doc, w_new):
    bge_pool = load_pool(POOLS["top300"])
    ftrr_pool = load_pool(FTRR_POOL)
    assert set(bge_pool) == set(ftrr_pool), "hai pool khác tập câu"
    out = {"bge": {}, "ftrr": {}, "ens": {}}
    for q in sorted(bge_pool):
        bge = rank_docs(bge_pool[q], None, q, "orig")
        ftrr = [c["doc_id"] for c in sorted(ftrr_pool[q], key=lambda c: -c["rerank_score"])]
        assert set(bge) == set(ftrr), f"câu {q}: hai pool khác ứng viên"
        key = {d: 0.0 for d in bge}
        for w, order in ((w_new, ftrr), (1 - w_new, bge)):
            for r, d in enumerate(order, 1):
                key[d] += w / (RRF_K + r)
        ens = sorted(key, key=lambda d: (-key[d], ftrr.index(d)))
        out["bge"][q], out["ftrr"][q], out["ens"][q] = bge[:m_doc], ftrr[:m_doc], ens[:m_doc]
    return out


def sota():
    z = zipfile.ZipFile(SOTA_ZIP)
    return {r["id"]: r for r in json.loads(z.read(z.namelist()[0]))}


def chunk_list(m_chunk, cap):
    """Cùng logic fuse_submission: nối top60 rồi top300, bỏ trùng; bỏ qua doc thiếu bản gốc."""
    rank = {}
    for label in ("top60", "top300"):
        for q, cs in load_pool(POOLS[label]).items():
            seen = set(rank.setdefault(q, []))
            rank[q] += [c["doc_id"] for c in cs if c["doc_id"] not in seen]
    raw_docs = RawDocs(load_index())
    out, missing = {}, 0
    for i, q in enumerate(sorted(rank), 1):
        chunks = []
        for d in rank[q][:m_chunk]:
            raw = raw_docs.markdown(d)
            if not raw:
                missing += 1
                continue
            chunks.append({"doc_id": submission_doc_id(d),
                           "chunk_text": submission_chunk_text(strip_markdown(raw[:cap]))})
        out[q] = chunks
        if i % 200 == 0:
            print(f"[chunk] {i}/{len(rank)} câu", flush=True)
    chars = sorted(len(c["chunk_text"]) for cs in out.values() for c in cs)
    print(f"[chunk] c{m_chunk} trần {cap}: TB {statistics.mean(chars):.0f} ký tự, "
          f"median {chars[len(chars) // 2]}, thiếu bản gốc {missing}")
    return out


def write(name, docs, chunks):
    entries = [{"id": int(q), "relevant_docs": [submission_doc_id(d) for d in docs[q]],
                "relevant_chunks": chunks[q]} for q in sorted(docs)]
    assert [e["id"] for e in entries] == list(range(1, 1201))
    path = os.path.join(D, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False)
    zp = zip_submission(path)
    size = os.path.getsize(zp)
    flag = "OK" if size < LIMIT else "VƯỢT 100 MB"
    print(f"-> {zp} ({size / 1e6:.1f} MB) {flag}")
    return size < LIMIT


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m-doc", type=int, default=160)
    ap.add_argument("--m-chunk", type=int, default=120)
    ap.add_argument("--cap", type=int, default=2050)
    ap.add_argument("--w-new", type=float, default=0.6, help="trọng số RRF của reranker fine-tune")
    args = ap.parse_args()

    docs = doc_lists(args.m_doc, args.w_new)
    s = sota()
    same = sum(docs["bge"][q] == [str(d) for d in s[q]["relevant_docs"]] or
               [submission_doc_id(d) for d in docs["bge"][q]] == s[q]["relevant_docs"] for q in s)
    print(f"[doc] bge d{args.m_doc} trùng relevant_docs SOTA: {same}/1200")
    for k in ("ftrr", "ens"):
        ov = statistics.mean(len(set(docs[k][q]) & set(docs["bge"][q])) for q in docs[k])
        print(f"[doc] {k}: trùng bge {ov:.1f}/{args.m_doc} doc mỗi câu")
    ov = statistics.mean(len(set(docs["ens"][q]) & set(docs["ftrr"][q])) for q in docs["ens"])
    print(f"[doc] ens: trùng ftrr {ov:.1f}/{args.m_doc} doc mỗi câu")

    c100 = {q: r["relevant_chunks"] for q, r in s.items()}
    c120 = chunk_list(args.m_chunk, args.cap)
    ct = f"c{args.m_chunk}_ctop60+top300_cap{args.cap}"
    w = f"w{args.w_new:g}"
    plan = [
        (f"fuse_d{args.m_doc}_c100_ftrr_qc_orig_ctop60+top300_cap2700", docs["ftrr"], c100),
        (f"priv_d{args.m_doc}_bge_{ct}", docs["bge"], c120),
        (f"priv_d{args.m_doc}_ftrr_{ct}", docs["ftrr"], c120),
        (f"priv_d{args.m_doc}_ens_ftrr{w}_bge_{ct}", docs["ens"], c120),
    ]
    ok = all([write(n, d, c) for n, d, c in plan])
    if not ok:
        raise SystemExit("Có bài vượt 100 MB: chạy lại với --cap nhỏ hơn.")


if __name__ == "__main__":
    main()
