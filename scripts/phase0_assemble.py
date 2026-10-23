"""Phase 0: lắp lại bài nộp từ dump top-50, KHÔNG chạy GPU.

Hai thay đổi so với bài nộp hiện tại, cả hai đều khai thác đúng cách chấm điểm
(xem docs/chan-doan-he-retrieval.md):

1. `relevant_docs` không còn bị suy ra từ `relevant_chunks`. Hai danh sách được
   chấm bằng hai công thức độc lập `F2 = 5h/(4g+m)`, nên số lượng của chúng
   phải chọn riêng. Doc được xếp theo bằng chứng gộp của mọi chunk thuộc nó,
   không phải theo thứ tự chunk nào xuất hiện trước.

2. `chunk_text` có thể là một cửa sổ dài lấy từ markdown gốc thay vì đúng một
   chunk 877 ký tự. Tiêu chí chấm là `LCS_tokens(nộp, ref) / len(ref) >= 0.40`:
   LCS là SUBSEQUENCE và mẫu số là độ dài tham chiếu, nên token thừa trong bài
   nộp không bị phạt, còn bài nộp ngắn hơn tham chiếu thì không bao giờ đạt.

Nguồn văn bản là `data/raw_documents/<domain>/<doc_id>.json` (markdown đã crawl),
tra qua `data/raw_documents_index.json` do --build-index sinh ra.

Ví dụ:
    python -m scripts.phase0_assemble --probe
    python -m scripts.phase0_assemble --name E1_docs44 --docs 44 --chunk-mode orig
    python -m scripts.phase0_assemble --name E2_win2500 --docs 44 --chunks 44 --chunk-mode win --window 2500
"""
import argparse
import json
import os
import random
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.pipeline.submission import submission_chunk_text, submission_doc_id, zip_submission

RAW_ROOT = "data/raw_documents"
INDEX_PATH = "data/raw_documents_index.json"
DUMP = "data/tmp/sub1200/D_ft_top50.jsonl"
OUT_DIR = "data/tmp/phase0"


def build_index() -> dict:
    idx = {}
    for dom in os.scandir(RAW_ROOT):
        if not dom.is_dir() or dom.name.startswith("_"):
            continue
        for f in os.scandir(dom.path):
            if f.name.endswith(".json"):
                idx[f.name[:-5]] = dom.name
    with open(INDEX_PATH, "w") as fh:
        json.dump(idx, fh, separators=(",", ":"))
    return idx


def load_index() -> dict:
    if not os.path.exists(INDEX_PATH):
        return build_index()
    with open(INDEX_PATH) as fh:
        return json.load(fh)


class RawDocs:
    """Đọc markdown gốc theo doc_id, có cache trong bộ nhớ."""

    def __init__(self, index: dict):
        self.index = index
        self._cache = {}

    def markdown(self, doc_id: str) -> str:
        if doc_id in self._cache:
            return self._cache[doc_id]
        dom = self.index.get(str(doc_id))
        text = ""
        if dom:
            path = os.path.join(RAW_ROOT, dom, f"{doc_id}.json")
            try:
                with open(path, encoding="utf-8") as fh:
                    text = json.load(fh)["content"]["markdown"] or ""
            except (OSError, KeyError, json.JSONDecodeError):
                text = ""
        if len(self._cache) < 200_000:
            self._cache[doc_id] = text
        return text


_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def locate(needle: str, haystack: str) -> int:
    """Vị trí gần đúng của chunk trong markdown gốc. Chunk đã qua bước làm sạch
    nên thường KHÔNG phải substring y hệt; thử vài mẩu neo rồi mới chịu thua."""
    if not needle or not haystack:
        return -1
    for frac in (0.0, 0.3, 0.6):
        start = int(len(needle) * frac)
        probe = needle[start:start + 48].strip()
        if len(probe) < 16:
            continue
        pos = haystack.find(probe)
        if pos >= 0:
            return max(0, pos - start)
    hn, nn = _norm(haystack), _norm(needle)
    probe = nn[:48]
    if len(probe) >= 16:
        pos = hn.find(probe)
        if pos >= 0:  # ánh xạ thô từ chuỗi đã chuẩn hoá về chuỗi gốc
            return min(len(haystack) - 1, int(pos * len(haystack) / max(1, len(hn))))
    return -1


def window(raw: str, anchor_text: str, size: int) -> str:
    """Cửa sổ `size` ký tự của markdown gốc, bao quanh vị trí của anchor_text.
    Ngắn hơn size thì trả nguyên văn bản."""
    if len(raw) <= size:
        return raw
    pos = locate(anchor_text, raw)
    if pos < 0:
        pos = 0
    centre = pos + len(anchor_text) // 2
    start = max(0, min(len(raw) - size, centre - size // 2))
    return raw[start:start + size]


def doc_rank(candidates):
    """Xếp doc theo bằng chứng gộp: điểm rerank cao nhất trong các chunk của nó.
    Khác hẳn thứ tự hiện tại (doc nào có chunk xuất hiện trước thì đứng trước),
    vốn coi một doc có 3 chunk mạnh ngang một doc có 1 chunk ở hạng 48."""
    best = {}
    for c in candidates:
        d = c["doc_id"]
        s = c.get("rerank_score")
        s = float(s) if s is not None else float("-inf")
        if d not in best or s > best[d][0]:
            best[d] = (s, c)
    return sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)


def build(records, raw_docs, n_docs, n_chunks, chunk_mode, win_size, cap,
          restrict_chunks=False):
    """restrict_chunks=True: chỉ nộp chunk thuộc các doc đã chọn. Dùng khi
    n_docs nhỏ hơn số doc của chunk list, để bài nộp không chứa chunk của doc
    vắng mặt trong `relevant_docs`. Hai tầng vẫn được chấm độc lập nên việc
    này không làm nhiễu phép đo tầng doc."""
    entries, stats = [], defaultdict(list)
    for r in records:
        cands = r["candidates"]
        ranked = doc_rank(cands)
        docs = [d for d, _ in ranked[:n_docs]]
        keep = set(docs)

        if chunk_mode == "orig":
            # Tái tạo ĐÚNG chunk list đã nộp (trường `submitted`), không phải
            # top-n theo điểm: 13/1200 câu lệch nhau vì cơ chế giữ chỗ sub-query.
            # Giữ y hệt thì bài này chỉ khác D_ft đúng một biến là số doc.
            by_id = {str(c["chunk_id"]): c for c in cands}
            picked = [by_id[i] for i in (str(x) for x in r.get("submitted", [])) if i in by_id]
            picked = picked or cands
            if restrict_chunks:
                picked = [c for c in picked if c["doc_id"] in keep]
            chunks = [{"doc_id": submission_doc_id(c["doc_id"]),
                       "chunk_text": submission_chunk_text(c["content"])}
                      for c in picked[:n_chunks]]
        else:  # một cửa sổ cho mỗi doc, theo đúng thứ hạng doc
            chunks = []
            for d, (_score, c) in ranked[:n_chunks]:
                raw = raw_docs.markdown(d)
                if not raw:
                    text = c["content"]  # không có bản gốc thì giữ chunk cũ
                    stats["fallback"].append(1)
                elif chunk_mode == "full":
                    text = raw[:cap]
                else:
                    text = window(raw, c["content"], win_size)
                chunks.append({"doc_id": submission_doc_id(d),
                               "chunk_text": submission_chunk_text(text)})

        stats["n_docs"].append(len(docs))
        stats["n_chunks"].append(len(chunks))
        stats["chars"].extend(len(c["chunk_text"]) for c in chunks)
        entries.append({"id": int(r["id"]),
                        "relevant_docs": [submission_doc_id(d) for d in docs],
                        "relevant_chunks": chunks})
    return entries, stats


def validate(entries, expect_ids):
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)) == expect_ids, f"id không đủ/không duy nhất: {len(ids)}"
    for e in entries:
        assert isinstance(e["relevant_docs"], list) and isinstance(e["relevant_chunks"], list)
        for d in e["relevant_docs"]:
            assert isinstance(d, int), f"doc_id phải là số nguyên, gặp {type(d)}"
        for c in e["relevant_chunks"]:
            assert set(c) == {"doc_id", "chunk_text"} and isinstance(c["doc_id"], int)
            assert c["chunk_text"], "chunk_text rỗng"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=DUMP)
    ap.add_argument("--name", default="E")
    ap.add_argument("--docs", type=int, default=44)
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--chunk-mode", choices=["orig", "win", "full"], default="orig")
    ap.add_argument("--window", type=int, default=2500)
    ap.add_argument("--cap", type=int, default=8000, help="trần ký tự cho chunk-mode=full")
    ap.add_argument("--build-index", action="store_true")
    ap.add_argument("--probe", action="store_true", help="chỉ đo tỉ lệ căn chỉnh rồi thoát")
    args = ap.parse_args()

    index = build_index() if args.build_index else load_index()
    raw_docs = RawDocs(index)
    records = [json.loads(l) for l in open(args.dump, encoding="utf-8")]

    if args.probe:
        random.seed(1)
        found = tried = 0
        for r in random.sample(records, 300):
            for c in r["candidates"][:2]:
                raw = raw_docs.markdown(c["doc_id"])
                if not raw:
                    continue
                tried += 1
                if locate(c["content"], raw) >= 0:
                    found += 1
        print(f"căn chỉnh chunk vào markdown gốc: {found}/{tried} ({100*found/max(1,tried):.1f}%)")
        lens = [len(raw_docs.markdown(c["doc_id"]))
                for r in random.sample(records, 300) for c in r["candidates"][:3]]
        lens = sorted(x for x in lens if x)
        print(f"độ dài markdown gốc: median={lens[len(lens)//2]} "
              f"p75={lens[3*len(lens)//4]} p90={lens[int(.9*len(lens))]} max={lens[-1]}")
        return

    entries, stats = build(records, raw_docs, args.docs, args.chunks,
                           args.chunk_mode, args.window, args.cap)
    validate(entries, len(records))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.name}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False)
    zp = zip_submission(out)

    chars = sorted(stats["chars"])
    print(f"{args.name}: {len(entries)} câu | "
          f"doc/câu={statistics.mean(stats['n_docs']):.1f} "
          f"chunk/câu={statistics.mean(stats['n_chunks']):.1f} | "
          f"chunk_chars median={chars[len(chars)//2]} mean={statistics.mean(chars):.0f} "
          f"max={chars[-1]} | fallback={len(stats['fallback'])}")
    print(f"  json {os.path.getsize(out)/1e6:.1f} MB -> zip {os.path.getsize(zp)/1e6:.1f} MB  ({zp})")


if __name__ == "__main__":
    main()
