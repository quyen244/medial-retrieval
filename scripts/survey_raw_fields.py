"""Khảo sát các trường có sẵn trong data/raw_documents để thiết kế doc-level ingest.

Câu hỏi cần trả lời trước khi encode lại 1M tài liệu:
  - `source.description` phủ bao nhiêu phần trăm, dài bao nhiêu? Nó là bản tóm
    tắt do CMS/tác giả viết sẵn, quý hơn nhiều so với việc cắt 600 ký tự đầu.
  - Bao nhiêu tài liệu thật sự có heading? (`quality.has_headings`)
  - `source.language` và `source.language_category` lệch nhau bao nhiêu?
    Mẫu ask.39.net/2136736 có language='en' nhưng language_category='zh'.
  - Mỗi loại nguồn (báo vi / PubMed en / hỏi đáp zh) có cấu trúc khác nhau ra sao.

Chỉ đọc file trên đĩa, không cần Qdrant lẫn GPU.
"""
import argparse
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW = "data/raw_documents"
INDEX = "data/raw_documents_index.json"


def pct(n, d):
    return f"{100 * n / d:5.1f}%" if d else "    -"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    idx = json.load(open(INDEX))
    random.seed(args.seed)
    doc_ids = random.sample(list(idx), args.sample)

    by_lang = defaultdict(lambda: {
        "n": 0, "desc": 0, "desc_len": [], "headings": 0, "title": 0,
        "chars": [], "lang_mismatch": 0, "too_short": 0, "empty": 0})
    domains = Counter()
    mismatch_examples = []

    for did in doc_ids:
        try:
            with open(os.path.join(RAW, idx[did], did + ".json"), encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        src, con, q = d.get("source", {}), d.get("content", {}), d.get("quality", {})
        lang = src.get("language_category") or src.get("language") or "?"
        s = by_lang[lang]
        s["n"] += 1
        domains[src.get("domain", "?")] += 1
        desc = (src.get("description") or "").strip()
        if desc:
            s["desc"] += 1
            s["desc_len"].append(len(desc))
        if q.get("has_headings"):
            s["headings"] += 1
        if q.get("has_title"):
            s["title"] += 1
        if q.get("too_short"):
            s["too_short"] += 1
        if q.get("is_empty"):
            s["empty"] += 1
        s["chars"].append(con.get("char_count", 0))
        if src.get("language") and src.get("language_category") and \
                src["language"] != src["language_category"]:
            s["lang_mismatch"] += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append(
                    (did, src.get("domain"), src["language"], src["language_category"]))

    print(f"mẫu {sum(s['n'] for s in by_lang.values())} tài liệu "
          f"trên {len(domains)} domain\n")
    hdr = (f"{'lang':5s} {'n':>6s} {'có desc':>9s} {'desc median':>12s} "
           f"{'có heading':>11s} {'có title':>9s} {'lang lệch':>10s} "
           f"{'content median':>15s}")
    print(hdr)
    print("-" * len(hdr))
    for lang, s in sorted(by_lang.items(), key=lambda kv: -kv[1]["n"]):
        dl = sorted(s["desc_len"])
        ch = sorted(s["chars"])
        print(f"{lang:5s} {s['n']:6d} {pct(s['desc'], s['n']):>9s} "
              f"{(dl[len(dl)//2] if dl else 0):12d} "
              f"{pct(s['headings'], s['n']):>11s} {pct(s['title'], s['n']):>9s} "
              f"{pct(s['lang_mismatch'], s['n']):>10s} "
              f"{(ch[len(ch)//2] if ch else 0):15d}")

    if mismatch_examples:
        print("\nví dụ language != language_category:")
        for did, dom, a, b in mismatch_examples:
            print(f"   {did:>9s} {dom:28s} language={a!r} category={b!r}")

    print("\n10 domain nhiều nhất trong mẫu:")
    for dom, n in domains.most_common(10):
        print(f"   {dom:34s} {n:5d}  ({100*n/args.sample:.1f}%)")


if __name__ == "__main__":
    main()
