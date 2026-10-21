"""Ghép bài nộp đầy đủ từ bài nền và bài chạy thử trên N query đầu, và so sánh
các biến thể với nhau.

Bài nộp thiếu dù một id là bị loại (xem run_full_submission.py), nên bài chạy
`--limit 200` không nộp thẳng được. Cách làm công bằng: lấy bài nền A (đủ 1200
query), thay đúng các id mà biến thể có chạy bằng kết quả của biến thể. Hai bài
ghép chỉ khác nhau ở N query đó, nên chênh điểm là do N query đó.

    # ghép
    python -m scripts.compose_submission merge \\
        --base tmp/sub/A_full.json --overlay tmp/sub/B_200.json --out tmp/sub/B_merged.json

    # so sánh độ chồng lấn doc/chunk giữa các biến thể (không cần nhãn)
    python -m scripts.compose_submission compare --a tmp/sub/A_200.json \\
        --b tmp/sub/B_200.json --b tmp/sub/Bp_200.json --b tmp/sub/C_200.json
"""
import argparse
import json
import sys
from typing import Dict, List


def load(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cmd_merge(args) -> int:
    base = load(args.base)
    overlay = {e["id"]: e for e in load(args.overlay)}
    ids = [e["id"] for e in base]
    unknown = [i for i in overlay if i not in set(ids)]
    if unknown:
        print(f"Lỗi: overlay có id không nằm trong bài nền: {unknown[:5]}")
        return 1
    out, replaced, empty = [], 0, 0
    for e in base:
        o = overlay.get(e["id"])
        if o is not None:
            if not o["relevant_chunks"]:
                empty += 1
            e, replaced = o, replaced + 1
        out.append(e)
    assert [e["id"] for e in out] == ids, "thứ tự id bị đổi"
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"{args.out}: {len(out)} entry, thay {replaced} từ overlay ({empty} entry rỗng)")
    return 0


def key(c: dict) -> str:
    return f"{c['doc_id']}|{hash(c['chunk_text'])}"


def cmd_compare(args) -> int:
    a = {e["id"]: e for e in load(args.a)}
    for path in args.b:
        b = {e["id"]: e for e in load(path)}
        common = sorted(set(a) & set(b))
        doc_j, chunk_j, top1 = [], [], 0
        for i in common:
            da, db = set(a[i]["relevant_docs"]), set(b[i]["relevant_docs"])
            ca = {key(c) for c in a[i]["relevant_chunks"]}
            cb = {key(c) for c in b[i]["relevant_chunks"]}
            if da | db:
                doc_j.append(len(da & db) / len(da | db))
            if ca | cb:
                chunk_j.append(len(ca & cb) / len(ca | cb))
            if a[i]["relevant_chunks"] and b[i]["relevant_chunks"]:
                top1 += key(a[i]["relevant_chunks"][0]) == key(b[i]["relevant_chunks"][0])
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        print(f"{path} vs {args.a}: {len(common)} query chung | "
              f"Jaccard doc {mean(doc_j):.3f} | Jaccard chunk {mean(chunk_j):.3f} | "
              f"cùng chunk đầu {top1}/{len(common)}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--base", required=True)
    m.add_argument("--overlay", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(fn=cmd_merge)
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True, help="bài chuẩn so sánh (thường là A)")
    c.add_argument("--b", action="append", required=True, help="các bài cần so, lặp cờ được")
    c.set_defaults(fn=cmd_compare)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
