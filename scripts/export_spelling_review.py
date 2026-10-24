"""Xuất kết quả sửa chính tả ra file để duyệt bằng mắt.

    python -m scripts.export_spelling_review                    # đọc data/spelling/corrected_queries.jsonl

Ra hai file cạnh file vào:
  review_1200.tsv  đủ mọi câu: id, đổi?, câu gốc, câu cuối, chỗ thay theo bảng,
                   chỗ LLM sửa được nhận, chỗ LLM sửa bị trả về
  review_1200.md   chỉ câu bị đổi hoặc có chỗ bị trả về; mỗi chỗ đổi đánh dấu
                   ngay trong câu dạng [cũ → mới]
"""
import argparse
import difflib
import json
import os
import re
import unicodedata

_PIECE = re.compile(r"\s+|\S+")
_NONLETTER = re.compile(r"[\W_]+", re.U)


def letters(s):
    """Chỉ còn chữ/số, thường hoá: hai chuỗi bằng nhau nghĩa là chỉ khác dấu câu,
    khoảng trắng hoặc hoa/thường."""
    return _NONLETTER.sub("", unicodedata.normalize("NFC", s or "").lower())


def inline_diff(before, after):
    """Câu cuối, chỗ nào khác câu gốc thì viết [cũ → mới]."""
    a, b = _PIECE.findall(before), _PIECE.findall(after)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        old, new = "".join(a[i1:i2]), "".join(b[j1:j2])
        if op == "equal":
            out.append(old)
        else:
            lead = old[: len(old) - len(old.lstrip())] if old.strip() else ""
            out.append(f"{lead}[{old.strip() or '∅'} → {new.strip() or '∅'}]"
                       f"{' ' if old.endswith((' ', chr(10))) or new.endswith(' ') else ''}")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/spelling/corrected_queries.jsonl")
    args = ap.parse_args()

    rows = sorted((json.loads(l) for l in open(args.inp, encoding="utf-8")), key=lambda r: int(r["id"]))
    base = os.path.dirname(args.inp)
    tsv, md = os.path.join(base, "review_1200.tsv"), os.path.join(base, "review_1200.md")
    one_line = lambda s: (s or "").replace("\t", " ").replace("\n", " ⏎ ")

    with open(tsv, "w", encoding="utf-8") as f:
        f.write("id\tchanged\toriginal\tfinal\ttable_changes\tllm_kept\tllm_reverted\tflags\n")
        for r in rows:
            table = " | ".join(f"{c['old']} → {c['new']} ({c['kind']})" for c in r.get("normalize_changes", []))
            f.write("\t".join([str(r["id"]), "1" if r["changed"] else "0", one_line(r["query"]),
                               one_line(r["corrected_query"]), one_line(table),
                               one_line(" | ".join(r["edits_kept"])), one_line(" | ".join(r["edits_reverted"])),
                               ",".join(r["flags"])]) + "\n")

    # Câu chỉ đổi dấu câu/khoảng trắng/hoa-thường không đổi chữ nào: tách riêng,
    # đó là phần lớn số "câu bị đổi" nhưng không cần đọc.
    real = [r for r in rows if letters(r["query"]) != letters(r["corrected_query"])]
    rev_only = [r for r in rows if r["edits_reverted"] and r not in real]
    punct = [r for r in rows if r["changed"] and r not in real]
    n_table = sum(bool(r.get("normalize_changes")) for r in rows)
    n_llm = sum(bool(r["edits_kept"]) for r in rows)
    n_rev = sum(bool(r["edits_reverted"]) for r in rows)
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# Duyệt sửa chính tả {len(rows)} câu\n\n")
        f.write(f"- Câu bị đổi: {sum(r['changed'] for r in rows)}, trong đó **đổi chữ: {len(real)}**, "
                f"chỉ đổi dấu câu/khoảng trắng/hoa-thường: {len(punct)}\n")
        f.write(f"- Có chỗ thay theo bảng tiền xử lý: {n_table}\n")
        f.write(f"- Có chỗ LLM sửa được nhận: {n_llm}\n")
        f.write(f"- Có chỗ LLM sửa bị luật trả về: {n_rev}\n\n")
        f.write("Ký hiệu: `[cũ → mới]` là chỗ đã đổi trong câu cuối. Mục *trả về* là chỗ LLM đề xuất "
                "nhưng bị luật hoặc corpus chặn, câu cuối giữ chữ gốc.\n\n")
        def block(r):
            orig = unicodedata.normalize("NFC", r["query"])
            f.write(f"### {r['id']}\n\n{inline_diff(orig, r['corrected_query'])}\n\n")
            for c in r.get("normalize_changes", []):
                f.write(f"- bảng: `{c['old']} → {c['new']}` ({c['kind']})\n")
            for e in r["edits_kept"]:
                if not e.endswith("[format]") or letters(e.split(" -> ")[0]) != letters(e.split(" -> ")[1][:-9]):
                    f.write(f"- LLM nhận: `{e}`\n")
            for e in r["edits_reverted"]:
                f.write(f"- LLM **trả về**: `{e}`\n")
            f.write("\n")

        f.write(f"## 1. Câu bị đổi chữ ({len(real)})\n\n")
        for r in real:
            block(r)
        f.write(f"## 2. Câu giữ nguyên chữ nhưng có chỗ LLM đề xuất bị trả về ({len(rev_only)})\n\n")
        for r in rev_only:
            block(r)
        f.write(f"## 3. Câu chỉ đổi dấu câu/khoảng trắng/hoa-thường ({len(punct)})\n\n")
        f.write(", ".join(str(r["id"]) for r in punct) + "\n")
    shown = real + rev_only
    print(f"{len(rows)} câu -> {tsv}\n{len(shown)} câu cần xem -> {md}")


if __name__ == "__main__":
    main()
