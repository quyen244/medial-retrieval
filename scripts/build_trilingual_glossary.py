"""Dựng bảng thuật ngữ y khoa vi - en - zh, dùng làm few-shot/ràng buộc khi dịch query.

Hai nguồn, hai cầu nối khác nhau:

  ICD-11 (data/knowledge/icd11/SimpleTabulation-ICD-11-MMS-{en,zh}.txt)
      File zh có SẴN cả cột TitleEN lẫn Title(zh) trên cùng một dòng, chung
      Foundation URI. Tức en<->zh do WHO căn, không phải ta đoán. Đây là nguồn
      sạch nhất, thay hẳn cho ý định tra Wikidata (Wikidata nhiễu: K29 là viêm
      dạ dày nhưng trả về 胰酶 pancreatin).

  ICD-10 (data/vocab/icd_bilingual.tsv)
      vi <-> en, cầu nối là mã ICD-10. Đã dựng từ trước.

Ghép hai bên bằng CHUỖI TIẾNG ANH đã chuẩn hoá, vì mã ICD-10 và ICD-11 là hai
hệ khác nhau, không ánh xạ 1-1. Ghép theo chuỗi nên chỉ phủ được phần tên gọi
trùng nhau giữa hai phiên bản - script in ra độ phủ thật để biết mà liệu.

Đầu ra data/vocab/trilingual_terms.tsv: en_term, vi_term, zh_term, nguồn.
"""
import argparse
import csv
import os
import re
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

csv.field_size_limit(10 ** 9)

ICD11_ZH = "data/knowledge/icd11/SimpleTabulation-ICD-11-MMS-zh.txt"
ICD10_VI_EN = "data/vocab/icd_bilingual.tsv"
OUT = "data/vocab/trilingual_terms.tsv"

_LEAD = re.compile(r"^[-\s]+")
_PAREN = re.compile(r"\s*\([^)]*\)")
_NONWORD = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def norm_en(s: str) -> str:
    """Chuẩn hoá để ghép: bỏ dấu gạch đầu dòng của ICD, bỏ ngoặc đơn bổ nghĩa,
    bỏ dấu câu, gộp khoảng trắng, hạ chữ thường."""
    s = unicodedata.normalize("NFKC", s or "").strip().strip('"')
    s = _LEAD.sub("", s)
    s = _PAREN.sub("", s)
    s = _NONWORD.sub(" ", s)
    return _WS.sub(" ", s).strip().lower()


def clean_title(s: str) -> str:
    return _LEAD.sub("", unicodedata.normalize("NFKC", s or "").strip().strip('"')).strip()


def load_icd11_en_zh(path):
    """en -> zh. Chỉ lấy dòng có cả hai title và là mục thực thể, bỏ chapter/block
    vì chúng là tiêu đề nhóm, không phải tên bệnh tra cứu được."""
    pairs, stats = {}, Counter()
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            stats["dòng"] += 1
            kind = (row.get("ClassKind") or "").strip()
            en, zh = clean_title(row.get("TitleEN")), clean_title(row.get("Title"))
            if not en or not zh:
                stats["thiếu một phía"] += 1
                continue
            if kind in ("chapter", "block"):
                stats[f"bỏ {kind}"] += 1
                continue
            key = norm_en(en)
            if len(key) < 3:
                continue
            pairs.setdefault(key, (en, zh))
            stats["dùng được"] += 1
    return pairs, stats


def load_icd10_vi_en(path):
    """en đã chuẩn hoá -> (en gốc, vi). Ưu tiên vi_name, bỏ alias quá ngắn."""
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            en = clean_title(row.get("en_term") or row.get("en_name_raw") or "")
            vi = clean_title(row.get("vi_name") or "")
            if not en or not vi:
                continue
            key = norm_en(en)
            if len(key) >= 3:
                out.setdefault(key, (en, vi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    en_zh, st = load_icd11_en_zh(ICD11_ZH)
    print(f"ICD-11 zh: {dict(st)}")
    print(f"  -> {len(en_zh):,} cặp en<->zh (WHO căn sẵn, không phải ta đoán)")

    vi_en = load_icd10_vi_en(ICD10_VI_EN)
    print(f"ICD-10 vi<->en: {len(vi_en):,} cặp")

    both = set(en_zh) & set(vi_en)
    print(f"\nghép theo chuỗi tiếng Anh chuẩn hoá: {len(both):,} thuật ngữ có ĐỦ BA ngôn ngữ")
    print(f"  phủ {100*len(both)/max(1,len(vi_en)):.1f}% phía vi, "
          f"{100*len(both)/max(1,len(en_zh)):.1f}% phía zh")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n3 = n2 = 0
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["key_en", "en", "vi", "zh", "nguon"])
        for key in sorted(set(en_zh) | set(vi_en)):
            en11 = en_zh.get(key)
            vi10 = vi_en.get(key)
            en = (en11 or vi10)[0]
            zh = en11[1] if en11 else ""
            vi = vi10[1] if vi10 else ""
            src = "icd11+icd10" if (en11 and vi10) else ("icd11" if en11 else "icd10")
            w.writerow([key, en, vi, zh, src])
            n3 += bool(vi and zh)
            n2 += bool(vi) != bool(zh)
    print(f"\nghi {args.out}: {n3:,} dòng đủ ba ngôn ngữ, {n2:,} dòng chỉ hai")

    print("\n10 ví dụ đủ ba ngôn ngữ:")
    shown = 0
    for key in sorted(both, key=len):
        if len(key) < 6:
            continue
        en, zh = en_zh[key][0], en_zh[key][1]
        vi = vi_en[key][1]
        print(f"   {en[:44]:44s} | {vi[:34]:34s} | {zh}")
        shown += 1
        if shown >= 10:
            break


if __name__ == "__main__":
    main()
