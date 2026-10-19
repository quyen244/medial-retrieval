"""Ghép icd10_vn_aliases_v1.csv với icd10cm_2027.csv thành bảng ICD song ngữ vi-en.

Bảng này dùng ở hai chỗ (xem docs/pipeline-sinh-nhan-reranker.md muc 7.4):
  - Bước 2 sinh query: chunk EN -> tra ra tên bệnh tiếng Việt -> chèn vào prompt
    làm ràng buộc từ ngữ, ép query dùng giọng Việt thay vì dịch máy móc.
  - Inference: THÊM tên bệnh tiếng Anh vào nhánh sparse, không thay thế từ gốc.

Mã ICD là cầu nối: nó là định danh quốc tế nên việc ghép kiểm chứng được,
khác hẳn với dịch bằng LLM.

Hai điều cần biết về dữ liệu:
  1. ICD-10-CM (Mỹ) chi tiết hơn ICD-10 (WHO, bản Việt dùng). Chỉ 42,7% mã khớp
     tuyệt đối; phần còn lại phải gom các mã con về mã cha, chọn tên chung nhất.
  2. ICD-10-CM gắn thêm định tố không có trong bản WHO ("right side",
     "initial encounter", "first trimester"). Phải cắt bỏ, nếu không thuật ngữ
     mở rộng sẽ nhiễu.
"""
import argparse
import collections
import csv
import re
from pathlib import Path

csv.field_size_limit(10**9)

VN_CSV = Path("data/knowledge/icd10_vn_aliases_v1.csv")
CM_CSV = Path("data/knowledge/icd10cm_2027.csv")
OUT_TSV = Path("data/vocab/icd_bilingual.tsv")

# Chương không mang nội dung bệnh: Z yếu tố ảnh hưởng sức khỏe, V/W/X/Y ngoại nhân,
# U mã tạm. Alias của chúng là các cụm tiếng Việt quá chung ("tư vấn", "khó khăn")
# nên khớp bừa vào query.
SKIP_CHAPTERS = set("ZVWXYU")

# Định tố đuôi của ICD-10-CM, cắt lặp cho tới khi không còn.
QUALIFIERS = re.compile(
    r",\s*("
    r"unspecified(\s+\w+)?|"
    r"(right|left|bilateral)(\s+\w+)*|"
    r"(initial|subsequent)\s+encounter.*|"
    r"sequela.*|"
    r"(first|second|third|unspecified)\s+trimester|"
    r"with(out)?\s+bleeding|"
    r"not\s+elsewhere\s+classified|"
    r"so\s+stated|"
    r"in\s+diseases\s+classified\s+elsewhere"
    r")\s*$",
    re.IGNORECASE,
)


LEADING = re.compile(r"^(unspecified|other|other specified)\s+", re.IGNORECASE)


def clean_en(name: str) -> str:
    """Cắt định tố của ICD-10-CM để còn lại thuật ngữ bệnh dùng được.

    Định tố nằm cả hai đầu: đuôi ("Allergy, unspecified") và đầu
    ("Unspecified epiphora"). Chỉ cắt đầu khi còn lại ít nhất một từ.
    """
    prev = None
    while prev != name:
        prev = name
        name = QUALIFIERS.sub("", name).strip().rstrip(",")
    stripped = LEADING.sub("", name).strip()
    return stripped or name


def load_vn():
    rows = {}
    with VN_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["code"]] = (r["name"], r.get("aliases") or "")
    return rows


def load_cm():
    rows = {}
    with CM_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["code"]] = r["name"]
    return rows


def build_prefix_index(cm):
    """Gom mã CM theo tiền tố, để mã WHO cha tìm được các mã CM con."""
    idx = collections.defaultdict(list)
    for code, name in cm.items():
        for k in (3, 4, 5, 6):
            if len(code) >= k:
                idx[code[:k].rstrip(".")].append((code, name))
    return idx


def pick_generic(cands):
    """Trong các mã con, lấy tên chung nhất: ưu tiên 'unspecified', rồi tên ngắn nhất.

    Mã cha WHO mang nghĩa rộng, nên tên tương ứng phải là tên rộng. Lấy bừa một
    mã con sẽ cho ra thuật ngữ quá hẹp (ví dụ 'Typhoid arthritis' cho A01.0).
    """
    uns = [c for c in cands if "unspecified" in c[1].lower()]
    pool = uns or cands
    return min(pool, key=lambda c: (len(c[1]), c[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-all-chapters", action="store_true",
                    help="Giữ cả chương Z/V/W/X/Y/U (mặc định là bỏ)")
    ap.add_argument("--out", default=str(OUT_TSV))
    args = ap.parse_args()

    vn, cm = load_vn(), load_cm()
    idx = build_prefix_index(cm)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = collections.Counter()
    rows = []
    for code, (vi_name, vi_aliases) in sorted(vn.items()):
        if not args.keep_all_chapters and code[0] in SKIP_CHAPTERS:
            stats["bỏ_chương"] += 1
            continue

        if code in cm:
            en_name, how = cm[code], "exact"
        else:
            cands = idx.get(code.rstrip(".")) or idx.get(code[:3])
            if not cands:
                stats["không_khớp"] += 1
                continue
            en_name, how = pick_generic(cands)[1], "rollup"
        stats[how] += 1

        # Alias tiếng Việt: bỏ alias 1 từ, chúng quá chung nên khớp bừa.
        aliases = [a.strip() for a in vi_aliases.split("|") if len(a.strip().split()) >= 2]
        aliases = sorted({a for a in aliases if 5 < len(a) < 60})

        rows.append((code, vi_name, "|".join(aliases), en_name, clean_en(en_name), how))

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["code", "vi_name", "vi_aliases", "en_name_raw", "en_term", "join"])
        w.writerows(rows)

    print(f"-> {out_path}  ({len(rows)} dòng)")
    for k, v in stats.most_common():
        print(f"   {k:12} {v}")


if __name__ == "__main__":
    main()
