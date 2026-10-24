"""Tải bảng ICD-11 MMS chính thức từ WHO (en + zh) và ghép thành một bảng song ngữ.

Nguồn: icdcdn.who.int xuất sẵn "Simple Tabulation" cho mỗi ngôn ngữ, đây là
file tĩnh WHO công bố công khai, không cần đăng ký ICD-API, không cần cào
trang https://icd.who.int/browse (trang đó chỉ là UI JS render từ cùng dữ
liệu này).

Vì sao dùng nguồn này thay Wikidata (xem docs/ke-hoach-phase-1-2.md muc 5.3):
Wikidata P494 có nhiễu đã đo được (K29 -> 胰酶 sai). Bảng WHO chính thức
không có loại nhiễu ánh xạ đó vì tiếng Trung và tiếng Anh cùng một hàng,
cùng một Foundation URI, do chính WHO dịch.

Hai file en/zh join bằng "Foundation URI" (định danh ổn định của thực thể,
không đổi giữa các bản ICD-11). Chỉ giữ hàng có Code (category thật, bỏ
block/chapter vì chúng không phải đơn vị chẩn đoán).
"""
import argparse
import csv
import io
import re
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

csv.field_size_limit(10**9)

RELEASE = "2026-01"
BASE = f"https://icdcdn.who.int/static/releasefiles/{RELEASE}"
RAW_DIR = Path("data/knowledge/icd11")
OUT_TSV = Path("data/vocab/icd11_en_zh.tsv")

LEADING_DASH = re.compile(r"^[\-\s]+")


def download(lang: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / f"SimpleTabulation-ICD-11-MMS-{lang}.zip"
    txt_path = RAW_DIR / f"SimpleTabulation-ICD-11-MMS-{lang}.txt"
    if txt_path.exists():
        return txt_path

    url = f"{BASE}/SimpleTabulation-ICD-11-MMS-{lang}.zip"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as resp:
        data = resp.read()
    zip_path.write_bytes(data)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        member = f"SimpleTabulation-ICD-11-MMS-{lang}.txt"
        txt_path.write_bytes(zf.read(member))
    return txt_path


def load_rows(path: Path) -> dict:
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("ClassKind") != "category" or not (r.get("Code") or "").strip():
                continue
            uri = r["Foundation URI"]
            rows[uri] = r
    return rows


def clean_title(title: str) -> str:
    return LEADING_DASH.sub("", title or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_TSV))
    args = ap.parse_args()

    en_path = download("en")
    zh_path = download("zh")

    en_rows = load_rows(en_path)
    zh_rows = load_rows(zh_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["foundation_uri", "code", "chapter_no", "title_en", "title_zh"])
        for uri, en in en_rows.items():
            zh = zh_rows.get(uri)
            if not zh:
                continue
            w.writerow([
                uri,
                en["Code"],
                en["ChapterNo"],
                clean_title(en["Title"]),
                clean_title(zh["Title"]),
            ])
            n += 1

    print(f"-> {out_path}  ({n} dòng, en có {len(en_rows)}, zh có {len(zh_rows)})")


if __name__ == "__main__":
    main()
