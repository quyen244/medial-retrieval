"""Làm sạch data/doc_texts.jsonl ĐÃ CÓ, không cần đọc lại data/raw_documents.

File hiện có sinh bằng code CŨ (trước commit 60d6820): trường `text` (title |
description | heading_path | lead, đã nối bằng " | ") chưa bóc cú pháp
markdown, chưa tẩy rác khuôn mẫu. build_doc_texts.py bản mới áp hai bước đó
cho TỪNG field rồi mới nối, nên không chạy lại nguyên vẹn được nếu không có
data/raw_documents (bước đó là build lại từ đầu, không phải làm sạch).

Script này chạy strip_markdown() + denoise() (scripts/md_strip.py,
scripts/text_denoise.py) thẳng trên trường `text` đã ghép sẵn. Dấu `|` ngăn
field cũng bị xoá theo (thành khoảng trắng) như bảng markdown thật - chấp
nhận được vì BGE-M3 không cần ký hiệu phân field, chỉ cần nội dung.

Giới hạn so với build lại: không tái tạo được bước "cắt description bị lặp
trong thân bài" (cần biết ranh giới desc/lead riêng), ảnh hưởng ~6,2% doc vi,
~0,5% ngân sách ký tự toàn tập - nhỏ, bỏ qua được (docs/phat-hien-tang-diem.md
mục 12 phân biệt rõ hai cơ chế này). Cũng không refill được ngân sách vừa
giải phóng bằng nội dung mới, vì text gốc dài hơn đã bị cắt từ trước.

Chạy:
    python -m scripts.denoise_doc_texts
    python -m scripts.denoise_doc_texts --in-place      # ghi đè file cũ sau khi xong
"""
import argparse
import json
import os
import re
import shutil
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.md_strip import strip_markdown
from scripts.text_denoise import denoise

_WS = re.compile(r"[ \t]{2,}")


def clean_text(t: str) -> str:
    t = strip_markdown(t)
    t = denoise(t)
    return _WS.sub(" ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/doc_texts.jsonl")
    ap.add_argument("--out", default="data/doc_texts.clean.jsonl")
    ap.add_argument("--in-place", action="store_true",
                     help="ghi xong rồi mv đè lên --in (bỏ qua --out)")
    args = ap.parse_args()

    out_path = args.inp + ".tmp" if args.in_place else args.out
    lens_before, lens_after, n = [], [], 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(args.inp, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            before = r["text"]
            after = clean_text(before)
            lens_before.append(len(before))
            lens_after.append(len(after))
            r["text"] = after
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
            if n % 100_000 == 0:
                print(f"  {n:,}", flush=True)

    if args.in_place:
        shutil.move(out_path, args.inp)
        out_path = args.inp

    lens_before.sort()
    lens_after.sort()
    saved = sum(lens_before) - sum(lens_after)
    print(f"\n[denoise_doc_texts] {n:,} dòng -> {out_path}")
    print(f"  độ dài trước: median={lens_before[n//2]} mean={statistics.mean(lens_before):.0f}")
    print(f"  độ dài sau : median={lens_after[n//2]} mean={statistics.mean(lens_after):.0f}")
    print(f"  tổng ký tự giảm: {saved:,} ({100*saved/sum(lens_before):.1f}%)")


if __name__ == "__main__":
    main()
