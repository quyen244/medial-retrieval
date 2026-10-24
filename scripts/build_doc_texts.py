"""Dựng văn bản đại diện cho từng tài liệu, để encode thành doc-level index.

Chỉ đọc đĩa, KHÔNG cần Qdrant lẫn GPU. Chạy được trên máy yếu, kết quả đem
sang server để encode.

Vì sao bốn trường này, theo đúng thứ tự này (số đo trên mẫu 4.000 tài liệu):

                vi (65%)   en (24%)   zh (11%)
    title         100%       100%       100%
    description   88.8%        0%       86.1%
    heading       43.7%      100%       11.5%

`description` và `heading` bù trừ cho nhau gần như hoàn hảo: PubMed không có
description nhưng heading 100% (abstract có cấu trúc Background/Methods/...),
còn hỏi đáp tiếng Trung hiếm heading nhưng description 86%. Nên MỘT template
chạy được cả ba loại nguồn, không cần rẽ nhánh theo ngôn ngữ.

Thứ tự xếp theo mật độ tín hiệu giảm dần, để khi cắt theo ngân sách thì thứ
bị bỏ là thứ ít quan trọng nhất.

Hai điều `title` + `description` mang lại mà chunk vector hiện tại KHÔNG có:
  - Chunk ở giữa bài không chứa title. Chunk 521089_c002 mở đầu bằng "Tuy
    nhiên, hầu hết có thể cải thiện bằng..." mà không chỗ nào nhắc tới chủ đề
    "huyết áp thấp".
  - `description` chưa từng được đưa vào Qdrant, nó chỉ nằm ở raw_documents.
Đó là lý do nên ENCODE MỚI thay vì lấy trung bình chunk vector đã có: lấy
trung bình thì tái tạo lại đúng khoảng trống đang cần vá.

Lưu ý `source.language` không đáng tin ('eng', 'pol', None). Dùng
`source.language_category` - đây cũng là trường Qdrant đang dùng.
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW = "data/raw_documents"
INDEX = "data/raw_documents_index.json"

_MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~<])")
_WS = re.compile(r"[ \t]+")
_HEAD = re.compile(r"^#{1,6}\s+(.*)$", re.M)


def clean(s: str) -> str:
    return _WS.sub(" ", _MD_ESCAPE.sub(r"\1", s or "")).strip()


def doc_text(doc: dict, budget: int, head_budget: int, max_heads: int) -> str:
    src, con = doc.get("source", {}), doc.get("content", {})
    md = con.get("markdown") or ""

    title = clean(src.get("title") or "")
    desc = clean(src.get("description") or "")
    heads = [clean(h) for h in _HEAD.findall(md)][:max_heads]
    # Bỏ heading trùng title và heading rỗng/quá ngắn ("Results", "Abstract"
    # vẫn giữ: với PubMed chúng đánh dấu cấu trúc, rẻ và không hại).
    heads = [h for h in heads if h and h != title]
    head_s = " > ".join(heads)[:head_budget]

    body = clean(_HEAD.sub(" ", md))

    # description thường được CMS cắt ra từ chính thân bài, nên lead hay lặp
    # lại nguyên văn nó. Đo trên mẫu ask.39.net: 300/612 ký tự bị trùng, tức
    # gần nửa ngân sách tiêu cho cùng một đoạn. Cắt bản trong thân bài đi.
    if len(desc) >= 40:
        anchor = desc[:60]
        pos = body.find(anchor)
        if pos >= 0:
            body = (body[:pos] + " " + body[pos + len(desc):]).strip()

    # Rác điều hướng của trang hỏi đáp và trang báo. Mỗi mẫu phải neo chắc để
    # không cắt nhầm nội dung thật.
    for junk in ("我要咨询", "医生回答", "向医生提问", "举报", "有帮助",
                 "展开全部", "收起", "Xem thêm", "Chia sẻ"):
        body = body.replace(junk, " ")
    body = _WS.sub(" ", body).strip()

    parts = [p for p in (title, desc, head_s) if p]
    used = sum(len(p) + 3 for p in parts)
    lead = body[: max(0, budget - used)]
    if lead:
        parts.append(lead)
    return " | ".join(parts)[:budget]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/doc_texts.jsonl")
    ap.add_argument("--budget", type=int, default=1500, help="tổng ký tự mỗi doc")
    ap.add_argument("--head-budget", type=int, default=300)
    ap.add_argument("--max-heads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true",
                    help="trộn trước khi cắt --limit; index xếp theo domain nên 3000 id đầu toàn một nguồn")
    args = ap.parse_args()

    idx = json.load(open(INDEX))
    ids = list(idx)
    if args.shuffle:
        import random
        random.Random(0).shuffle(ids)
    if args.limit:
        ids = ids[: args.limit]
    print(f"[doc_texts] {len(ids):,} tài liệu -> {args.out}")

    lens, langs, skipped = [], Counter(), Counter()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for n, did in enumerate(ids, 1):
            path = os.path.join(RAW, idx[did], did + ".json")
            try:
                with open(path, encoding="utf-8") as f:
                    doc = json.load(f)
            except (OSError, json.JSONDecodeError):
                skipped["đọc lỗi"] += 1
                continue
            q = doc.get("quality", {})
            if q.get("is_empty"):
                skipped["rỗng"] += 1
                continue
            text = doc_text(doc, args.budget, args.head_budget, args.max_heads)
            if len(text) < 40:
                skipped["quá ngắn"] += 1
                continue
            src = doc.get("source", {})
            lang = src.get("language_category") or "?"
            langs[lang] += 1
            lens.append(len(text))
            fh.write(json.dumps({
                "doc_id": int(did), "language": lang, "domain": src.get("domain"),
                "title": clean(src.get("title") or "")[:300],
                "url": src.get("final_url") or src.get("requested_url"),
                "char_count": doc.get("content", {}).get("char_count", 0),
                "text": text,
            }, ensure_ascii=False) + "\n")
            if n % 100_000 == 0:
                print(f"  {n:,}/{len(ids):,}", flush=True)

    lens.sort()
    print(f"\n[doc_texts] ghi {len(lens):,} tài liệu")
    print(f"  độ dài: median={lens[len(lens)//2]} mean={statistics.mean(lens):.0f} "
          f"p90={lens[int(.9*len(lens))]}")
    print(f"  ngôn ngữ: {dict(langs.most_common())}")
    if skipped:
        print(f"  bỏ qua: {dict(skipped)}")


if __name__ == "__main__":
    main()
