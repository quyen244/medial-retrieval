"""Khảo sát payload THẬT trong Qdrant theo từng ngôn ngữ.

Câu hỏi cần trả lời trước khi quyết định đưa title/heading_path vào vector:
heading_path có mang THÊM thông tin so với title không, hay chỉ lặp lại title?

Không dùng filter IsEmpty vì trường đó chưa có payload index - Qdrant sẽ quét
toàn bộ 3,9M điểm và timeout. Lọc theo `language` thì nhanh vì đã có index.
"""
import sys
from collections import Counter

from qdrant_client import QdrantClient, models

URL, COLL, PER_LANG = "http://localhost:6333", "medical_documents", 1500


def show(pl, label):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    for k in ("doc_id", "chunk_id", "chunk_index", "language", "domain", "char_length"):
        if k in pl:
            print(f"  {k:16s} = {pl[k]!r}")
    for k in ("title", "heading_path", "headings", "headings_before", "headings_after"):
        v = pl.get(k)
        tag = "<THIẾU>" if v is None else ("<RỖNG>" if v in ("", []) else repr(v)[:300])
        print(f"  {k:16s} = {tag}")
    c = pl.get("content") or pl.get("text") or ""
    print(f"  content ({len(c)} ký tự):")
    print("      " + c[:300].replace("\n", "\n      "))


def main():
    client = QdrantClient(url=URL, timeout=180)
    info = client.get_collection(COLL)
    print(f"collection {COLL}: {info.points_count:,} điểm, status {info.status}\n")

    examples = {}
    for lang in ("vi", "en", "zh"):
        flt = models.Filter(must=[models.FieldCondition(
            key="language", match=models.MatchValue(value=lang))])
        pts, offset, got = [], None, 0
        while got < PER_LANG:
            batch, offset = client.scroll(collection_name=COLL, scroll_filter=flt,
                                          limit=512, offset=offset,
                                          with_payload=True, with_vectors=False)
            pts.extend(batch)
            got += len(batch)
            if offset is None:
                break

        st = Counter()
        for p in pts:
            pl = p.payload or {}
            title, hp = (pl.get("title") or "").strip(), (pl.get("heading_path") or "").strip()
            if not hp:
                st["heading_path rỗng"] += 1
                examples.setdefault(f"{lang}-khong", pl)
            elif hp == title:
                st["heading_path TRÙNG title"] += 1
                examples.setdefault(f"{lang}-trung", pl)
            else:
                st["heading_path có thêm thông tin"] += 1
                examples.setdefault(f"{lang}-giau", pl)
            if not title:
                st["title rỗng"] += 1
            st["_n"] += 1

        n = st.pop("_n")
        print(f"--- {lang.upper()}  (mẫu {n} chunk) ---")
        for k, v in st.most_common():
            print(f"    {k:34s} {v:5d}  ({100 * v / n:5.1f}%)")

    print()
    for key, label in [("vi-giau", "CÓ heading_path GIÀU HƠN title (vi)"),
                       ("vi-trung", "heading_path TRÙNG title (vi)"),
                       ("en-giau", "CÓ heading_path giàu hơn title (en)"),
                       ("en-trung", "heading_path TRÙNG title (en)"),
                       ("vi-khong", "KHÔNG có heading_path (vi)"),
                       ("en-khong", "KHÔNG có heading_path (en)"),
                       ("zh-khong", "KHÔNG có heading_path (zh)")]:
        if key in examples:
            show(examples[key], label)


if __name__ == "__main__":
    sys.exit(main())
