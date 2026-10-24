"""In đầu vào truy xuất của một câu theo dạng cây: câu gốc, câu đã sửa, bản dịch,
khía cạnh (retrieval target), và dòng nào thành một list tìm kiếm.

    python -m scripts.show_query_inputs 358
    python -m scripts.show_query_inputs 358 502 --variants orig,corrected,en,zh,aspects

Đọc đúng ba file mà scripts/phase1_doc_retrieve.py đọc (config.yaml,
retrieval.doc.*), và đánh số list bằng chính variant_texts() của script đó.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.phase1_doc_retrieve import QUERIES, load_variant_sources, variant_texts
from src.rag.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+", type=int)
    ap.add_argument("--variants", default="orig,corrected,en,zh,aspects")
    args = ap.parse_args()

    cfg = load_config()
    variants = args.variants.split(",")
    src = load_variant_sources(cfg, ["orig", "corrected", "en", "zh", "aspects"])
    queries = {int(q["id"]): q for q in map(json.loads, open(QUERIES, encoding="utf-8"))}

    for qid in args.ids:
        q = queries[qid]
        lists = variant_texts(q, variants, src)
        num = {text: i for i, (_, text) in enumerate(lists, 1)}
        mark = lambda t: f"  ← list {num[t.strip()]}" if t and t.strip() in num else "  (không tìm)"
        tr = src["tr"].get(qid, {})
        aug = src["aug"].get(qid, {})

        print(f"\n{'═' * 90}\nid={qid}  |  variants={variants}  |  {len(lists)} list\n{'═' * 90}")
        print(f"[1] CÂU GỐC (vi){mark(q['query'])}\n    {q['query']}")
        corr = src["corrected"].get(qid, "")
        print(f"[2] CÂU ĐÃ SỬA (vi){mark(corr) if corr.strip() != q['query'].strip() else '  (trùng câu gốc)'}"
              f"\n    {corr}")
        print(f"[3] BẢN DỊCH NGUYÊN CÂU — dịch từ [2], một lời gọi LLM ra cả hai")
        print(f"    en{mark(tr.get('query_en'))}: {tr.get('query_en')}")
        print(f"    zh{mark(tr.get('query_zh'))}: {tr.get('query_zh')}")
        print(f"[4] KHÍA CẠNH (retrieval_target) — sinh MỘT lần từ tiếng Việt, mỗi khía cạnh viết sẵn vi/en/zh,")
        print(f"    DÙNG CHUNG: [3] en/zh KHÔNG có khía cạnh riêng")
        for u in aug.get("units", []):
            if u["kind"] == "raw":
                print(f"    unit raw (= câu [2], không có khía cạnh)")
                continue
            print(f"    unit {u['kind']} (vi, bản thân câu này KHÔNG được tìm): {u['text']}")
            for t in u.get("targets", []):
                print(f"      • {t.get('aspect')}")
                for lang in ("vi", "en", "zh"):
                    print(f"          {lang}: {t.get(lang)}{mark(t.get(lang))}")


if __name__ == "__main__":
    main()
