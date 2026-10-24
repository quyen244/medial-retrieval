"""In lại một record của data/augment_v2/augmented.jsonl cho dễ đọc.

    python -m scripts.show_augmented --id 2
    python -m scripts.show_augmented --id 2 --file data/augment_v2/augmented.jsonl

Chỉ đọc và định dạng lại file augment đã sinh sẵn - không gọi LLM, không cần
vLLM đang chạy. Dùng khi đã có augmented.jsonl và muốn soát nhanh một qid.
"""
import argparse
import json
import sys
from typing import Any, Dict, Optional

RULE = "─" * 90
DOUBLE_RULE = "═" * 90


def find_record(path: str, qid: int) -> Optional[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if int(d.get("qid")) == qid:
                return d
    return None


def fmt_terms(terms, label: str) -> None:
    if not terms:
        print(f"  {label:<10}: (rỗng)")
        return
    print(f"  {label:<10} ({len(terms)}):")
    for t in terms:
        print(f"    - vi={t.get('vi', '')!r}  en={t.get('en', '')!r} ({t.get('source_en', '?')})  zh={t.get('zh', '')!r}")


def fmt_targets(targets) -> None:
    if not targets:
        print("    (không có retrieval target)")
        return
    for t in targets:
        print(f"    - [{t.get('aspect', '')}] vi={t.get('vi', '')!r} | en={t.get('en', '')!r} | zh={t.get('zh', '')!r}")


def source_breakdown(terms) -> str:
    counts: Dict[str, int] = {}
    for t in terms:
        s = t.get("source_en", "?")
        counts[s] = counts.get(s, 0) + 1
    return ", ".join(f"{k}={v}" for k, v in counts.items()) if counts else "(rỗng)"


def show(d: Dict[str, Any], threshold: int = 30) -> None:
    qid = d.get("qid")
    n_words = d.get("n_words", 0)
    is_long = d.get("is_long", False)
    nhanh = "DÀI" if is_long else "NGẮN"

    print(DOUBLE_RULE)
    print(f"qid={qid}  nhánh {nhanh}  ({n_words} từ, ngưỡng dài = {threshold})")
    print(DOUBLE_RULE)
    print(f"GỐC: {d.get('original', '')}")

    print("\nTHỰC THỂ ĐÃ TRÍCH XUẤT")
    fmt_terms(d.get("diseases", []), "disease")
    fmt_terms(d.get("medications", []), "medication")
    fmt_terms(d.get("symptoms", []), "symptom")

    units = d.get("units", [])
    print(f"\nUNITS ({len(units)})")
    for i, u in enumerate(units, 1):
        text = u.get("augmented") or u.get("text", "")
        n_lines = text.count("\n") + 1 if text else 0
        print(f"\n  unit {i}/{len(units)}  kind={u.get('kind')}  ({len(text)} ký tự, {n_lines} dòng)")
        targets = u.get("targets", [])
        print(f"  retrieval targets ({len(targets)}):")
        fmt_targets(targets)
        print("  văn bản gửi retrieval:")
        print("    ┌" + RULE)
        for line in text.split("\n"):
            print(f"    │ {line}")
        print("    └" + RULE)

    failures = d.get("failures", [])
    stats = d.get("stats", {})
    diseases = d.get("diseases", [])
    medications = d.get("medications", [])

    print("\nINSIGHTS")
    print(f"  - failures: {failures if failures else 'không có'}")
    print(f"  - nguồn disease   : {source_breakdown(diseases)}"
          + (f"  (ICD phủ {stats.get('n_disease_from_icd', 0)}/{len(diseases)})" if diseases else ""))
    print(f"  - nguồn medication: {source_breakdown(medications)}"
          + (f"  (RxNorm phủ {stats.get('n_medication_from_rxnorm', 0)}/{len(medications)})" if medications else ""))
    print(f"  - nguồn symptom   : {source_breakdown(d.get('symptoms', []))}")
    print(f"  - số unit: {stats.get('n_unit', len(units))}, văn bản dài nhất: {stats.get('len_max', 0)} ký tự")

    print(f"\nSTATS thô: {json.dumps(stats, ensure_ascii=False)}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--id", type=int, required=True, help="qid cần xem")
    p.add_argument("--file", default="data/augment_v2/augmented.jsonl")
    p.add_argument("--threshold", type=int, default=30, help="ngưỡng số từ phân loại query dài (chỉ để hiển thị)")
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    record = find_record(args.file, args.id)
    if record is None:
        print(f"Không tìm thấy qid={args.id} trong {args.file}")
        return 1

    show(record, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
