"""Lắp offline các biến thể chia top-k sub-query / giữ chỗ / số chunk nộp từ file
<checkpoint>_all.jsonl mà `run_full_submission --dump-all` ghi ra. Không cần GPU.

    # kiểm tra: lắp lại đúng cấu hình của lần chạy phải ra y hệt bài đã nộp
    python -m scripts.simulate_subquota check --all tmp/sub1200/D_all_all.jsonl \
        --base tmp/sub1200/D_all.json --final-k 40

    # lưới: 4 kiểu chia x 3 mức K -> 12 bài nộp (kèm .zip)
    python -m scripts.simulate_subquota grid --all tmp/sub1200/D_all_all.jsonl \
        --base tmp/sub1200/D_all.json --sub-quota global fixed sqrt corpus --final-k 30 40 50 \
        --out-prefix tmp/sub1200/D_tune

Mô phỏng đúng MultiQueryPipeline._run_branch_rerank: mỗi sub-query xếp ứng viên
của nó theo điểm sub-query, giữ top-k (chung hoặc chia theo ngôn ngữ), top
`slots` làm chỗ giữ; gom với ứng viên câu gốc; giữ branch_keep mỗi ngôn ngữ theo
điểm câu gốc; chọn final_k chunk có giữ chỗ. Câu ngắn không có sub-query nên
chỉ đổi theo final_k. Thay đổi phần TRUY XUẤT (số ứng viên lấy mỗi nhánh) thì
không mô phỏng được - phải chạy GPU lại.
"""
import argparse
import json
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from scripts.assemble_submission import to_entry
from src.rag.pipeline.components.routing import quota_from_counts
from src.rag.pipeline.multiquery import select_with_slots
from src.rag.pipeline.submission import zip_submission

# Số chunk mỗi ngôn ngữ trong collection medical_documents (đếm 2026-10-22).
CORPUS_COUNTS = {"vi": 3094307, "en": 592126, "zh": 241367}
BRANCH_KEEP = {"vi": 26, "en": 16, "zh": 8}


def sub_keep_for(quota: str, total: int) -> Optional[Dict[str, int]]:
    """Giống --sub-quota của run_full_submission."""
    if quota == "global":
        return None
    if quota == "fixed":
        return quota_from_counts(BRANCH_KEEP, total, power=1.0)
    return quota_from_counts(CORPUS_COUNTS, total, power=1.0 if quota == "corpus" else 0.5)


def simulate(rec: Dict[str, Any], sub_keep: Optional[Dict[str, int]], sub_top_k: int,
             branch_keep: Dict[str, int], slots: int, max_rank: int, final_k: int) -> List[Dict[str, Any]]:
    chunks = rec["chunks"]
    by_branch: Dict[str, List[str]] = {}
    subs: List[List[str]] = []
    for u in rec["units"]:
        if u["kind"] != "sub":
            for lang, ids in u["branches"].items():
                by_branch.setdefault(lang, []).extend(ids)
            continue
        subs.append(list(dict.fromkeys(cid for ids in u["branches"].values() for cid in ids)))

    slot_ids: List[List[str]] = []
    for n, ids in enumerate(subs, 1):
        key = f"sub{n}"
        ranked = sorted(ids, key=lambda c: chunks[c]["sub"][key], reverse=True)
        slot_ids.append(ranked[:slots])
        if sub_keep:
            left, scored = dict(sub_keep), []
            for c in ranked:
                if left.get(chunks[c]["branch"], 0) > 0:
                    left[chunks[c]["branch"]] -= 1
                    scored.append(c)
        else:
            scored = ranked[:sub_top_k]
        for c in scored:
            by_branch.setdefault(chunks[c]["branch"], []).append(c)

    pool = list(dict.fromkeys(cid for ids in by_branch.values() for cid in ids))
    cand = {cid: SimpleNamespace(chunk_id=cid, rerank_score=chunks[cid]["raw"], reserved_slot=False)
            for cid in pool}
    scored_all = sorted((cand[c] for c in pool), key=lambda c: c.rerank_score, reverse=True)
    kept = []
    for lang, keep in branch_keep.items():
        kept += [c for c in scored_all if chunks[c.chunk_id]["branch"] == lang][:keep]
    kept.sort(key=lambda c: c.rerank_score, reverse=True)
    if subs and slot_ids:
        final, _ = select_with_slots(kept, slot_ids, final_k, slots, max_rank)
    else:
        final = kept[:final_k]
    return [{"doc_id": chunks[c.chunk_id]["doc_id"], "content": chunks[c.chunk_id]["content"]} for c in final]


def load_all(path: str) -> Dict[int, Dict[str, Any]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[int(r["id"])] = r
    return out


def build(dump, base, a, quota: str, final_k: int) -> List[Dict[str, Any]]:
    keep = sub_keep_for(quota, a.sub_top_k)
    out = []
    for e in base:
        rec = dump.get(int(e["id"]))
        if rec is None:
            out.append(e)  # entry rỗng (--pad-empty) hoặc không có trong dump: giữ nguyên
            continue
        out.append(to_entry(int(e["id"]), simulate(rec, keep, a.sub_top_k, BRANCH_KEEP,
                                                    a.slots, a.slot_max_rank, final_k)))
    return out


def cmd_check(a) -> int:
    dump, base = load_all(a.all), json.load(open(a.base, encoding="utf-8"))
    got = build(dump, base, a, a.sub_quota[0], a.final_k[0])
    bad = [e["id"] for e, g in zip(base, got) if e != g]
    print(f"{len(base) - len(bad)}/{len(base)} entry khớp bài đã nộp "
          f"({len(dump)} query có dump)" + (f"; lệch: {bad[:10]}" if bad else ""))
    return 1 if bad else 0


def cmd_grid(a) -> int:
    dump, base = load_all(a.all), json.load(open(a.base, encoding="utf-8"))
    for quota in a.sub_quota:
        keep = sub_keep_for(quota, a.sub_top_k)
        for k in a.final_k:
            out = build(dump, base, a, quota, k)
            path = f"{a.out_prefix}_sub{quota}_k{k}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"{path}: sub-query giữ {keep or f'top {a.sub_top_k} chung'}, K={k} -> {zip_submission(path)}")
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("check", cmd_check), ("grid", cmd_grid)):
        s = sub.add_parser(name)
        s.add_argument("--all", required=True, help="<checkpoint>_all.jsonl từ --dump-all")
        s.add_argument("--base", required=True, help="bài nộp của cùng lần chạy (thứ tự id, entry rỗng)")
        s.add_argument("--sub-quota", nargs="+", default=["global"],
                       choices=["global", "fixed", "sqrt", "corpus"])
        s.add_argument("--sub-top-k", type=int, default=20)
        s.add_argument("--final-k", nargs="+", type=int, default=[40], help="tối đa 50")
        s.add_argument("--slots", type=int, default=3)
        s.add_argument("--slot-max-rank", type=int, default=30)
        if name == "grid":
            s.add_argument("--out-prefix", required=True)
        s.set_defaults(fn=fn)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
