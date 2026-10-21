"""Thí nghiệm 4: lắp bài nộp theo nhiều cách từ file top 50 mà run_full_submission
lưu kèm (<checkpoint>_top50.jsonl), không chạy GPU lại.

    # xem phân bố điểm rerank để chọn ngưỡng
    python -m scripts.assemble_submission stats --top50 tmp/sub/Bp_top50.jsonl

    # kiểm tra: lắp V1 k=20 phải ra đúng bài đã nộp
    python -m scripts.assemble_submission check --top50 tmp/sub/Bp_top50.jsonl --base tmp/sub/Bp.json

    # V1: luôn nộp k chunk đầu
    python -m scripts.assemble_submission build v1 --k 10 \
        --top50 tmp/sub/Bp_top50.jsonl --base tmp/sub/Bp.json --out tmp/sub/Bp_v1k10.json
    # V2: cắt theo điểm (giữ chunk có điểm >= max(tau, điểm top 1 - delta))
    python -m scripts.assemble_submission build v2 --tau-pct 30 --delta 0.5 \
        --top50 tmp/sub/Bp_top50.jsonl --base tmp/sub/Bp.json --out tmp/sub/Bp_v2.json
    # V3: gom theo tài liệu rồi cắt như V2
    python -m scripts.assemble_submission build v3 --per-doc 2 --tau-pct 30 \
        --top50 tmp/sub/Bp_top50.jsonl --base tmp/sub/Bp.json --out tmp/sub/Bp_v3.json

`--base` là bài nộp của cùng lần chạy: lấy thứ tự id và các entry rỗng (--pad-empty)
từ đó, nên bài lắp ra luôn đủ mọi id và nộp được. Chunk được giữ chỗ cho sub-query
(reserved_slot) mặc định luôn được giữ ở V2/V3; tắt bằng --no-keep-reserved.
"""
import argparse
import json
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from src.rag.pipeline.submission import submission_chunk_text, submission_doc_id, zip_submission


def load_top50(path: str) -> Dict[int, Dict[str, Any]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[int(r["id"])] = r  # bản ghi sau (chạy lại) thắng bản trước
    return out


def score(c: Dict[str, Any]) -> float:
    s = c.get("rerank_score")
    return float(s) if s is not None else float("-inf")


def to_entry(qid: int, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cùng định dạng với bài nộp gốc (submission.result_to_submission_entry)."""
    docs = list(OrderedDict.fromkeys(submission_doc_id(c["doc_id"]) for c in chunks))
    return {"id": qid, "relevant_docs": docs,
            "relevant_chunks": [{"doc_id": submission_doc_id(c["doc_id"]),
                                 "chunk_text": submission_chunk_text(c["content"])} for c in chunks]}


def submitted_chunks(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_id = {c["chunk_id"]: c for c in rec["candidates"]}
    return [by_id[cid] for cid in rec["submitted"] if cid in by_id]


def percentile(vals: List[float], p: float) -> float:
    vals = sorted(v for v in vals if v != float("-inf"))
    if not vals:
        return float("-inf")
    i = min(len(vals) - 1, max(0, round(p / 100 * (len(vals) - 1))))
    return vals[i]


def cut(ordered: List[Dict[str, Any]], tau: Optional[float], delta: Optional[float],
        min_k: int, max_k: int, keep_reserved: bool) -> List[Dict[str, Any]]:
    """Giữ chunk có điểm >= max(tau, top1 - delta); luôn giữ min_k chunk đầu và
    (nếu bật) chunk giữ chỗ; tối đa max_k. Giữ nguyên thứ tự của `ordered`."""
    if not ordered:
        return []
    top1 = max(score(c) for c in ordered)
    thr = float("-inf")
    if tau is not None:
        thr = max(thr, tau)
    if delta is not None:
        thr = max(thr, top1 - delta)
    out = []
    for i, c in enumerate(ordered):
        if i < min_k or score(c) >= thr or (keep_reserved and c.get("reserved_slot")):
            out.append(c)
    return out[:max_k]


def v1(rec, a) -> List[Dict[str, Any]]:
    return submitted_chunks(rec)[: a.k]


def v2(rec, a) -> List[Dict[str, Any]]:
    return cut(submitted_chunks(rec), a.tau, a.delta, a.min_k, a.max_k, a.keep_reserved)


def v3(rec, a) -> List[Dict[str, Any]]:
    """Gom 50 ứng viên theo tài liệu: điểm tài liệu = điểm chunk cao nhất +
    second_weight × điểm chunk cao thứ hai. Lấy tối đa per_doc chunk mỗi tài liệu
    theo thứ tự tài liệu, rồi cắt như V2."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in sorted(rec["candidates"], key=score, reverse=True):
        groups.setdefault(str(c["doc_id"]), []).append(c)

    def doc_score(cs):
        s = score(cs[0])
        return s + (a.second_weight * score(cs[1]) if len(cs) > 1 and score(cs[1]) != float("-inf") else 0.0)

    order = sorted(groups, key=lambda d: doc_score(groups[d]), reverse=True)
    picked: List[Dict[str, Any]] = []
    for d in order:
        picked += groups[d][: a.per_doc]
    if a.keep_reserved:
        have = {c["chunk_id"] for c in picked}
        rank = {d: i for i, d in enumerate(order)}
        for c in rec["candidates"]:
            if c.get("reserved_slot") and c["chunk_id"] not in have:
                picked.append(c)
        picked.sort(key=lambda c: (rank[str(c["doc_id"])], -score(c)))
    return cut(picked, a.tau, a.delta, a.min_k, a.max_k, a.keep_reserved)


VARIANTS = {"v1": v1, "v2": v2, "v3": v3}


def cmd_build(a) -> int:
    top = load_top50(a.top50)
    base = json.load(open(a.base, encoding="utf-8"))
    if a.tau_pct is not None:
        a.tau = percentile([score(c) for r in top.values() for c in submitted_chunks(r)], a.tau_pct)
        print(f"tau = phân vị {a.tau_pct} của điểm 20 chunk nộp = {a.tau:.4f}")
    fn = VARIANTS[a.variant]
    out, sizes = [], []
    for e in base:
        rec = top.get(int(e["id"]))
        if rec is None:
            out.append(e)  # query rỗng (--pad-empty) hoặc chạy lỗi: giữ nguyên
            continue
        chunks = fn(rec, a)
        sizes.append(len(chunks))
        out.append(to_entry(int(e["id"]), chunks))
    assert [x["id"] for x in out] == [x["id"] for x in base], "thứ tự id bị đổi"
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"file nộp: {zip_submission(a.out)}")
    if sizes:
        s = sorted(sizes)
        print(f"{a.out}: {len(out)} entry, {len(sizes)} query có top 50 | số chunk nộp: "
              f"min {s[0]}, trung vị {s[len(s) // 2]}, max {s[-1]}, trung bình {sum(s) / len(s):.1f}")
    return 0


def cmd_check(a) -> int:
    top = load_top50(a.top50)
    base = {int(e["id"]): e for e in json.load(open(a.base, encoding="utf-8"))}
    bad = []
    for qid, rec in top.items():
        got = to_entry(qid, submitted_chunks(rec))
        if got != base.get(qid):
            bad.append(qid)
    print(f"{len(top) - len(bad)}/{len(top)} query lắp lại khớp bài đã nộp" + (f"; lệch: {bad[:10]}" if bad else ""))
    return 1 if bad else 0


def cmd_stats(a) -> int:
    top = load_top50(a.top50)
    by_rank: Dict[int, List[float]] = {}
    gaps = []
    for r in top.values():
        sc = [score(c) for c in submitted_chunks(r)]
        for i, s in enumerate(sc, 1):
            by_rank.setdefault(i, []).append(s)
        if len(sc) >= 2:
            gaps.append(sc[0] - sc[1])
    print(f"{len(top)} query. Điểm rerank theo hạng trong bài nộp (phân vị 10 / 50 / 90):")
    for i in (1, 2, 3, 5, 10, 15, 20):
        if i in by_rank:
            v = by_rank[i]
            print(f"  hạng {i:2d}: {percentile(v, 10):.4f} / {percentile(v, 50):.4f} / {percentile(v, 90):.4f}")
    if gaps:
        print(f"  khoảng cách hạng 1 - hạng 2: trung vị {percentile(gaps, 50):.4f}, phân vị 90 {percentile(gaps, 90):.4f}")
    n_res = sum(1 for r in top.values() for c in r["candidates"] if c.get("reserved_slot"))
    print(f"  chunk được giữ chỗ cho sub-query: {n_res}")
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stats")
    s.add_argument("--top50", required=True)
    s.set_defaults(fn=cmd_stats)

    c = sub.add_parser("check")
    c.add_argument("--top50", required=True)
    c.add_argument("--base", required=True)
    c.set_defaults(fn=cmd_check)

    b = sub.add_parser("build")
    b.add_argument("variant", choices=sorted(VARIANTS))
    b.add_argument("--top50", required=True)
    b.add_argument("--base", required=True, help="bài nộp của cùng lần chạy (thứ tự id, entry rỗng)")
    b.add_argument("--out", required=True)
    b.add_argument("--k", type=int, default=20, help="V1: số chunk nộp")
    b.add_argument("--tau", type=float, default=None, help="V2/V3: ngưỡng điểm tuyệt đối")
    b.add_argument("--tau-pct", type=float, default=None, help="V2/V3: ngưỡng = phân vị này của điểm 20 chunk nộp")
    b.add_argument("--delta", type=float, default=None, help="V2/V3: giữ chunk có điểm >= điểm top 1 - delta")
    b.add_argument("--min-k", type=int, default=3)
    b.add_argument("--max-k", type=int, default=20)
    b.add_argument("--per-doc", type=int, default=2, help="V3: tối đa số chunk mỗi tài liệu")
    b.add_argument("--second-weight", type=float, default=0.5, help="V3: trọng số chunk cao thứ hai của tài liệu")
    b.add_argument("--no-keep-reserved", dest="keep_reserved", action="store_false")
    b.set_defaults(fn=cmd_build)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
