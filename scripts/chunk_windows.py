"""Chọn VỊ TRÍ cửa sổ chunk_text trong từng doc của chunk list (T18a/T18b), thay cho cửa sổ đầu bài.

Mỗi doc được tiền xử lý TRƯỚC (strip_markdown + submission_chunk_text, đúng phép biến đổi khi nộp),
rồi mới chia thành các cửa sổ ~`--cap` ký tự SẠCH, bước `--stride` (mặc định 2100/630, overlap 70%).
Cửa sổ lai: độ dài gần cố định nhưng đầu lùi về đầu dòng, cuối lùi về hết câu / hết dòng (xem window_spans).
`start`/`end` trong file ra là vị trí trên văn bản sạch; write_submission --window best cắt đúng chỗ đó.
Chấm từng cửa sổ với câu hỏi, giữ cửa sổ điểm cao nhất (hoà thì lấy cửa sổ sớm nhất, tức đầu bài).

    # smoke: 20 câu
    python -m scripts.chunk_windows --scorer bge --n-docs 200 --limit-queries 20
    # T18a: 100 doc đầu chunk list SOTA, chấm bằng bge-reranker-v2-m3
    python -m scripts.chunk_windows --scorer bge --n-docs 100
    # T18b: chấm 200 doc để fuse_submission chọn lại 100 doc theo điểm cửa sổ
    python -m scripts.chunk_windows --scorer bge --n-docs 200

--scorer lex: tổng IDF các âm tiết / cặp âm tiết của câu hỏi có trong cửa sổ (CPU, không cần GPU).
--scorer bge: bge-reranker-v2-m3 (CrossEncoder) chấm cặp (câu hỏi, cửa sổ), cần GPU (~2-3 GB).
Câu hỏi: câu SỬA (data/spelling/corrected_queries.jsonl), theo quyết định 2026-10-27.

Ra jsonl: {"id", "scorer", "cap", "stride", "windows": {doc_id: {"start", "end", "score", "head", "n"}}} theo thứ tự chunk list.
Nộp bằng fuse_submission --window best --windows-file <file> (T18a) và thêm --chunk-order window (T18b).
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.fuse_submission import POOLS, load_pool
from scripts.md_strip import strip_markdown
from scripts.phase0_assemble import RawDocs, load_index
from src.rag.pipeline.submission import submission_chunk_text

TOK = re.compile(r"\w+", re.UNICODE)
OUT_DIR = "data/T18_chunk_window"


def prep(raw_slice):
    """Đúng phép biến đổi write_submission áp lên cửa sổ trước khi nộp: chấm chính văn bản sẽ nộp."""
    return submission_chunk_text(strip_markdown(raw_slice))


def terms(text):
    t = [w.lower() for w in TOK.findall(text)]
    return set(t) | {a + " " + b for a, b in zip(t, t[1:])}


SENT_END = re.compile(r"[.!?。！？;:](?=\s)|\n")


def window_spans(text, cap, stride, slack=200):
    """Cửa sổ lai (T18): độ dài gần cố định, ranh giới đẹp.
    Đầu cửa sổ: 0, stride, 2*stride... lùi về đầu dòng trong `slack` ký tự (không có thì khoảng trắng trong 50).
    Cuối cửa sổ: start + cap lùi về hết câu / hết dòng gần nhất trong `slack` ký tự (không có thì khoảng trắng).
    Cửa sổ cuối luôn chạm cuối bài, đầu của nó TIẾN tới đầu dòng kế tiếp để không dài quá cap.
    Bài ngắn hơn cap chỉ có một cửa sổ là cả bài. Trả về [(start, end)] trên văn bản đã tiền xử lý."""
    n = len(text)
    if n <= cap:
        return [(0, n)]

    def snap_start(a):
        nl = text.rfind("\n", a - slack, a)
        if nl >= 0:
            return nl + 1
        sp = text.rfind(" ", a - 50, a)
        return sp + 1 if sp >= 0 else a

    def snap_start_fwd(a):
        nl = text.find("\n", a, a + slack)
        if nl >= 0:
            return nl + 1
        sp = text.find(" ", a, a + 50)
        return sp + 1 if sp >= 0 else a

    def snap_end(e):
        if e >= n:
            return n
        cut = [m.end() for m in SENT_END.finditer(text, e - slack, e)]
        if cut:
            return cut[-1]
        sp = text.rfind(" ", e - 50, e)
        return sp if sp > 0 else e

    spans, a = [], 0
    while a + cap < n:
        s0 = 0 if a == 0 else snap_start(a)
        spans.append((s0, snap_end(s0 + cap)))
        a += stride
    spans.append((snap_start_fwd(n - cap), n))
    return sorted(set(spans))


def chunk_lists(labels, n_docs):
    ranks = {}
    for label in labels:
        for q, cs in load_pool(POOLS.get(label, label)).items():
            r = ranks.setdefault(q, [])
            seen = set(r)
            r += [c["doc_id"] for c in cs if c["doc_id"] not in seen]
    return {q: r[:n_docs] for q, r in ranks.items()}


class Progress:
    def __init__(self, total, unit):
        self.total, self.unit, self.t0, self.last = total, unit, time.time(), time.time()

    def tick(self, done, force=False):
        now = time.time()
        if force or now - self.last >= 60:
            self.last = now
            rate = done / max(now - self.t0, 1e-9)
            print(f"[windows] {done:,}/{self.total:,} {self.unit}, {rate:,.1f} {self.unit}/s, "
                  f"còn ~{(self.total - done) / max(rate, 1e-9) / 60:.0f} phút", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-pool", nargs="+", default=["top60", "top300"], help="tên trong POOLS hoặc đường dẫn")
    ap.add_argument("--n-docs", type=int, default=100, help="số doc đầu chunk list được chấm cửa sổ")
    ap.add_argument("--cap", type=int, default=2100,
                    help="độ dài cửa sổ, ký tự SẠCH (2100 ~ độ dài thực nộp của SOTA cắt 2700 thô)")
    ap.add_argument("--stride", type=int, default=630, help="bước dịch; overlap = cap - stride (630 -> 70%%)")
    ap.add_argument("--scorer", choices=["lex", "bge"], default="bge")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--queries", default="data/spelling/corrected_queries.jsonl")
    ap.add_argument("--limit-queries", type=int, default=0, help="smoke: chỉ N câu đầu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    query = {int(r["id"]): r.get("corrected_query") or r["query"] for r in map(json.loads, open(args.queries, encoding="utf-8"))}
    lists = chunk_lists(args.chunk_pool, args.n_docs)
    qids = sorted(lists)[: args.limit_queries or None]
    raw_docs = RawDocs(load_index())
    jobs = []                                   # (qid, doc_id, [(start, end)...])
    t = time.time()
    for qid in qids:
        for d in lists[qid]:
            clean = prep(raw_docs.markdown(d) or "")
            jobs.append((qid, d, window_spans(clean, args.cap, args.stride) if clean else [(0, 0)]))
    n_win = sum(len(s) for _, _, s in jobs)
    print(f"[windows] {len(qids)} câu, {len(jobs):,} doc, {n_win:,} cửa sổ ({n_win / len(jobs):.1f}/doc), "
          f"đọc bài {time.time() - t:.0f}s", flush=True)

    scores = {}
    if args.scorer == "lex":
        df = Counter()                         # IDF tính trên chính tập doc của chunk list
        qterms = {q: terms(query[q]) for q in qids}
        vocab = set().union(*qterms.values())
        uniq = {d for _, d, _ in jobs}
        for d in uniq:
            df.update(terms(prep(raw_docs.markdown(d))) & vocab)
        idf = {w: math.log(1 + len(uniq) / (1 + df[w])) for w in vocab}
        prog = Progress(len(jobs), "doc")
        for i, (qid, d, starts) in enumerate(jobs, 1):
            clean = prep(raw_docs.markdown(d) or "")
            scores[(qid, d)] = [sum(idf[w] for w in terms(clean[a:b]) & qterms[qid]) for a, b in starts]
            prog.tick(i)
    else:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(args.model, max_length=args.max_length, model_kwargs={"torch_dtype": "float16"})
        pairs = []
        for qid, d, starts in jobs:
            clean = prep(raw_docs.markdown(d) or "")
            pairs += [(query[qid], clean[a:b]) for a, b in starts]
        flat, prog = [], Progress(len(pairs), "cặp")
        step = args.batch * 64
        for i in range(0, len(pairs), step):
            flat += model.predict(pairs[i: i + step], batch_size=args.batch, show_progress_bar=False).tolist()
            prog.tick(len(flat))
        prog.tick(len(flat), force=True)
        k = 0
        for qid, d, starts in jobs:
            scores[(qid, d)] = flat[k: k + len(starts)]
            k += len(starts)

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{args.scorer}_n{args.n_docs}_cap{args.cap}_st{args.stride}" + \
          (f"_q{len(qids)}" if args.limit_queries else "")
    out = args.out or f"{OUT_DIR}/{tag}_windows.jsonl"
    moved, starts_moved, lens = 0, [], [b - a for _, _, sp in jobs for a, b in sp]
    by_q = {}
    for qid, d, starts in jobs:
        sc = scores[(qid, d)]
        b = max(range(len(sc)), key=lambda i: (sc[i], -i))
        by_q.setdefault(qid, {})[d] = {"start": starts[b][0], "end": starts[b][1], "score": round(float(sc[b]), 5),
                                       "head": round(float(sc[0]), 5), "n": len(starts)}
        if starts[b][0] > 0:
            moved += 1
            starts_moved.append(starts[b][0])
    with open(out, "w", encoding="utf-8") as fh:
        for qid in qids:
            fh.write(json.dumps({"id": qid, "scorer": args.scorer, "cap": args.cap, "stride": args.stride, "windows": by_q[qid]},
                                ensure_ascii=False) + "\n")
    multi = sum(1 for _, _, s in jobs if len(s) > 1)
    lens.sort()
    print(f"[windows] độ dài cửa sổ (ký tự sạch): mean {statistics.mean(lens):,.0f}, p10 {lens[len(lens) // 10]:,}, "
          f"median {lens[len(lens) // 2]:,}, max {lens[-1]:,}", flush=True)
    print(f"[windows] doc có >1 cửa sổ: {multi:,}/{len(jobs):,} ({multi / len(jobs):.0%}) | cửa sổ tốt nhất KHÔNG ở đầu bài: "
          f"{moved:,} ({moved / len(jobs):.0%}), vị trí median {statistics.median(starts_moved) if starts_moved else 0:,.0f} ký tự "
          f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
