"""Kaggle: chấm một job rerank (định dạng của kaggle/rerank_job.py) bằng cross-encoder host trên vLLM,
mỗi GPU một server `--task score`, client chia câu theo id % số server.

Job zip (scripts/kaggle_rerank_vllm.py dựng): meta.json, queries.jsonl, docs.jsonl, pairs.jsonl, check.jsonl.
Ra: <out>/scores_<job>.jsonl, mỗi dòng {"id", "scores": {doc_id: điểm}}, ghi tới đâu flush tới đó, chạy lại thì
bỏ qua câu đã chấm.

Cắt độ dài ở client, không để server cắt: CrossEncoder (sentence-transformers, cách chấm các pool cũ) cắt
`longest_first`, tức cắt đuôi doc cho tới khi <s> q </s></s> d </s> vừa max_length; còn `truncate_prompt_tokens`
của vLLM giữ các token CUỐI, sẽ cắt mất câu hỏi. Trước khi chấm thật, check.jsonl (các cặp đã chấm ở pool cũ)
phải khớp trong --check-tol, không thì dừng: điểm khác thang thì không gộp với 567 câu đã có được.

    python rerank_vllm_job.py --job t2048_rest --out /kaggle/working
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doc_label_job import start_servers, wait_ready     # noqa: E402
from rerank_job import Job, find_job, load_work, read_done     # noqa: E402

SPECIAL = 4     # XLM-R cặp câu: <s> q </s></s> d </s>


def say(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class Trimmer:
    """Cắt đuôi doc để q + d + SPECIAL <= max_length, như truncation longest_first khi câu hỏi ngắn hơn doc."""

    def __init__(self, model, max_length):
        import threading
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.max_length = max_length
        self.n_cut = 0
        self.lock = threading.Lock()    # tokenizer nhanh của HF dùng chung nhiều luồng có thể báo "Already borrowed"

    def __call__(self, query, doc):
        with self.lock:
            q = len(self.tok(query, add_special_tokens=False)["input_ids"])
            ids = self.tok(doc, add_special_tokens=False)["input_ids"]
            budget = self.max_length - SPECIAL - q
            if len(ids) <= budget:
                return doc
            self.n_cut += 1
            return self.tok.decode(ids[:max(budget, 1)])


def make_scorer(model, urls, per_request):
    import requests

    def score(key, query, docs):
        """Điểm (sigmoid, như CrossEncoder.predict với model 1 nhãn) cho từng doc, cùng thứ tự."""
        url = urls[key % len(urls)]
        out = []
        for b in range(0, len(docs), per_request):
            part = docs[b: b + per_request]
            for attempt in range(4):
                try:
                    r = requests.post(f"{url}/score", json={"model": model, "text_1": query, "text_2": part},
                                      timeout=600)
                    r.raise_for_status()
                    data = sorted(r.json()["data"], key=lambda x: x["index"])
                    if len(data) != len(part):
                        raise ValueError(f"server trả {len(data)} điểm cho {len(part)} doc")
                    out += [float(x["score"]) for x in data]
                    break
                except Exception as e:
                    if attempt == 3:
                        raise
                    say(f"[score] câu {key} lỗi ({e}), thử lại")
                    time.sleep(2 ** attempt)
        return out
    return score


def check_scale(job, score, trim, tol, mean_tol):
    """Chấm lại các cặp đã có điểm ở pool cũ. Sai text / max_length / cách cắt làm lệch CẢ LOẠT (dùng nhầm
    doc_texts.jsonl từng lệch tới 0,24), nên cổng chính là lệch trung bình; lệch lớn nhất chỉ chặn ngoại lệ thô.
    Smoke 2026-11-03 (vLLM 0.33.0, T4 fp16): trung bình 0,00046, lớn nhất 0,02249 - nhiễu fp16 lẻ, cùng thang."""
    if not job.has("check.jsonl"):
        say("[check] job không có check.jsonl: bỏ qua")
        return
    rows = list(job.jsonl("check.jsonl"))
    queries = {int(r["id"]): r["query"] for r in job.jsonl("queries.jsonl")}
    need = {str(r["doc_id"]) for r in rows}
    docs = {str(r["doc_id"]): r["text"] for r in job.jsonl("docs.jsonl") if str(r["doc_id"]) in need}
    diffs = []
    for qid in sorted({int(r["id"]) for r in rows}):
        mine = [r for r in rows if int(r["id"]) == qid]
        got = score(qid, queries[qid], [trim(queries[qid], docs[str(r["doc_id"])]) for r in mine])
        diffs += [(abs(g - r["expected"]), qid, str(r["doc_id"]), r["expected"], g) for g, r in zip(got, mine)]
    worst, mean = max(d[0] for d in diffs), sum(d[0] for d in diffs) / len(diffs)
    say(f"[check] {len(diffs)} cặp đã chấm ở pool cũ: lệch trung bình {mean:.5f} (ngưỡng {mean_tol}), "
        f"lớn nhất {worst:.5f} (ngưỡng {tol})")
    for d, qid, did, exp, got in sorted(diffs, reverse=True)[:5]:
        say(f"[check]   câu {qid} doc {did} ({len(docs[did])} ký tự): cũ {exp:.5f} mới {got:.5f} lệch {d:.5f}")
    if mean > mean_tol or worst > tol:
        raise SystemExit("[check] điểm vLLM KHÁC thang pool cũ: không gộp được. Xem lại text, max_length, dtype.")


def run(args):
    t_end = time.time() + 3600 * args.max_hours
    job = Job(find_job(args.job))
    meta = job.meta
    model, max_length = meta["model"], int(meta["max_length"])
    out = os.path.join(args.out, f"scores_{meta['name']}.jsonl")
    os.makedirs(args.out, exist_ok=True)
    queries, docs, work = load_work(job, (0, 1), read_done(out))
    n_pairs = sum(len(ds) for _, ds in work)
    say(f"[job] {meta['name']} | {model} len {max_length} | còn {len(work)} câu, {n_pairs:,} cặp -> {out}")
    if not work:
        return 0
    trim = Trimmer(model, max_length)
    procs, urls = [], args.urls.split(",") if args.urls else []
    try:
        if not urls:
            import torch
            n_gpu = args.gpus or torch.cuda.device_count()
            # server dài hơn max_length một chút: decode rồi tokenize lại doc đã cắt có thể lệch vài token
            procs = start_servers(model, n_gpu, args.out, max_length + 64, args.gpu_mem, args.max_num_seqs,
                                  # --runner pooling: các bản vLLM mới bỏ `--task score`; cross-encoder 1 nhãn
                                  # tự thành model score và có endpoint /score
                                  ["--runner", "pooling"] + args.vllm_arg, prefix_cache=False)
            wait_ready(procs, args.out)
            urls = [f"http://127.0.0.1:{8000 + i}" for i in range(n_gpu)]
        score = make_scorer(model, urls, args.per_request)
        check_scale(job, score, trim, args.check_tol, args.check_mean_tol)

        def one(item):
            qid, ds = item
            if time.time() > t_end - 600:
                return None
            q = queries[qid]
            return qid, ds, score(qid, q, [trim(q, docs[d]) for d in ds])

        t0, scored = time.time(), 0
        with open(out, "a", encoding="utf-8") as fh, \
                ThreadPoolExecutor(max_workers=args.concurrency * len(urls)) as ex:
            for i, res in enumerate(ex.map(one, work), 1):
                if res is None:
                    continue
                qid, ds, s = res
                bad = [d for d, v in zip(ds, s) if not math.isfinite(v)]
                if bad:
                    raise SystemExit(f"câu {qid}: {len(bad)} điểm không hữu hạn")
                fh.write(json.dumps({"id": qid, "scores": {d: round(v, 6) for d, v in zip(ds, s)}}) + "\n")
                fh.flush()
                scored += len(ds)
                if i == 1 or i % 20 == 0 or i == len(work):
                    el = time.time() - t0
                    say(f"[score] {i}/{len(work)} câu | {scored / el:.1f} cặp/s | còn ~"
                        f"{(n_pairs - scored) / max(scored / el, 1e-9) / 60:.0f} phút | {trim.n_cut} doc bị cắt")
    finally:
        for p in procs:
            p.terminate()
    say(f"[job] xong: {len(read_done(out))} câu trong {out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--urls", default="", help="server có sẵn, phẩy ngăn cách (không tự dựng server)")
    ap.add_argument("--gpus", type=int, default=0, help="0 = mọi GPU")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--vllm-arg", action="append", default=[], help="tham số thêm cho vllm serve")
    ap.add_argument("--per-request", type=int, default=50, help="số doc mỗi request /score")
    ap.add_argument("--concurrency", type=int, default=4, help="số câu chấm song song MỖI server")
    ap.add_argument("--check-tol", type=float, default=0.05, help="lệch lớn nhất cho phép")
    ap.add_argument("--check-mean-tol", type=float, default=0.005, help="lệch trung bình cho phép")
    ap.add_argument("--max-hours", type=float, default=11.5)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
