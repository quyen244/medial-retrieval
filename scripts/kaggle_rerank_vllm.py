"""Chấm nốt pool bge 2048 trên text 8k bằng Kaggle (2 x T4, mỗi GPU một server vLLM), rồi gộp lại.

Pool `w0.6_qc_vi90_bge_q3etext2048_pool.jsonl` (bge-reranker-v2-m3, len 2048, text data/doc_texts_8k.jsonl,
câu sửa) mới chấm xong 567/1.200 câu. Lệnh build lấy các câu còn thiếu từ pool gốc `w0.6_qc_vi90_pool.jsonl`
(cùng 300 doc mỗi câu), kèm check.jsonl là các cặp đã chấm của pool 2048 để job tự kiểm cùng thang trước khi chấm.

    python -m scripts.kaggle_rerank_vllm build --owner <kaggle-user>
    # push dataset + kernel theo lệnh in ra; tải scores_*.jsonl về data/kaggle/out/<tên>
    python -m scripts.kaggle_rerank_vllm merge
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.kaggle_jobs import ensure_dataset_meta, write_job

D = "data/T28_rrf_bge_q3e"
BASE_POOL = f"{D}/w0.6_qc_vi90_pool.jsonl"
DONE_POOL = f"{D}/w0.6_qc_vi90_bge_q3etext2048_pool.jsonl"
FULL_POOL = f"{D}/w0.6_qc_vi90_bge_q3etext2048_full_pool.jsonl"
TEXTS = "data/doc_texts_8k.jsonl"
BGE = "BAAI/bge-reranker-v2-m3"
ROOT = "data/kaggle"
DATASET_DIR = f"{ROOT}/rerank_vllm_dataset"
OUT = f"{ROOT}/out"
SRC = ["kaggle/rerank_vllm_job.py", "kaggle/rerank_job.py", "kaggle/doc_label_job.py"]

KERNEL_RUN = r'''"""Kaggle kernel: chấm job __JOB__ bằng __MODEL__ host trên vLLM, mỗi GPU một server.
Sinh bởi scripts/kaggle_rerank_vllm.py; đừng sửa tay. Code nhúng ngay trong file: dataset chỉ chứa dữ liệu."""
import os, subprocess, sys

JOB = "__JOB__"
ARGS = __ARGS__
VLLM_SPEC = "__VLLM__"
SRC = __SRC__

W = "/kaggle/working"
code = os.path.join(W, "code")
os.makedirs(code, exist_ok=True)
for name, text in SRC.items():
    open(os.path.join(code, name), "w", encoding="utf-8").write(text)

def sh(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

try:
    import vllm  # noqa: F401
except Exception:
    sh([sys.executable, "-m", "pip", "install", "-q", VLLM_SPEC])
# in phiên bản thật đã cài để ghim lại (--vllm-spec "vllm==<bản>") khi bản này chạy được trên T4
subprocess.run([sys.executable, "-c", "import vllm, torch, sys; print('vllm', vllm.__version__, '| torch', "
                "torch.__version__, '| cuda', torch.version.cuda, '| python', sys.version.split()[0], flush=True)"])
sh([sys.executable, os.path.join(code, "rerank_vllm_job.py"), "--job", JOB, "--out", W, *ARGS])
'''


# vLLM 0.33 trên T4 chỉ đạt ~0,2 cặp/s cho bge 2048 (Kaggle 2026-11-03: không có FlashAttention 2 cho compute 7.5),
# tức hàng chục giờ cho 190k cặp. Bản này chạy CrossEncoder (sentence-transformers fp16, đúng cách đã chấm 567 câu
# cũ) bằng kaggle/rerank_job.py, mỗi GPU một tiến trình chấm một shard câu (id % 2).
KERNEL_CE = r'''"""Kaggle kernel: chấm job __JOB__ bằng CrossEncoder (rerank_job.py), mỗi GPU một shard câu.
Sinh bởi scripts/kaggle_rerank_vllm.py (kernels --backend ce); đừng sửa tay."""
import os, subprocess, sys, threading, time

JOB = "__JOB__"
EXTRA = __EXTRA__
RERANK_JOB_SRC = __RERANK_SRC__
W = "/kaggle/working"
script = os.path.join(W, "rerank_job.py")
open(script, "w", encoding="utf-8").write(RERANK_JOB_SRC)

import torch
n = max(torch.cuda.device_count(), 1)
print(f"GPU: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]} -> {n} shard", flush=True)

def tee(p, tag, fh):
    for line in p.stdout:
        line = f"{time.strftime('%H:%M:%S')} [gpu{tag}] {line.rstrip()}"
        fh.write(line + "\n"); fh.flush(); print(line, flush=True)

procs = []
log = open(os.path.join(W, f"log_{JOB}.txt"), "a", encoding="utf-8")
for k in range(n):
    # fp16 như các pool cũ; check.jsonl chỉ chạy ở shard 0
    cmd = [sys.executable, script, "--job", JOB, "--out", os.path.join(W, f"scores_{JOB}.s{k}.jsonl"),
           "--shard", f"{k}/{n}", "--device", f"cuda:{k}", "--dtype", "fp16"] + EXTRA
    print(" ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, env=dict(os.environ, PYTHONUNBUFFERED="1"), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    t = threading.Thread(target=tee, args=(p, k, log), daemon=True); t.start()
    procs.append((p, t))
codes = []
for p, t in procs:
    p.wait(); t.join(timeout=5); codes.append(p.returncode)
print(f"--- exit {codes}; output: {sorted(os.listdir(W))}", flush=True)
if any(codes):
    sys.exit("rerank_job lỗi, xem log ở trên")
'''


def jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def read_texts(path, need):
    """doc_id -> text của các doc cần, quét file 4 GB một lượt, chỉ parse dòng cần."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            i = line.find('"doc_id": ')
            did = line[i + 10: line.find(",", i)].strip().strip('"')
            if did in need:
                out[did] = json.loads(line)["text"]
                if len(out) == len(need):
                    break
    missing = sorted(need - out.keys())
    if missing:
        raise SystemExit(f"{len(missing)} doc không có trong {path}, vd {missing[:5]}")
    return out


def build(args):
    done = {int(r["id"]): r for r in jsonl(args.done)}
    if {(r.get("rr_model"), r.get("rr_len"), r.get("rr_text")) for r in done.values()} != {(BGE, 2048, "q3e")}:
        raise SystemExit(f"{args.done}: không phải bge 2048 trên text q3e")
    base = list(jsonl(args.pool))
    todo = [r for r in base if int(r["id"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    queries = {int(r["id"]): r["query"] for r in todo}
    pairs = {int(r["id"]): [str(c["doc_id"]) for c in r["candidates"]] for r in todo}

    # check: vài câu đã chấm, rải theo id, mỗi câu một mẫu cặp rải đều thứ hạng
    rng = random.Random(42)
    check_q = sorted(rng.sample(sorted(done), min(args.check_queries, len(done))))
    check = []
    for q in check_q:
        cs = done[q]["candidates"]
        queries[q] = done[q]["query"]
        for c in cs[:: max(1, len(cs) // args.check_per_query)][: args.check_per_query]:
            check.append({"id": q, "doc_id": str(c["doc_id"]), "expected": c["rerank_score"]})
    by_id = {int(r["id"]): r["query"] for r in base}
    bad = [q for q, t in queries.items() if by_id.get(q) != t]
    if bad:
        raise SystemExit(f"câu hỏi lệch giữa pool gốc và pool 2048: {bad[:5]}")

    need = {d for ds in pairs.values() for d in ds} | {c["doc_id"] for c in check}
    print(f"[build] {len(todo)} câu cần chấm (id {min(pairs)}-{max(pairs)}), {sum(map(len, pairs.values())):,} cặp, "
          f"{len(need):,} doc | check {len(check)} cặp từ {len(check_q)} câu đã chấm", flush=True)
    texts = read_texts(args.texts, need)
    name = args.name
    meta = {"model": BGE, "kind": "vllm-score", "max_length": 2048, "query_source": "corrected", "rr_text": "q3e",
            "source_pool": args.pool, "done_pool": args.done}
    zpath = write_job(name, meta, queries, pairs, texts, check, dataset_dir=DATASET_DIR)
    print(f"[build] {zpath} ({os.path.getsize(zpath) / 1e6:.0f} MB)")
    kernels(args, dataset_create=True)


def kernels(args, dataset_create=False):
    """Sinh lại hai kernel (code nhúng) mà không dựng lại zip dữ liệu: sửa code chỉ cần push lại kernel."""
    name = args.name
    dataset_id = ensure_dataset_meta(args.owner, args.dataset_slug, DATASET_DIR)
    owner = args.owner or dataset_id.split("/")[0]
    src = {os.path.basename(p): open(p, encoding="utf-8").read() for p in SRC}
    backend = getattr(args, "backend", "vllm")
    kernels = []
    if backend == "ce":
        # check: lệch lớn nhất 0,02249 ở smoke vLLM là nhiễu fp16 T4 (trung bình 0,00046), nên ngưỡng 0,05
        variants = (("_ce", ["--batch-size", "16", "--check-tol", "0.05", "--max-hours", "11"]),)
    else:
        variants = (("_smoke", ["--max-hours", "1"]), ("", []))
    for suffix, extra in variants:
        kdir = f"{ROOT}/kernel_{name}{suffix}"
        os.makedirs(kdir, exist_ok=True)
        if backend == "ce":
            code = (KERNEL_CE.replace("__JOB__", name).replace("__EXTRA__", json.dumps(extra + args.job_arg))
                    .replace("__RERANK_SRC__", repr(src["rerank_job.py"])))
        else:
            code = (KERNEL_RUN.replace("__JOB__", name).replace("__MODEL__", BGE).replace("__VLLM__", args.vllm_spec)
                    .replace("__ARGS__", json.dumps(extra + args.job_arg)).replace("__SRC__", repr(src)))
        open(f"{kdir}/run.py", "w", encoding="utf-8").write(code)
        slug = f"vibiomir-{name.replace('_', '-')}{suffix.replace('_', '-')}"
        json.dump({"id": f"{owner}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
                   "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
                   "machine_shape": "NvidiaTeslaT4", "dataset_sources": [dataset_id],
                   "competition_sources": [], "kernel_sources": [], "model_sources": []},
                  open(f"{kdir}/kernel-metadata.json", "w", encoding="utf-8"), indent=2)
        kernels.append((os.path.normpath(kdir), f"{owner}/{slug}"))

    # kaggle CLI trên Windows hỏng với "/" trong -p, nên in theo dấu của hệ điều hành
    ds, od = os.path.normpath(DATASET_DIR), os.path.normpath(f"{OUT}/{name}")
    print(f"[kernels] vLLM: {args.vllm_spec}")
    print("Bước tiếp (máy có kaggle CLI):")
    if dataset_create:
        print(f"  kaggle datasets create -p {ds}       # lần đầu; lần sau: kaggle datasets version -p {ds} -m \"{name}\"")
    print(f"  kaggle datasets status {dataset_id}   # phải là ready trước khi push kernel, không thì kernel không gắn được dataset")
    for kdir, kid in kernels:
        print(f"  kaggle kernels push -p {kdir}")
        print(f"  kaggle kernels status {kid}")
        print(f"  kaggle kernels output {kid} -p {od} --file-pattern \"scores_.*jsonl|vllm_gpu.*log\"")
    print(f"  python -m scripts.kaggle_rerank_vllm merge --scores \"{od}\"")


def merge(args):
    """Pool 2048 đã có + điểm Kaggle -> pool đủ 1.200 câu, cùng định dạng (xếp theo điểm, giữ điểm cũ ở _prev)."""
    done = {int(r["id"]): r for r in jsonl(args.done)}
    scores = {}
    for path in sorted(glob.glob(os.path.join(args.scores, "**", "scores_*.jsonl"), recursive=True)):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue                    # dòng cuối ghi dở khi phiên bị ngắt
            scores[int(r["id"])] = r["scores"]
    rows, miss = [], []
    for r in jsonl(args.pool):
        q = int(r["id"])
        if q in done:
            rows.append(done[q])
            continue
        s = scores.get(q)
        if s is None or any(s.get(str(c["doc_id"])) is None for c in r["candidates"]):
            miss.append(q)
            continue
        cands = [{**c, "rerank_score_prev": c["rerank_score"], "rerank_score": s[str(c["doc_id"])]}
                 for c in r["candidates"]]
        cands.sort(key=lambda c: c["rerank_score"], reverse=True)
        rows.append({**r, "rr_model": BGE, "rr_len": 2048, "rr_text": "q3e", "candidates": cands})
    if miss and not args.allow_partial:
        raise SystemExit(f"{len(miss)} câu chưa có điểm đủ (vd {miss[:10]}); chạy tiếp kernel hoặc --allow-partial")
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[merge] {len(rows)} câu ({len(done)} có sẵn + {len(rows) - len(done)} từ Kaggle), thiếu {len(miss)} "
          f"-> {args.out}")


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pool", default=BASE_POOL)
    common.add_argument("--done", default=DONE_POOL)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", parents=[common])
    p.add_argument("--owner", default=None, help="kaggle username (lần đầu bắt buộc)")
    p.add_argument("--dataset-slug", default="vibiomir-rerank-vllm")
    p.add_argument("--name", default="t2048_rest")
    p.add_argument("--texts", default=TEXTS)
    p.add_argument("--limit", type=int, default=0, help="chỉ N câu đầu (thử)")
    p.add_argument("--check-queries", type=int, default=5)
    p.add_argument("--check-per-query", type=int, default=20)
    p.add_argument("--vllm-spec", default="vllm==0.33.0", help="gói pip cài vLLM trên Kaggle (0.33.0 chạy được trên T4, smoke 2026-11-03)")
    p.add_argument("--job-arg", action="append", default=[], help="tham số thêm cho rerank_vllm_job.py")
    p.set_defaults(fn=build)
    p = sub.add_parser("kernels", help="chỉ sinh lại kernel, dùng dataset đã upload")
    p.add_argument("--owner", default=None)
    p.add_argument("--dataset-slug", default="vibiomir-rerank-vllm")
    p.add_argument("--name", default="t2048_rest")
    p.add_argument("--vllm-spec", default="vllm==0.33.0")
    p.add_argument("--backend", choices=["vllm", "ce"], default="vllm",
                   help="ce: CrossEncoder qua kaggle/rerank_job.py, mỗi GPU một shard (vLLM trên T4 quá chậm)")
    p.add_argument("--job-arg", action="append", default=[])
    p.set_defaults(fn=kernels)
    p = sub.add_parser("merge", parents=[common])
    p.add_argument("--scores", default=f"{OUT}/t2048_rest", help="thư mục chứa scores_*.jsonl tải từ Kaggle")
    p.add_argument("--out", default=FULL_POOL)
    p.add_argument("--allow-partial", action="store_true")
    p.set_defaults(fn=merge)
    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
