"""Dựng job embed Qwen3-Embedding cho Kaggle, đọc báo cáo tốc độ, tìm từ cache vector đã tải về.

Kaggle không có Qdrant và server không có GPU, nên toàn bộ việc nặng nằm trong một kernel
(kaggle/embed_job.py được nhúng vào run.py): embed doc + câu hỏi (P1, P2), rồi tìm chính xác
bằng nhân ma trận. Mọi thứ ghi ra /kaggle/working là cache tải về được: vector doc, vector câu hỏi,
thống kê, file ứng viên top K.

    # 1. text 8k cho 2.000 doc ngẫu nhiên (CPU, cần data/raw_documents)
    python -m scripts.embed_q3e texts --limit 2000 --shuffle --out data/doc_texts_8k_sample2k.jsonl
    # 2. job + kernel đo tốc độ, chạy cả 0.6B lẫn 4B
    python -m scripts.kaggle_embed build --owner <kaggle-user> --docs data/doc_texts_8k_sample2k.jsonl --name emb_smoke2k \\
        --shard-size 250 --max-hours 2
    # 3. sau khi tải output về data/kaggle/out/emb_smoke2k
    python -m scripts.kaggle_embed report --dir data/kaggle/out/emb_smoke2k
    # (nhiều phiên Kaggle cho cả corpus) tìm lại từ cache đã gộp
    python -m scripts.kaggle_embed search --vecs data/kaggle/out/emb_full_p0/q3e06b data/kaggle/out/emb_full_p1/q3e06b \\
        --instruction p1 --retriever q3e06b_p1
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.kaggle_jobs import ensure_dataset_meta, load_queries

ROOT = "data/kaggle"
DATASET_DIR = f"{ROOT}/embed_dataset"
OUT = f"{ROOT}/out"
PHASE1 = "data/tmp/phase1"
EMBED_JOB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kaggle", "embed_job.py")
INSTRUCTIONS = {"p1": "prompts/q3e_instruct_p1.txt", "p2": "prompts/q3e_instruct_p2.txt"}
N_CORPUS = 1_003_391
# tag model -> (tên HF, ngân sách token mỗi lô trên T4 16 GB; OOM thì job tự giảm)
MODELS = {"q3e06b": ("Qwen/Qwen3-Embedding-0.6B", 32768), "q3e4b": ("Qwen/Qwen3-Embedding-4B", 12288)}


def embed_job_module():
    spec = importlib.util.spec_from_file_location("embed_job", EMBED_JOB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_instructions(files):
    from scripts.embed_q3e import load_instruction
    return {tag: load_instruction(path) for tag, path in files.items()}


def priority_ids(pool_path):
    """Doc trong pool (vd. top1000 bge-m3 của 1.200 câu) -> embed TRƯỚC. Chạy dở chừng thì phần đã xong vẫn
    đo được: tìm trên tập ưu tiên = xếp lại top1000 của bge-m3 bằng embedding mới."""
    if not pool_path:
        return set()
    with open(pool_path, encoding="utf-8") as fh:
        return {int(c["doc_id"]) for line in fh if line.strip() for c in json.loads(line)["candidates"]}


def write_job(name, docs_path, instructions, limit_queries=0, dataset_dir=None, priority=None, shard_size=4000):
    """Ghi <dataset_dir>/<name>.zip; docs.jsonl chép (doc_id, language, text), doc ưu tiên lên đầu
    (hai lượt đọc file, không giữ text trong RAM), giữ thứ tự gốc trong mỗi nhóm."""
    dataset_dir = dataset_dir or DATASET_DIR
    os.makedirs(dataset_dir, exist_ok=True)
    zpath = os.path.join(dataset_dir, f"{name}.zip")
    qs = load_queries("corrected", limit_queries)
    priority = priority or set()
    n, n_prio = 0, 0
    with zipfile.ZipFile(zpath + ".part", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        # ghi luồng thì zipfile không biết trước cỡ file; cả corpus > 4 GB nên phải bật zip64 từ đầu
        with z.open("docs.jsonl", "w", force_zip64=True) as out:
            for first in ((True, False) if priority else (None,)):
                with open(docs_path, encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        if first is not None and (int(r["doc_id"]) in priority) != first:
                            continue
                        out.write((json.dumps({"doc_id": int(r["doc_id"]), "language": r.get("language"),
                                               "text": r["text"]}, ensure_ascii=False) + "\n").encode("utf-8"))
                        n += 1
                        n_prio += bool(first)
        z.writestr("queries.jsonl", "".join(json.dumps({"id": q, "query": t}, ensure_ascii=False) + "\n" for q, t in qs))
        z.writestr("instructions.json", json.dumps(instructions, ensure_ascii=False, indent=2))
        z.writestr("meta.json", json.dumps({"name": name, "n_docs": n, "n_priority": n_prio, "n_queries": len(qs),
                                            "shard_size": shard_size,
                                            "query_source": "corrected", "docs_source": docs_path},
                                           ensure_ascii=False, indent=2))
    os.replace(zpath + ".part", zpath)
    return zpath, n, len(qs)


KERNEL_RUN = r'''"""Kaggle kernel: embed job __JOB__ bằng Qwen3-Embedding trên mọi GPU, rồi tìm top K. Ghi hết vào /kaggle/working.
Sinh bởi scripts/kaggle_embed.py; đừng sửa tay. embed_job.py nhúng ngay trong file này."""
import glob, os, subprocess, sys, threading, time

JOB = "__JOB__"
MODELS = __MODELS__        # [(tag, tên HF, ngân sách token/lô)]
PART = "__PART__"
EXTRA = __EXTRA__          # tham số thêm cho embed_job.py embed
TOP_K = __TOP_K__
MAX_HOURS = __MAX_HOURS__
EMBED_JOB_SRC = __EMBED_SRC__

W = "/kaggle/working"
LOG = open(f"{W}/log_{JOB}.txt", "a", encoding="utf-8")

def say(line):
    line = f"{time.strftime('%H:%M:%S')} {line.rstrip()}"
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)

def sh(cmd):
    say("$ " + cmd)
    subprocess.run(cmd, shell=True, check=True)

try:
    import transformers
    ok = tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 51)
except Exception:
    ok = False
if not ok:      # Qwen3 cần transformers >= 4.51
    sh(f"{sys.executable} -m pip install -q 'transformers>=4.51.0'")

script = f"{W}/embed_job.py"
open(script, "w", encoding="utf-8").write(EMBED_JOB_SRC)
import torch
n_gpu = max(torch.cuda.device_count(), 1)
say(f"GPU: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]} -> {n_gpu} worker")
sys.path.insert(0, W)
from embed_job import find_job      # cùng cách dò với worker: zip, thư mục <job>, hoặc meta.json có name == JOB
try:
    job = [find_job(JOB)]
except SystemExit as e:
    say(str(e))
    sys.exit("dataset đã gắn vào kernel và Kaggle xử lý xong chưa?")
say(f"job: {job[0]}")
deadline = time.time() + 3600 * MAX_HOURS

def complete(out):
    from embed_job import Job, my_shards
    meta = Job(job[0]).meta
    size = meta.get("shard_size") or int(EXTRA[EXTRA.index("--shard-size") + 1])
    want = my_shards(meta["n_docs"], size, "0/1", "0/1")
    return all(os.path.exists(f"{out}/docs_{s:05d}.npy") for s in want)

def tee(p, tag):
    for line in p.stdout:
        say(f"[{tag}] {line}")

for tag, model, batch_tokens in MODELS:
    out = f"{W}/{tag}"
    left = (deadline - time.time()) / 3600
    if left < 0.1:
        say(f"hết giờ, bỏ qua {tag}"); continue
    procs = []
    for w in range(n_gpu):
        cmd = [sys.executable, script, "embed", "--job", job[0], "--model", model, "--out", out, "--part", PART,
               "--worker", f"{w}/{n_gpu}", "--device", f"cuda:{w}" if torch.cuda.is_available() else "cpu",
               "--batch-tokens", str(batch_tokens), "--max-hours", f"{left:.2f}"] + EXTRA
        say(" ".join(cmd))
        p = subprocess.Popen(cmd, env=dict(os.environ, PYTHONUNBUFFERED="1"), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        t = threading.Thread(target=tee, args=(p, f"{tag} gpu{w}"), daemon=True); t.start()
        procs.append((p, t))
    for p, t in procs:
        try:
            p.wait(timeout=max(deadline - time.time() + 600, 60))
        except subprocess.TimeoutExpired:
            say("quá hạn chót, dừng worker"); p.kill()
        t.join(timeout=5)
    codes = [p.returncode for p, _ in procs]
    say(f"{tag}: exit {codes}")
    if any(codes):
        continue
    # Tìm ngay chỉ khi một phiên làm cả job; chia nhiều phiên thì mỗi phiên chỉ có một phần corpus,
    # tìm ở đây sẽ ra file ứng viên thiếu doc -> tìm ở local sau khi tải đủ các phần.
    if PART != "0/1" or not complete(out):
        say(f"{tag}: chưa đủ shard của cả job, bỏ qua bước tìm (chạy kaggle_embed search ở local)")
        continue
    for ins in sorted(os.path.basename(p)[len("queries_"):-4] for p in glob.glob(f"{out}/queries_*.npy")):
        try:
            sh(f"{sys.executable} {script} search --vecs {out} --instruction {ins} --top-k {TOP_K} "
               f"--retriever {tag}_{ins} --out {out}/cands_top{TOP_K}_{tag}_{ins}_qcorr.jsonl")
        except subprocess.CalledProcessError as e:
            say(f"search lỗi: {e}")
say("--- xong; output: " + ", ".join(sorted(os.listdir(W))))
'''


def write_kernel(name, owner, dataset_id, models, part="0/1", extra=None, top_k=3000, max_hours=11.0, suffix="", job=None):
    kdir = f"{ROOT}/kernel_{name}{suffix}"
    os.makedirs(kdir, exist_ok=True)
    code = (KERNEL_RUN.replace("__JOB__", job or name).replace("__MODELS__", repr([(t, *MODELS[t]) for t in models]))
            .replace("__PART__", part).replace("__EXTRA__", json.dumps(extra or []))
            .replace("__TOP_K__", str(top_k)).replace("__MAX_HOURS__", repr(float(max_hours)))
            .replace("__EMBED_SRC__", repr(open(EMBED_JOB, encoding="utf-8").read())))
    open(f"{kdir}/run.py", "w", encoding="utf-8").write(code)
    slug = f"vibiomir-{name.replace('_', '-')}{suffix.replace('_', '-')}"
    json.dump({"id": f"{owner}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4", "dataset_sources": [dataset_id],
               "competition_sources": [], "kernel_sources": [], "model_sources": []},
              open(f"{kdir}/kernel-metadata.json", "w", encoding="utf-8"), indent=2)
    return kdir, f"{owner}/{slug}"


def build(args):
    """Một zip job, --parts N kernel (mỗi kernel một phiên Kaggle <= 12 giờ, làm shard s % N == i)."""
    ins = read_instructions(INSTRUCTIONS)
    prio = priority_ids(args.priority_pool)
    zpath, n_docs, n_q = write_job(args.name, args.docs, ins, args.limit_queries, priority=prio, shard_size=args.shard_size)
    if prio:
        print(f"[{args.name}] doc ưu tiên (trong {args.priority_pool}) xếp lên đầu: {len(prio):,} id")
    dataset_id = ensure_dataset_meta(args.owner, args.dataset_slug, DATASET_DIR)
    owner = args.owner or dataset_id.split("/")[0]
    extra = ["--max-tokens", str(args.max_tokens), "--shard-size", str(args.shard_size)]
    kernels = []
    for i in range(args.parts):
        suffix = f"_p{i}of{args.parts}" if args.parts > 1 else ""
        kernels.append((suffix, *write_kernel(args.name, owner, dataset_id, args.models, f"{i}/{args.parts}", extra,
                                              args.top_k, args.max_hours, suffix=suffix, job=args.name)))
    mb = os.path.getsize(zpath) / 1e6
    print(f"[{args.name}] {n_docs:,} doc, {n_q} câu, instruction {sorted(ins)}, model {args.models}, "
          f"{args.parts} phiên -> {zpath} ({mb:.0f} MB)")
    print("Bước tiếp:")
    # kaggle CLI trên Windows hỏng với "/" trong -p (đường dẫn tạm .kaggle-uploads), nên in theo dấu của hệ điều hành
    ds = os.path.normpath(DATASET_DIR)
    print(f"  kaggle datasets version -p {ds} -m \"{args.name}\"     # lần đầu: kaggle datasets create -p {ds}")
    for suffix, kdir, kid in kernels:
        od = os.path.normpath(f"{OUT}/{args.name}{suffix}")
        print(f"  kaggle kernels push -p {os.path.normpath(kdir)}        # chờ dataset xử lý xong trên web rồi mới push")
        print(f"  kaggle kernels status {kid}")
        print(f"  kaggle kernels output {kid} -p {od}")
    print(f"  python -m scripts.kaggle_embed report --dir {os.path.normpath(OUT + '/' + args.name)}*")


def summarize(stats, n_gpu=2):
    """stats_*.json của một model -> tốc độ đo được và thời gian suy ra cho cả corpus."""
    n = sum(s["n"] for s in stats)
    enc = sum(s["encode_s"] + s["tokenize_s"] for s in stats)
    tok = sum(s["tokens"] for s in stats)
    devices = {s["device"] for s in stats}
    per_gpu = n / enc if enc else 0.0           # doc/s của MỘT GPU (thời gian cộng dồn qua các GPU), chưa tính nạp model
    hours = N_CORPUS / (per_gpu * n_gpu) / 3600 if per_gpu else float("inf")
    return {"docs": n, "gpus_used": len(devices), "doc_per_s_per_gpu": round(per_gpu, 2),
            "token_per_s_per_gpu": round(tok / enc, 0) if enc else 0, "tokens_per_doc": round(tok / max(n, 1), 0),
            "chars_per_doc": round(sum(s.get("chars", 0) for s in stats) / max(n, 1), 0),
            "truncated": sum(s["truncated"] for s in stats), "nan": sum(s["nan"] for s in stats),
            f"hours_full_corpus_{n_gpu}gpu": round(hours, 1)}


def report(args):
    dirs = sorted({os.path.dirname(p) for p in glob.glob(f"{args.dir}/**/stats_*.json", recursive=True)})
    if not dirs:
        raise SystemExit(f"không thấy stats_*.json dưới {args.dir}")
    for d in dirs:
        stats = [json.load(open(p)) for p in sorted(glob.glob(f"{d}/stats_*.json"))]
        s = summarize(stats, args.gpus)
        print(f"{d}: {stats[0]['model']}")
        for k, v in s.items():
            print(f"  {k:28s} {v}")
        cands = sorted(glob.glob(f"{d}/cands_*.jsonl"))
        print(f"  {'ứng viên':28s} {[os.path.basename(c) for c in cands] or 'chưa có'}")


def run_local(args):
    """Chạy tiếp trên máy có GPU (server): mỗi GPU một worker, bỏ qua shard đã có trong --done-dirs
    (output Kaggle đã tải về). Ctrl+C rồi chạy lại cùng lệnh thì làm tiếp từ shard còn thiếu."""
    import subprocess
    import threading
    n_gpu = args.gpus
    if not n_gpu:
        import torch
        n_gpu = max(torch.cuda.device_count(), 1)
    model, batch_tokens = MODELS[args.model]
    out = os.path.join(args.out, args.model)
    procs = []
    for w in range(n_gpu):
        cmd = [sys.executable, EMBED_JOB, "embed", "--job", args.job, "--model", model, "--out", out,
               "--part", args.part, "--worker", f"{w}/{n_gpu}", "--device", f"cuda:{w}", "--dtype", args.dtype,
               "--batch-tokens", str(args.batch_tokens or batch_tokens), "--max-tokens", str(args.max_tokens),
               "--max-hours", str(args.max_hours), "--engine", args.engine, "--gpu-mem", str(args.gpu_mem),
               *(["--reverse"] if args.reverse else []),
               "--done-dirs", *[os.path.join(d, args.model) for d in args.done_dirs]]
        print(" ".join(cmd), flush=True)
        procs.append(subprocess.Popen(cmd, env=dict(os.environ, PYTHONUNBUFFERED="1"), stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"))
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "log_local.txt"), "a", encoding="utf-8")

    def tee(p, w):
        for line in p.stdout:
            line = f"{time.strftime('%H:%M:%S')} [gpu{w}] {line.rstrip()}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

    ts = [threading.Thread(target=tee, args=(p, w), daemon=True) for w, p in enumerate(procs)]
    for t in ts:
        t.start()
    codes = [p.wait() for p in procs]
    for t in ts:
        t.join(timeout=5)
    print(f"[local] exit {codes} -> {out}")
    return codes


def search(args):
    mod = embed_job_module()
    out = args.out or f"{PHASE1}/cands_top{args.top_k}_{args.retriever}_qcorr.jsonl"
    mod.main(["search", "--vecs", *args.vecs, "--instruction", args.instruction, "--dim", str(args.dim),
              "--top-k", str(args.top_k), "--retriever", args.retriever, "--out", out,
              *(["--pool", args.pool, "--pool-mode", args.pool_mode] if args.pool else [])])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", help="dựng zip job + kernel")
    p.add_argument("--owner", default=os.environ.get("KAGGLE_USERNAME"))
    p.add_argument("--dataset-slug", default="vibiomir-embed-jobs")
    p.add_argument("--docs", required=True, help="jsonl của embed_q3e texts")
    p.add_argument("--name", required=True)
    p.add_argument("--models", nargs="+", default=["q3e06b", "q3e4b"], choices=sorted(MODELS))
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--shard-size", type=int, default=4000, help="nhỏ thì hết giờ mất ít việc; local PHẢI dùng cùng số")
    p.add_argument("--parts", type=int, default=1, help="chia N phiên Kaggle (N kernel, cùng một zip)")
    p.add_argument("--top-k", type=int, default=3000)
    p.add_argument("--max-hours", type=float, default=11.0)
    p.add_argument("--limit-queries", type=int, default=0)
    p.add_argument("--priority-pool", default=f"{PHASE1}/doc_top1000_bge_pool.jsonl",
                   help="doc trong pool này embed trước ('' = giữ thứ tự file)")
    p.set_defaults(fn=build)
    p = sub.add_parser("local", help="chạy tiếp job trên máy có GPU, bỏ qua shard đã làm ở Kaggle")
    p.add_argument("--job", required=True, help="file zip job (cùng file đã upload Kaggle)")
    p.add_argument("--model", default="q3e4b", choices=sorted(MODELS))
    p.add_argument("--out", required=True, help="thư mục gốc; vector ghi vào <out>/<model>")
    p.add_argument("--done-dirs", nargs="*", default=[], help="thư mục output Kaggle đã tải (chứa <model>/)")
    p.add_argument("--part", default="0/1")
    p.add_argument("--gpus", type=int, default=0, help="0 = mọi GPU thấy được")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16",
                   help="fp16 cho khớp vector đã làm trên T4")
    p.add_argument("--batch-tokens", type=int, default=0, help="0 = mặc định theo model; GPU lớn thì nâng")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-hours", type=float, default=1000.0)
    p.add_argument("--reverse", action="store_true", help="làm từ shard cuối lên trong lúc Kaggle làm từ đầu xuống")
    p.add_argument("--engine", choices=["hf", "vllm"], default="hf",
                   help="vllm: chạy bằng python của venv có vLLM (vd. .venv-vllm/bin/python)")
    p.add_argument("--gpu-mem", type=float, default=0.80, help="vLLM gpu_memory_utilization")
    p.set_defaults(fn=run_local)

    p = sub.add_parser("report", help="tốc độ đo được + thời gian suy ra cho 1.003.391 doc")
    p.add_argument("--dir", required=True)
    p.add_argument("--gpus", type=int, default=2)
    p.set_defaults(fn=report)
    p = sub.add_parser("search", help="tìm từ cache vector đã tải về (CPU, numpy)")
    p.add_argument("--vecs", nargs="+", required=True)
    p.add_argument("--instruction", default="p1")
    p.add_argument("--retriever", required=True, help="vd. q3e06b_p1")
    p.add_argument("--dim", type=int, default=0)
    p.add_argument("--top-k", type=int, default=3000)
    p.add_argument("--out", default=None)
    p.add_argument("--pool", default=None, help="pool jsonl giới hạn doc được tìm (vd. doc_top1000_bge_pool.jsonl)")
    p.add_argument("--pool-mode", choices=["own", "union"], default="own",
                   help="own: mỗi câu chỉ xếp lại pool của chính nó; union: tìm trên hợp pool của mọi câu")
    p.set_defaults(fn=search)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
