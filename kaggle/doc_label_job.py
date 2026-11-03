"""Kaggle: sinh nhãn doc 3 mức bằng Qwen3-8B trên mọi GPU (2 x T4 = 2 server vLLM). Logic ở doc_labels.py.

Job do `python -m src.rag.labeling.run_doc_labels kaggle` dựng ở local, nén thành <tên>.zip:
  meta.json          {"name", "prompt_sha", ...}
  prompt.txt         prompts/judge_doc_3level.txt
  candidates.jsonl   60 doc/câu (bước select)
  texts.jsonl        text 8k đã cắt còn phần reranker len 2048 thấy, cho mọi doc cần
  human.json         nhãn tay {qid: {"query", "labels": {doc_id: [điểm, ghi chú, hạng]}}}
  queries.jsonl      {"id", "query"} câu đã sửa, để embed tìm few-shot
  query_vecs.npz     (tuỳ chọn) có sẵn thì bỏ bước embed

Các bước (đều ghi vào --out, chạy lại là làm tiếp):
  embed   bge-m3 (sentence-transformers) trên cuda:0, tiến trình riêng để trả GPU trước khi vLLM chiếm
  serve   mỗi GPU một `vllm serve` cổng 8000+i, chờ /health
  agree   chấm các cặp gán tay -> agree_<sha>.jsonl + agree_<sha>_report.json
  judge   chấm 60 doc/câu -> judged_<sha>.jsonl

    python doc_label_job.py run --job doc_labels_v1 --out /kaggle/working --steps embed,serve,agree,judge --limit 50
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FILES = ("meta.json", "prompt.txt", "candidates.jsonl", "texts.jsonl", "human.json", "queries.jsonl")


def say(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def find_job(job, roots=("/kaggle/input",)):
    """Tên job -> thư mục chứa FILES. Nhận đường dẫn .zip / thư mục; còn không thì tìm dưới roots
    (Kaggle có thể giữ nguyên zip hoặc tự giải nén, và có thể đổi tên thư mục)."""
    cands = [job] if os.path.exists(job) else []
    for root in roots:
        cands += glob.glob(os.path.join(root, "**", f"{job}.zip"), recursive=True)
        for p in glob.glob(os.path.join(root, "**", "meta.json"), recursive=True):
            try:
                if json.load(open(p, encoding="utf-8")).get("name") == job:
                    cands.append(os.path.dirname(p))
            except (OSError, ValueError):
                continue
    for c in cands:
        if c.endswith(".zip"):
            # /kaggle/input chỉ đọc: giải nén ra thư mục tạm của hệ điều hành
            # Luôn giải nén lại: bản giải nén từ lần chạy trước có thể là của zip cũ cùng tên.
            dest = os.path.join(tempfile.gettempdir(), "job_" + os.path.basename(c)[:-4])
            shutil.rmtree(dest, ignore_errors=True)
            with zipfile.ZipFile(c) as z:
                z.extractall(dest)
            c = dest
        if all(os.path.exists(os.path.join(c, f)) for f in FILES):
            return c
    raise SystemExit(f"không tìm thấy job {job} (cần {FILES}) dưới {roots}")


def load_human(path):
    return {int(q): h for q, h in json.load(open(path, encoding="utf-8")).items()}


# ------------------------------------------------------------------ embed
def embed(job, out, model="BAAI/bge-m3", device="cuda:0"):
    """Vector dense bge-m3 (CLS, chuẩn hoá) của câu đã sửa. Cùng biểu diễn với BgeM3Embedder ở local."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import doc_labels as dl
    rows = [(int(r["id"]), r["query"]) for r in dl.read_jsonl(os.path.join(job, "queries.jsonl"))]
    m = SentenceTransformer(model, device=device)
    m.max_seq_length = 512
    vecs = m.encode([q for _, q in rows], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    np.savez(os.path.join(out, "query_vecs.npz"), ids=np.array([i for i, _ in rows]), vecs=np.asarray(vecs, np.float32))
    say(f"[embed] {len(rows)} câu -> {out}/query_vecs.npz")


# ------------------------------------------------------------------ vLLM
def start_servers(model, n_gpu, out, max_len, mem, max_seqs, extra, prefix_cache=True):
    """Mỗi GPU một server, cổng 8000+i. Tải model MỘT lần trước: hai server cùng tải vào một cache dễ giẫm nhau.
    Dùng chung cho kaggle/rerank_vllm_job.py (reranker, `--task score`, không prefix cache)."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(model)
    procs = []
    for i in range(n_gpu):
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", path,
               "--served-model-name", model, "--port", str(8000 + i), "--dtype", "half",
               "--max-model-len", str(max_len), "--gpu-memory-utilization", str(mem), "--max-num-seqs", str(max_seqs),
               *(["--enable-prefix-caching"] if prefix_cache else []), *extra]
        log = open(os.path.join(out, f"vllm_gpu{i}.log"), "a", encoding="utf-8")
        say(f"[serve] gpu{i}: {' '.join(cmd)}")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      # WARNING: bản vLLM cũ log nguyên văn mọi prompt ở INFO (hàng trăm MB);
                                      # cờ tắt log request lại đổi tên giữa các bản, biến môi trường thì không.
                                      env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), VLLM_USE_FLASHINFER_SAMPLER="0",
                                               VLLM_LOGGING_LEVEL=os.environ.get("VLLM_LOGGING_LEVEL", "WARNING"))))
    return procs


def wait_ready(procs, out, timeout=1800):
    import requests
    t0 = time.time()
    pending = set(range(len(procs)))
    while pending:
        for i in sorted(pending):
            if procs[i].poll() is not None:
                tail = open(os.path.join(out, f"vllm_gpu{i}.log"), encoding="utf-8", errors="replace").read()[-3000:]
                raise SystemExit(f"[serve] gpu{i} chết (exit {procs[i].returncode}):\n{tail}")
            try:
                if requests.get(f"http://127.0.0.1:{8000 + i}/health", timeout=5).ok:
                    say(f"[serve] gpu{i} sẵn sàng sau {time.time() - t0:.0f}s")
                    pending.discard(i)
            except requests.RequestException:
                pass
        if time.time() - t0 > timeout:
            raise SystemExit(f"[serve] {sorted(pending)} chưa sẵn sàng sau {timeout}s; xem vllm_gpu*.log")
        time.sleep(10)


# ------------------------------------------------------------------ chấm
def judge(job, out, args, urls, deadline):
    import numpy as np
    import doc_labels as dl
    template = open(os.path.join(job, "prompt.txt"), encoding="utf-8").read().strip()
    sha = dl.prompt_sha(template)
    meta = json.load(open(os.path.join(job, "meta.json"), encoding="utf-8"))
    if meta.get("prompt_sha") != sha:
        raise SystemExit(f"prompt.txt sha {sha} khác meta {meta.get('prompt_sha')}")
    texts = {str(r["doc_id"]): r["text"] for r in dl.read_jsonl(os.path.join(job, "texts.jsonl"))}
    human = load_human(os.path.join(job, "human.json"))
    vec_path = next(p for p in (os.path.join(out, "query_vecs.npz"), os.path.join(job, "query_vecs.npz"))
                    if os.path.exists(p))
    z = np.load(vec_path)
    vecs = dict(zip(z["ids"].tolist(), z["vecs"]))
    vars_of = dl.make_vars_of(human, texts, vecs, args.shots, args.per_label, args.shot_chars)
    chat = dl.make_chat(args.model, urls, max_tokens=args.max_tokens)
    cands = list(dl.read_jsonl(os.path.join(job, "candidates.jsonl")))
    run_meta = {"prompt": "judge_doc_3level", "prompt_sha": sha, "llm_model": args.model}
    ids_of = lambda r: [c["doc_id"] for c in r["candidates"]]
    kw = dict(batch_size=args.batch_size, seed=42, concurrency=args.concurrency, deadline=deadline)
    say(f"[judge] prompt sha {sha} | {len(urls)} server | {len(human)} câu gán tay | {len(cands)} câu")
    if "agree" in args.steps:
        path = os.path.join(out, f"agree_{sha}.jsonl")
        st = dl.run_judge(dl.agree_rows(human, cands), ids_of, vars_of, texts, template, dl.fill, chat, path, run_meta, **kw)
        rep = dl.agreement(dl.read_jsonl(path), human)
        json.dump(rep, open(os.path.join(out, f"agree_{sha}_report.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        say(f"[agree] {dict(st)}\n{json.dumps(rep, ensure_ascii=False, indent=2)}")
    if "judge" in args.steps:
        rows = cands[: args.limit] if args.limit else cands
        st = dl.run_judge(rows, ids_of, vars_of, texts, template, dl.fill, chat,
                          os.path.join(out, f"judged_{sha}.jsonl"), run_meta, **kw)
        say(f"[judge] {dict(st)}")


def run(args):
    t_end = time.time() + 3600 * args.max_hours
    job = find_job(args.job)
    os.makedirs(args.out, exist_ok=True)
    say(f"[job] {job} | bước {args.steps}")
    if "embed" in args.steps and not any(os.path.exists(os.path.join(d, "query_vecs.npz")) for d in (args.out, job)):
        subprocess.run([sys.executable, os.path.abspath(__file__), "embed", "--job", job, "--out", args.out], check=True)
    procs, urls = [], args.urls.split(",") if args.urls else []
    try:
        if "serve" in args.steps:
            import torch
            n_gpu = args.gpus or torch.cuda.device_count()
            procs = start_servers(args.model, n_gpu, args.out, args.max_model_len, args.gpu_mem, args.max_num_seqs,
                                  args.vllm_arg)
            wait_ready(procs, args.out)
            urls = [f"http://127.0.0.1:{8000 + i}/v1" for i in range(n_gpu)]
        if {"agree", "judge"} & set(args.steps):
            if not urls:
                raise SystemExit("cần bước serve hoặc --urls")
            # chừa 15 phút cho câu đang chấm và việc ghi output trước giới hạn phiên Kaggle
            judge(job, args.out, args, urls, t_end - 900)
    finally:
        for p in procs:
            p.terminate()
    say("[job] xong")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("embed")
    p.add_argument("--job", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--job", required=True)
    p.add_argument("--out", default="/kaggle/working")
    p.add_argument("--steps", default="embed,serve,agree,judge")
    p.add_argument("--limit", type=int, default=0, help="judge chỉ N câu đầu (chạy thử, đo tốc độ)")
    p.add_argument("--model", default="Qwen/Qwen3-8B-AWQ")
    p.add_argument("--urls", default="", help="server có sẵn, phẩy ngăn cách (bỏ bước serve)")
    p.add_argument("--gpus", type=int, default=0, help="0 = mọi GPU")
    p.add_argument("--max-model-len", type=int, default=12288, help="3 doc x ~1.930 token + few-shot + instruction")
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--vllm-arg", action="append", default=[], help="tham số thêm cho vllm serve, lặp lại được")
    p.add_argument("--concurrency", type=int, default=32, help="số câu chấm song song (tổng mọi server)")
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=320)
    p.add_argument("--shots", type=int, default=2)
    p.add_argument("--per-label", type=int, default=1)
    p.add_argument("--shot-chars", type=int, default=500)
    p.add_argument("--max-hours", type=float, default=11.5)
    a = ap.parse_args(argv)
    a.steps = a.steps.split(",") if hasattr(a, "steps") else []
    if a.cmd == "embed":
        embed(find_job(a.job), a.out)
    else:
        run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
