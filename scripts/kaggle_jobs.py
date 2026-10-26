"""Dựng job rerank cho Kaggle và gộp điểm về thành pool (T12, T14).

Chia việc: phần nhẹ chạy ở máy local (tìm Qdrant, lấy text doc, lắp bài nộp), phần nặng GPU
(rerank) chạy trên Kaggle bằng kaggle/rerank_job.py. Kaggle không cần Qdrant, không cần
snapshot index, không cần data/raw_documents: chỉ cần file job (câu hỏi + text doc + cặp cần chấm).

    # T14: chấm lại pool top300 bằng Qwen3-Reranker-4B (không cần Qdrant)
    python -m scripts.kaggle_jobs t14 --owner <kaggle-user>
    # T12: tìm top1000 trên Qdrant local, chỉ đưa lên Kaggle các cặp CHƯA có điểm bge trong các pool cũ
    python -m scripts.kaggle_jobs t12 --owner <kaggle-user>
    # sau khi tải điểm Kaggle về data/kaggle/out/<job>/
    python -m scripts.kaggle_jobs merge --name t14_qwen3r4b
    python -m scripts.kaggle_jobs merge --name t12_top1000_bge

Ra:
  data/kaggle/dataset/            thư mục upload dataset: dataset-metadata.json, rerank_job.py, <job>.zip
  data/kaggle/kernel_<job>/       thư mục push kernel: kernel-metadata.json, run.py
  data/kaggle/work/<job>.*        file local để gộp: mô tả job, ứng viên (T12)
  data/tmp/phase1/<pool>.jsonl    pool sau khi gộp, cùng định dạng phase1_doc_retrieve
"""
import argparse
import glob
import json
import os
import random
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = "data/kaggle"
DATASET_DIR = f"{ROOT}/dataset"
WORK = f"{ROOT}/work"
OUT = f"{ROOT}/out"
PHASE1 = "data/tmp/phase1"
DOC_TEXTS = "data/doc_texts.jsonl"
QUERIES = "data/queries/real_queries.jsonl"
BGE = "BAAI/bge-reranker-v2-m3"
QWEN4B = "Qwen/Qwen3-Reranker-4B"
# Pool phase1 chấm bằng bge-reranker-v2-m3, len 1024, câu GỐC: điểm dùng lại được làm cache.
BGE_POOLS = [f"{PHASE1}/{n}_pool.jsonl" for n in ("doc_d35_c44_top60_bge", "doc_top300_bge", "doc_top300_bge_vcorr_enzh",
                                                  "doc_top300_bge_vcorr_enzh_asp", "doc_top300_bge_vcorr_enzh_hyde")]
RERANK_JOB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kaggle", "rerank_job.py")


def jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_texts(path, need):
    """doc_id (str) -> text, chỉ cho các doc cần. data/doc_texts.jsonl là đúng text đã index và rerank."""
    out = {}
    for r in jsonl(path):
        d = str(r["doc_id"])
        if d in need:
            out[d] = r["text"]
    missing = need - out.keys()
    if missing:
        raise SystemExit(f"{len(missing)} doc không có trong {path}, vd {sorted(missing)[:5]}")
    return out


def load_cache(paths, rr_model=BGE, rr_len=1024):
    """(qid, doc_id) -> điểm rerank, từ các pool cùng reranker + max_length + câu gốc."""
    cache = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        for r in jsonl(path):
            if (r.get("rr_model", rr_model), r.get("rr_len", rr_len)) != (rr_model, rr_len):
                continue
            for c in r["candidates"]:
                cache[(int(r["id"]), str(c["doc_id"]))] = c["rerank_score"]
    return cache


def write_job(name, meta, queries, pairs, texts, check=None, dataset_dir=None):
    """Ghi <dataset_dir>/<name>.zip gồm meta.json, queries.jsonl, docs.jsonl, pairs.jsonl[, check.jsonl]."""
    dataset_dir = dataset_dir or DATASET_DIR
    os.makedirs(dataset_dir, exist_ok=True)
    zpath = os.path.join(dataset_dir, f"{name}.zip")
    tmp = zpath + ".part"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("meta.json", json.dumps({**meta, "name": name}, ensure_ascii=False, indent=2))
        z.writestr("queries.jsonl", "".join(json.dumps({"id": q, "query": t}, ensure_ascii=False) + "\n"
                                            for q, t in sorted(queries.items())))
        z.writestr("pairs.jsonl", "".join(json.dumps({"id": q, "doc_ids": ds}) + "\n" for q, ds in sorted(pairs.items()) if ds))
        z.writestr("docs.jsonl", "".join(json.dumps({"doc_id": d, "text": t}, ensure_ascii=False) + "\n"
                                         for d, t in sorted(texts.items())))
        if check:
            z.writestr("check.jsonl", "".join(json.dumps(r) + "\n" for r in check))
    os.replace(tmp, zpath)
    shutil.copy(RERANK_JOB, os.path.join(dataset_dir, "rerank_job.py"))
    return zpath


def ensure_dataset_meta(owner, slug, dataset_dir=None):
    dataset_dir = dataset_dir or DATASET_DIR
    path = os.path.join(dataset_dir, "dataset-metadata.json")
    if not os.path.exists(path):
        if not owner:
            raise SystemExit("lần đầu cần --owner <kaggle-username> để ghi dataset-metadata.json")
        os.makedirs(dataset_dir, exist_ok=True)
        json.dump({"title": "ViBioMIR rerank jobs", "id": f"{owner}/{slug}", "licenses": [{"name": "CC0-1.0"}]},
                  open(path, "w", encoding="utf-8"), indent=2)
    return json.load(open(path, encoding="utf-8"))["id"]


KERNEL_RUN = r'''"""Kaggle kernel: chấm job __JOB__ trên mọi GPU (mỗi GPU một shard), gộp điểm vào /kaggle/working.
Sinh bởi scripts/kaggle_jobs.py; đừng sửa tay, sinh lại thì sửa template trong đó."""
import glob, os, shutil, subprocess, sys, time

JOB = "__JOB__"
NEED_QWEN = __NEED_QWEN__
EXTRA = __EXTRA__          # tham số thêm cho rerank_job.py

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

if NEED_QWEN:
    try:
        import transformers
        ok = tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 51)
    except Exception:
        ok = False
    if not ok:
        sh(f"{sys.executable} -m pip install -q 'transformers>=4.51.0'")

script = sorted(glob.glob("/kaggle/input/**/rerank_job.py", recursive=True), key=len)
if not script:
    sys.exit("không thấy rerank_job.py trong /kaggle/input: đã gắn dataset chưa?")
script = script[0]

import torch
n = max(torch.cuda.device_count(), 1)
print(f"GPU: {n} x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
W = "/kaggle/working"
# Điểm của lượt chạy trước (nếu gắn làm input) -> bỏ qua câu đã chấm
prev = f"/kaggle/input/**/scores_{JOB}*.jsonl"
procs = []
for k in range(n):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k))
    out = f"{W}/scores_{JOB}.s{k}of{n}.jsonl"
    log = open(f"{W}/log_{JOB}.s{k}.txt", "w")
    cmd = [sys.executable, script, "--job", JOB, "--out", out, "--shard", f"{k}/{n}", "--device", "cuda:0",
           "--skip-done", prev] + EXTRA
    print(" ".join(cmd), flush=True)
    procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, k))
    time.sleep(20)          # nạp model lệch nhau cho đỡ tranh RAM
while any(p.poll() is None for p, _, _ in procs):
    time.sleep(60)
    for _, _, k in procs:
        lines = open(f"{W}/log_{JOB}.s{k}.txt", encoding="utf-8", errors="replace").read().splitlines()
        if lines:
            print(f"[s{k}] {lines[-1]}", flush=True)
for p, log, k in procs:
    log.close()
    print(f"--- log s{k} (exit {p.returncode}) ---")
    print("\n".join(open(f"{W}/log_{JOB}.s{k}.txt", encoding="utf-8", errors="replace").read().splitlines()[-15:]))
with open(f"{W}/scores_{JOB}.jsonl", "w", encoding="utf-8") as fh:
    for path in sorted(glob.glob(f"{W}/scores_{JOB}.s*.jsonl")) + sorted(glob.glob(prev, recursive=True)):
        fh.write(open(path, encoding="utf-8").read())
print("xong:", os.path.getsize(f"{W}/scores_{JOB}.jsonl"), "byte")
if any(p.returncode for p, _, _ in procs):
    sys.exit("có shard lỗi, xem log ở trên")
'''


def write_kernels(name, owner, dataset_id, need_qwen):
    """Hai kernel: bản chạy thử (20 câu mỗi GPU, đo tốc độ) và bản chạy thật."""
    write_kernel(name, owner, dataset_id, need_qwen, ["--limit-queries", "20", "--max-hours", "0.5"], suffix="_smoke")
    return write_kernel(name, owner, dataset_id, need_qwen)


def write_kernel(name, owner, dataset_id, need_qwen, extra=None, accelerator="NvidiaTeslaT4", suffix=""):
    kdir = f"{ROOT}/kernel_{name}{suffix}"
    os.makedirs(kdir, exist_ok=True)
    code = (KERNEL_RUN.replace("__JOB__", name).replace("__NEED_QWEN__", repr(bool(need_qwen)))
            .replace("__EXTRA__", json.dumps(extra or [])))
    open(f"{kdir}/run.py", "w", encoding="utf-8").write(code)
    slug = f"vibiomir-rerank-{name.replace('_', '-')}{suffix.replace('_', '-')}"
    json.dump({"id": f"{owner}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
               "machine_shape": accelerator, "dataset_sources": [dataset_id],
               "competition_sources": [], "kernel_sources": [], "model_sources": []},
              open(f"{kdir}/kernel-metadata.json", "w", encoding="utf-8"), indent=2)
    return kdir, f"{owner}/{slug}"


def save_local(name, info):
    os.makedirs(WORK, exist_ok=True)
    json.dump(info, open(f"{WORK}/{name}.local.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def report(name, zpath, kdir, kernel_id, n_q, n_pairs, n_docs):
    mb = os.path.getsize(zpath) / 1e6
    print(f"\n[{name}] {n_q} câu, {n_pairs:,} cặp, {n_docs:,} doc -> {zpath} ({mb:.0f} MB)")
    print("Bước tiếp (chạy ở máy có kaggle CLI):")
    print(f"  kaggle datasets create -p {DATASET_DIR}        # lần đầu; lần sau: kaggle datasets version -p {DATASET_DIR} -m \"{name}\"")
    print(f"  kaggle kernels push -p {kdir}_smoke      # chạy thử 20 câu/GPU, đọc tốc độ trong log trước khi chạy thật")
    print(f"  kaggle kernels push -p {kdir}")
    print(f"  kaggle kernels status {kernel_id}")
    print(f"  kaggle kernels output {kernel_id} -p {OUT}/{name} --file-pattern \"scores_.*jsonl|log_.*txt\"")
    print(f"  python -m scripts.kaggle_jobs merge --name {name}")


# ------------------------------------------------------------------ T14
def build_rescore(args):
    """Chấm lại MỌI doc của một pool có sẵn bằng reranker khác (T14)."""
    rows = list(jsonl(args.pool))
    queries = {int(r["id"]): r["query"] for r in rows}
    pairs = {}
    for r in rows:
        cs = sorted(r["candidates"], key=lambda c: -c["rerank_score"])
        if args.top_k:
            cs = cs[: args.top_k]
        pairs[int(r["id"])] = [str(c["doc_id"]) for c in cs]
    need = {d for ds in pairs.values() for d in ds}
    texts = load_texts(args.doc_texts, need)
    meta = {"model": args.model, "kind": "qwen3" if "qwen3-reranker" in args.model.lower() else "cross-encoder",
            "max_length": args.max_length, "source_pool": args.pool, "top_k": args.top_k}
    zpath = write_job(args.name, meta, queries, pairs, texts)
    save_local(args.name, {"mode": "rescore", "base_pool": args.pool, "top_k": args.top_k, "rr_model": args.model,
                           "rr_len": args.max_length, "out_pool": args.out_pool})
    dataset_id = ensure_dataset_meta(args.owner, args.dataset_slug)
    kdir, kid = write_kernels(args.name, args.owner or dataset_id.split("/")[0], dataset_id, meta["kind"] == "qwen3")
    report(args.name, zpath, kdir, kid, len(queries), sum(map(len, pairs.values())), len(texts))


# ------------------------------------------------------------------ T12
def search_qdrant(top_k, ef, limit=0, batch=16):
    """Câu gốc -> [(qid, query, [doc_id theo hạng])], dense BGE-M3 trên medical_docs như pool top300."""
    from qdrant_client import QdrantClient

    from scripts.phase1_doc_retrieve import request
    from src.rag.config import load_config
    from src.rag.embedding import BgeM3Embedder
    cfg = load_config(None, [])
    qs = list(jsonl(QUERIES))[: limit or None]
    client = QdrantClient(url=cfg.get("qdrant.url", "http://localhost:6333"), timeout=cfg.get("qdrant.timeout", 600))
    coll = cfg.get("qdrant.collection_docs", "medical_docs")
    if not client.collection_exists(coll):
        raise SystemExit(f"Qdrant chưa có collection {coll}: bật Docker và restore (scripts/restore_medical_docs.sh)")
    vecs = BgeM3Embedder().encode([q["query"] for q in qs], batch_size=cfg.get("retrieval.doc.encode_batch", 8))
    out = []
    for i in range(0, len(qs), batch):
        part = qs[i: i + batch]
        res = client.query_batch_points(collection_name=coll, requests=[
            request(v.tolist() if hasattr(v, "tolist") else list(v), None, top_k, False, ef)
            for v in vecs[i: i + batch]])
        for q, r in zip(part, res):
            out.append((int(q["id"]), q["query"], [str((p.payload or {}).get("doc_id")) for p in r.points]))
        print(f"[t12] tìm {min(i + batch, len(qs))}/{len(qs)}", end="\r", flush=True)
    print()
    return out


def plan_deep(hits, cache, n_check, seed=7):
    """Ứng viên + cặp cần chấm + mẫu kiểm thang điểm. Tách khỏi Qdrant để test được."""
    pairs, check = {}, []
    for qid, _, ids in hits:
        pairs[qid] = [d for d in ids if (qid, d) not in cache]
    cached = [(qid, d) for qid, _, ids in hits for d in ids if (qid, d) in cache]
    rnd = random.Random(seed)
    for qid, d in rnd.sample(cached, min(n_check, len(cached))):
        check.append({"id": qid, "doc_id": d, "expected": cache[(qid, d)]})
    return pairs, check


def build_deep(args):
    hits = search_qdrant(args.top_docs, args.hnsw_ef, args.limit)
    cache = load_cache(args.cache)
    pairs, check = plan_deep(hits, cache, args.n_check)
    os.makedirs(WORK, exist_ok=True)
    cand_path = f"{WORK}/{args.name}.candidates.jsonl"
    with open(cand_path, "w", encoding="utf-8") as fh:
        for qid, query, ids in hits:
            fh.write(json.dumps({"id": qid, "query": query, "doc_ids": ids}, ensure_ascii=False) + "\n")
    need = {d for ds in pairs.values() for d in ds} | {r["doc_id"] for r in check}
    texts = load_texts(args.doc_texts, need)
    queries = {qid: query for qid, query, _ in hits}
    meta = {"model": BGE, "kind": "cross-encoder", "max_length": 1024, "top_docs": args.top_docs,
            "hnsw_ef": args.hnsw_ef}
    zpath = write_job(args.name, meta, queries, pairs, texts, check)
    save_local(args.name, {"mode": "deep", "candidates": cand_path, "cache": args.cache, "rr_model": BGE,
                           "rr_len": 1024, "out_pool": args.out_pool})
    n_cached = sum(len(ids) for _, _, ids in hits) - sum(map(len, pairs.values()))
    print(f"[t12] {n_cached:,} cặp đã có điểm bge trong pool cũ, không chấm lại")
    dataset_id = ensure_dataset_meta(args.owner, args.dataset_slug)
    kdir, kid = write_kernels(args.name, args.owner or dataset_id.split("/")[0], dataset_id, False)
    report(args.name, zpath, kdir, kid, len(queries), sum(map(len, pairs.values())), len(texts))


# ------------------------------------------------------------------ gộp
def read_scores(paths):
    scores, nulls = {}, 0
    for path in paths:
        for r in jsonl(path):
            for d, s in r["scores"].items():
                if s is None:
                    nulls += 1
                    continue
                scores[(int(r["id"]), str(d))] = s
    return scores, nulls


def doc_meta(path, need):
    out = {}
    for r in jsonl(path):
        d = str(r["doc_id"])
        if d in need:
            out[d] = {"language": r.get("language"), "title": r.get("title")}
    return out


def merge_rows(info, scores, cache=None, meta_of=None):
    """-> dòng pool (id, query, rr_model, rr_len, candidates đã xếp) và số cặp thiếu điểm."""
    rows, missing = [], 0
    if info["mode"] == "rescore":
        for r in jsonl(info["base_pool"]):
            q = int(r["id"])
            cs = sorted(r["candidates"], key=lambda c: -c["rerank_score"])
            if info.get("top_k"):
                cs = cs[: info["top_k"]]
            new = []
            for c in cs:
                s = scores.get((q, str(c["doc_id"])))
                if s is None:
                    missing += 1
                    continue
                new.append({**c, "rerank_score_prev": c["rerank_score"], "rerank_score": s})
            new.sort(key=lambda c: -c["rerank_score"])
            rows.append({"id": q, "query": r["query"], "rr_model": info["rr_model"], "rr_len": info["rr_len"],
                         "candidates": new})
    else:
        for r in jsonl(info["candidates"]):
            q = int(r["id"])
            new = []
            for rank, d in enumerate(r["doc_ids"], 1):
                s = scores.get((q, d), cache.get((q, d)))
                if s is None:
                    missing += 1
                    continue
                m = meta_of.get(d, {})
                new.append({"doc_id": d, "language": m.get("language"), "title": m.get("title"),
                            "rrf_score": round(1.0 / (60 + rank), 5), "found_by": ["orig"], "rerank_score": float(s)})
            new.sort(key=lambda c: -c["rerank_score"])
            rows.append({"id": q, "query": r["query"], "rr_model": info["rr_model"], "rr_len": info["rr_len"],
                         "candidates": new})
    return rows, missing


def merge(args):
    info = json.load(open(f"{WORK}/{args.name}.local.json", encoding="utf-8"))
    paths = sorted(glob.glob(args.scores or f"{OUT}/{args.name}/**/scores_{args.name}.s*.jsonl", recursive=True))
    if not paths:
        raise SystemExit(f"không thấy file điểm shard nào ({args.scores or OUT + '/' + args.name})")
    scores, nulls = read_scores(paths)
    cache, meta_of = None, None
    if info["mode"] == "deep":
        cache = load_cache(info["cache"])
        need = {d for r in jsonl(info["candidates"]) for d in r["doc_ids"]}
        meta_of = doc_meta(args.doc_texts, need)
    rows, missing = merge_rows(info, scores, cache, meta_of)
    print(f"[merge] {len(paths)} file điểm, {len(scores):,} điểm Kaggle, {nulls} điểm NaN bỏ, {missing:,} cặp thiếu điểm")
    if missing and not args.allow_missing:
        raise SystemExit("còn cặp thiếu điểm: Kaggle chưa chạy xong (chạy tiếp lượt sau) hoặc thêm --allow-missing")
    out = args.out or info["out_pool"]
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[merge] -> {out} ({len(rows)} câu, TB {sum(len(r['candidates']) for r in rows) / max(len(rows), 1):.0f} doc/câu)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--owner", default=os.environ.get("KAGGLE_USERNAME"), help="tên user Kaggle")
    common.add_argument("--dataset-slug", default="vibiomir-rerank-jobs")
    common.add_argument("--doc-texts", default=DOC_TEXTS)

    p = sub.add_parser("t14", parents=[common], help="chấm lại một pool bằng reranker khác")
    p.add_argument("--pool", default=f"{PHASE1}/doc_top300_bge_pool.jsonl")
    p.add_argument("--model", default=QWEN4B)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--top-k", type=int, default=0, help="chỉ chấm K doc đầu theo điểm cũ mỗi câu (0 = cả pool)")
    p.add_argument("--name", default="t14_qwen3r4b")
    p.add_argument("--out-pool", default=f"{PHASE1}/doc_top300_qwen3r4b_pool.jsonl")
    p.set_defaults(fn=build_rescore)

    p = sub.add_parser("t12", parents=[common], help="tìm sâu trên Qdrant local, rerank phần mới trên Kaggle")
    p.add_argument("--top-docs", type=int, default=1000)
    p.add_argument("--hnsw-ef", type=int, default=1000, help=">= top-docs, không thì đuôi danh sách kém chính xác")
    p.add_argument("--cache", nargs="+", default=BGE_POOLS)
    p.add_argument("--n-check", type=int, default=400, help="số cặp đã chấm local đưa lên để kiểm thang điểm")
    p.add_argument("--limit", type=int, default=0, help="chỉ N câu đầu, để thử")
    p.add_argument("--name", default="t12_top1000_bge")
    p.add_argument("--out-pool", default=f"{PHASE1}/doc_top1000_bge_pool.jsonl")
    p.set_defaults(fn=build_deep)

    p = sub.add_parser("merge", parents=[common], help="gộp điểm Kaggle thành pool")
    p.add_argument("--name", required=True)
    p.add_argument("--scores", default=None, help="glob file điểm (mặc định data/kaggle/out/<name>/**)")
    p.add_argument("--out", default=None)
    p.add_argument("--allow-missing", action="store_true")
    p.set_defaults(fn=merge)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
