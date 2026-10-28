"""Embed corpus + câu hỏi bằng Qwen3-Embedding trên GPU Kaggle, rồi tìm chính xác (không Qdrant).
Tự chứa: chỉ cần numpy, torch, transformers; không import gì từ repo.

Job do scripts/kaggle_embed.py dựng ở máy local, nén thành <tên>.zip:
  meta.json          {"name", "n_docs", "query_source", ...}
  docs.jsonl         {"doc_id", "language", "text"}       text 8k của scripts/embed_q3e.py texts
  queries.jsonl      {"id", "query"}                      câu SỬA (CLAUDE.md, 2026-10-27)
  instructions.json  {"p1": "...", "p2": "..."}           prompts/q3e_instruct_*.txt

Ra (cache, tải về hết được), trong --out (mỗi model một thư mục):
  docs_<s>.npy / docs_<s>.ids.npy   vector doc fp16, ĐỦ chiều (cắt Matryoshka lúc tìm), shard s theo thứ tự docs.jsonl
  queries_<p>.npy, queries.ids.npy  vector câu hỏi cho từng instruction; queries.jsonl, instructions.json đi kèm
  stats_<s>.json                    tốc độ, token, số doc bị cắt, NaN của từng shard
  cands_top<K>_<model>_<p>_qcorr.jsonl   (lệnh search) cùng định dạng scripts/kaggle_jobs.write_candidates

Chia việc: shard s = dòng [s*shard_size, (s+1)*shard_size). --part p/P chọn shard s % P == p (chia nhiều phiên
Kaggle 12 giờ); --worker w/W chia tiếp các shard đó cho W GPU. Shard đã có file thì bỏ qua (chạy lại an toàn).

    python embed_job.py embed --job JOB.zip --model Qwen/Qwen3-Embedding-0.6B --out out/q3e06b --worker 0/2 --device cuda:0
    python embed_job.py search --vecs out/q3e06b --instruction p1 --top-k 3000 --retriever q3e06b_p1
"""
import argparse
import glob
import io
import json
import os
import sys
import time
import zipfile

FILES = ("meta.json", "docs.jsonl", "queries.jsonl", "instructions.json")


# ------------------------------------------------------------------ đọc job
class Job:
    """Đọc job từ file zip hoặc thư mục đã giải nén (Kaggle có thể tự giải nén dataset)."""

    def __init__(self, path):
        self.zip = zipfile.ZipFile(path) if path.endswith(".zip") else None
        names = self.zip.namelist() if self.zip else os.listdir(path)
        self.prefix = ""
        if "meta.json" not in names:
            inner = [n for n in names if n.endswith("/meta.json")] if self.zip else \
                [f"{d}/meta.json" for d in names if os.path.isfile(os.path.join(path, d, "meta.json"))]
            if not inner:
                raise SystemExit(f"{path}: không thấy meta.json")
            self.prefix = inner[0][: -len("meta.json")]
        self.path = path
        self.meta = json.loads(self.read("meta.json"))

    def open(self, name):
        if self.zip:
            return io.TextIOWrapper(self.zip.open(self.prefix + name), encoding="utf-8")
        return open(os.path.join(self.path, self.prefix, name), encoding="utf-8")

    def read(self, name):
        with self.open(name) as fh:
            return fh.read()

    def jsonl(self, name):
        with self.open(name) as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def find_job(job, root="/kaggle/input"):
    """Tên job -> đường dẫn .zip hoặc thư mục. Kaggle tự giải nén dataset một-zip thẳng vào gốc dataset
    (mất tên <job>), nên cuối cùng dò meta.json có "name" == job."""
    if os.path.exists(job):
        return job
    hits = (glob.glob(f"{root}/**/{job}.zip", recursive=True)
            + [os.path.dirname(p) for p in glob.glob(f"{root}/**/{job}/meta.json", recursive=True)])
    if hits:
        return sorted(hits, key=len)[0]
    for p in glob.glob(f"{root}/**/meta.json", recursive=True):
        try:
            if json.load(open(p, encoding="utf-8")).get("name") == job:
                return os.path.dirname(p)
        except (OSError, ValueError):
            continue
    seen = sorted(glob.glob(f"{root}/*/*") + glob.glob(f"{root}/*/*/*"))[:40]
    raise SystemExit(f"không thấy job {job} dưới {root}; đang có: {seen}")


def split(spec):
    k, n = (int(x) for x in spec.split("/"))
    assert 0 <= k < n, spec
    return k, n


def my_shards(n_docs, shard_size, part, worker):
    """Phần p/P = một KHỐI shard liền nhau (docs.jsonl xếp doc ưu tiên lên đầu, nên phần 0 quan trọng nhất);
    trong khối, các GPU lấy xen kẽ để cùng tiến từ đầu khối."""
    (p, P), (w, W) = split(part), split(worker)
    n = (n_docs + shard_size - 1) // shard_size
    mine = list(range(p * n // P, (p + 1) * n // P))
    return [s for i, s in enumerate(mine) if i % W == w]


def save_npy(path, arr):
    """Ghi qua file tạm rồi đổi tên: kernel bị giết giữa chừng không để lại .npy cụt (cụt mà vẫn 'có' thì bị bỏ qua mãi)."""
    import numpy as np
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def shard_done(s, dirs):
    return any(os.path.exists(f"{d}/docs_{s:05d}.npy") for d in dirs)


# ------------------------------------------------------------------ encode
class Encoder:
    """Qwen3-Embedding qua transformers: đệm TRÁI, vector = hidden state của token cuối (<|endoftext|>),
    như model card. Batch theo ngân sách token (độn x số câu), OOM thì tự giảm ngân sách và chạy lại."""

    def __init__(self, model, device, dtype, max_tokens, batch_tokens):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        try:
            self.model = AutoModel.from_pretrained(model, dtype=dt).to(device).eval()
        except TypeError:   # transformers cũ: torch_dtype
            self.model = AutoModel.from_pretrained(model, torch_dtype=dt).to(device).eval()
        self.device, self.max_tokens, self.batch_tokens = device, max_tokens, batch_tokens
        self.eod = self.tok.convert_tokens_to_ids("<|endoftext|>")
        probe = self.tok("a", add_special_tokens=True)["input_ids"]
        self.auto_eod = bool(probe) and probe[-1] == self.eod
        self.dim = self.model.config.hidden_size

    def ids(self, texts):
        """Cắt còn max_tokens-1 rồi chắc chắn token cuối là <|endoftext|> (tokenizer cũ không tự thêm)."""
        out, cut = [], 0
        for x in self.tok(texts, add_special_tokens=True)["input_ids"]:
            if self.auto_eod and x and x[-1] == self.eod:
                x = x[:-1]
            cut += len(x) > self.max_tokens - 1
            out.append(x[: self.max_tokens - 1] + [self.eod])
        return out, cut

    def encode_ids(self, ids):
        """-> (np.float16 [n, dim] chưa chuẩn hoá, theo đúng thứ tự ids)."""
        import numpy as np
        torch = self.torch
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))      # dài trước: OOM lộ ra ngay lô đầu
        out = np.zeros((len(ids), self.dim), dtype=np.float16)
        i = 0
        total_tok, done_tok = sum(len(x) for x in ids), 0
        t0 = t_log = time.time()
        while i < len(order):
            if time.time() - t_log >= 60:      # tiến độ trong shard, mỗi phút một dòng
                t_log = time.time()
                rate = done_tok / max(t_log - t0, 1e-9)
                print(f"[embed]   {i}/{len(order)} doc, {done_tok / total_tok:.0%} token, {rate:,.0f} token/s, "
                      f"còn ~{(total_tok - done_tok) / max(rate, 1e-9) / 60:.0f} phút", flush=True)
            n = max(1, self.batch_tokens // len(ids[order[i]]))
            idx = order[i: i + n]
            try:
                batch = self.tok.pad({"input_ids": [ids[j] for j in idx]}, padding=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    h = self.model(**batch).last_hidden_state[:, -1, :]
                out[idx] = h.float().cpu().numpy().astype(np.float16)
                i += len(idx)
                done_tok += sum(len(ids[j]) for j in idx)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.batch_tokens <= len(ids[order[i]]):
                    raise
                self.batch_tokens //= 2
                print(f"[embed] OOM, giảm ngân sách còn {self.batch_tokens} token/lô", flush=True)
        return out


def format_query(query, instruction):
    return f"Instruct: {instruction}\nQuery: {query}"


# ------------------------------------------------------------------ lệnh embed
def embed(args, enc=None):
    import numpy as np
    job = Job(find_job(args.job))
    os.makedirs(args.out, exist_ok=True)
    n_docs = min(job.meta["n_docs"], args.limit_docs or job.meta["n_docs"])
    # shard_size của job thắng tham số: phiên Kaggle và server phải chia shard y hệt nhau mới ghép được
    args.shard_size = job.meta.get("shard_size") or args.shard_size
    shards = my_shards(n_docs, args.shard_size, args.part, args.worker)
    todo = [s for s in shards if not shard_done(s, [args.out, *args.done_dirs])]
    if getattr(args, "reverse", False):     # server làm từ cuối lên, Kaggle từ đầu xuống: không giẫm chân nhau
        todo.reverse()
    first = split(args.part)[0] == 0 and split(args.worker)[0] == 0
    need_q = first and not any(os.path.exists(f"{d}/queries.ids.npy") for d in [args.out, *args.done_dirs])
    print(f"[embed] {args.model} | worker {args.worker} part {args.part} | {len(shards)} shard, còn {len(todo)}"
          + (" + câu hỏi" if need_q else ""), flush=True)
    if not todo and not need_q:
        return
    t0 = time.time()
    enc = enc or Encoder(args.model, args.device, args.dtype, args.max_tokens, args.batch_tokens)
    load_s = time.time() - t0
    print(f"[embed] nạp model {load_s:.0f}s | {enc.dim} chiều | tokenizer tự thêm <|endoftext|>: {enc.auto_eod}", flush=True)
    deadline = time.time() + 3600 * args.max_hours

    if need_q:    # câu hỏi ngắn, làm trước: lần thử nào cũng có vector câu hỏi để kiểm đường ống tìm
        qs = list(job.jsonl("queries.jsonl"))
        ins = json.loads(job.read("instructions.json"))
        for tag, text in sorted(ins.items()):
            ids, cut = enc.ids([format_query(q["query"], text) for q in qs])
            t = time.time()
            v = enc.encode_ids(ids)
            save_npy(f"{args.out}/queries_{tag}.npy", v)
            print(f"[query] {tag}: {len(qs)} câu, {time.time() - t:.0f}s, {cut} câu bị cắt, NaN {int(np.isnan(v).any(1).sum())}",
                  flush=True)
        save_npy(f"{args.out}/queries.ids.npy", np.array([int(q["id"]) for q in qs], dtype=np.int64))
        with open(f"{args.out}/queries.jsonl", "w", encoding="utf-8") as fh:
            for q in qs:
                fh.write(json.dumps({"id": int(q["id"]), "query": q["query"]}, ensure_ascii=False) + "\n")
        json.dump({**ins, "_query_source": job.meta.get("query_source")},
                  open(f"{args.out}/instructions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    want = set(todo)
    rows = {s: [] for s in todo}
    for i, r in enumerate(job.jsonl("docs.jsonl")):
        if i >= n_docs:
            break
        s = i // args.shard_size
        if s in want:
            rows[s].append(r)
    done_docs, start = 0, time.time()
    for s in todo:
        if time.time() > deadline:
            print(f"[embed] hết giờ (--max-hours {args.max_hours}), dừng trước shard {s}", flush=True)
            break
        rs = rows.pop(s)
        t = time.time()
        ids, cut = enc.ids([r["text"] for r in rs])
        t_tok = time.time() - t
        t = time.time()
        v = enc.encode_ids(ids)
        t_enc = time.time() - t
        nan = int(np.isnan(v).any(1).sum())
        save_npy(f"{args.out}/docs_{s:05d}.ids.npy", np.array([int(r["doc_id"]) for r in rs], dtype=np.int64))
        save_npy(f"{args.out}/docs_{s:05d}.npy", v)       # ghi vector SAU ids: có .npy là shard đã đủ
        lens = [len(x) for x in ids]
        by_lang = {}
        for r, n in zip(rs, lens):
            d = by_lang.setdefault(r.get("language") or "?", [0, 0])
            d[0] += 1
            d[1] += n
        stats = {"shard": s, "model": args.model, "device": args.device, "dtype": args.dtype, "n": len(rs),
                 "tokens": sum(lens), "max_tokens": args.max_tokens, "truncated": cut, "nan": nan,
                 "tokenize_s": round(t_tok, 2), "encode_s": round(t_enc, 2), "load_model_s": round(load_s, 1),
                 "batch_tokens_end": enc.batch_tokens, "chars": sum(len(r["text"]) for r in rs),
                 "by_language": {k: {"n": a, "tokens": b} for k, (a, b) in by_lang.items()}}
        json.dump(stats, open(f"{args.out}/stats_{s:05d}.json", "w"), indent=2)
        done_docs += len(rs)
        el = time.time() - start
        print(f"[embed] shard {s}: {len(rs)} doc, {t_enc:.0f}s encode ({len(rs) / max(t_enc, 1e-9):.1f} doc/s, "
              f"{sum(lens) / max(t_enc, 1e-9):,.0f} token/s), TB {sum(lens) / max(len(rs), 1):.0f} token/doc, "
              f"{cut} doc bị cắt, NaN {nan} | worker: {done_docs:,} doc trong {el / 60:.1f} phút", flush=True)


# ------------------------------------------------------------------ lệnh search
def load_doc_shards(dirs):
    import numpy as np
    paths = sorted({os.path.basename(p): p for d in dirs for p in glob.glob(f"{d}/docs_*.npy")
                    if not p.endswith(".ids.npy")}.values())
    for p in paths:
        yield p, np.load(p[:-4] + ".ids.npy")


def normalize(v, dim):
    import numpy as np
    v = np.asarray(v, dtype=np.float32)[:, : dim or None]
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def search(args):
    """Tìm CHÍNH XÁC (nhân ma trận, không HNSW): cosine trên vector đã cắt --dim chiều rồi chuẩn hoá lại."""
    import numpy as np
    dirs = args.vecs
    qdir = next(d for d in dirs if os.path.exists(f"{d}/queries_{args.instruction}.npy"))
    q = normalize(np.load(f"{qdir}/queries_{args.instruction}.npy"), args.dim)
    qids = np.load(f"{qdir}/queries.ids.npy")
    text = {int(r["id"]): r["query"] for r in map(json.loads, open(f"{qdir}/queries.jsonl", encoding="utf-8"))}
    src = json.load(open(f"{qdir}/instructions.json", encoding="utf-8")).get("_query_source") or "corrected"
    best_s = np.full((len(q), 0), -np.inf, dtype=np.float32)
    best_d = np.zeros((len(q), 0), dtype=np.int64)
    n, t = 0, time.time()
    for path, ids in load_doc_shards(dirs):
        d = normalize(np.load(path), args.dim)
        bad = ~np.isfinite(d).all(1)
        s = q @ d.T
        s[:, bad] = -np.inf                               # vector NaN không được lọt vào top
        all_s = np.concatenate([best_s, s], 1)
        all_d = np.concatenate([best_d, np.broadcast_to(ids, s.shape)], 1)
        k = min(args.top_k, all_s.shape[1])
        part = np.argpartition(-all_s, k - 1, axis=1)[:, :k]
        best_s, best_d = np.take_along_axis(all_s, part, 1), np.take_along_axis(all_d, part, 1)
        n += len(ids)
    order = np.argsort(-best_s, axis=1, kind="stable")
    best_s, best_d = np.take_along_axis(best_s, order, 1), np.take_along_axis(best_d, order, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, qid in enumerate(qids):
            keep = np.isfinite(best_s[i])
            fh.write(json.dumps({"id": int(qid), "query": text[int(qid)], "retriever": args.retriever,
                                 "query_source": src, "doc_ids": [str(x) for x in best_d[i][keep]],
                                 "scores": [round(float(x), 6) for x in best_s[i][keep]]}, ensure_ascii=False) + "\n")
    print(f"[search] {len(qids)} câu x {n:,} doc, {args.dim or q.shape[1]} chiều, top {args.top_k}, "
          f"{time.time() - t:.0f}s -> {args.out}", flush=True)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("embed")
    p.add_argument("--job", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--part", default="0/1")
    p.add_argument("--worker", default="0/1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16", help="T4 không có bf16 thật")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--batch-tokens", type=int, default=16384, help="ngân sách token (đã độn) mỗi lô")
    p.add_argument("--shard-size", type=int, default=20000)
    p.add_argument("--limit-docs", type=int, default=0)
    p.add_argument("--reverse", action="store_true", help="làm shard từ cuối lên")
    p.add_argument("--done-dirs", nargs="*", default=[], help="thư mục output của các phiên khác: shard có ở đó thì bỏ qua")
    p.add_argument("--max-hours", type=float, default=11.0)
    p.set_defaults(fn=embed)
    p = sub.add_parser("search")
    p.add_argument("--vecs", nargs="+", required=True, help="thư mục chứa docs_*.npy (nhiều phiên thì nhiều thư mục)")
    p.add_argument("--instruction", default="p1")
    p.add_argument("--dim", type=int, default=0, help="cắt còn N chiều (0 = đủ)")
    p.add_argument("--top-k", type=int, default=3000)
    p.add_argument("--retriever", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=search)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
