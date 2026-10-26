"""Chấm một job rerank trên Kaggle (T12, T14). Tự chứa: không import gì từ repo.

Job do scripts/kaggle_jobs.py dựng ở máy local, nén thành <tên>.zip:
  meta.json      {"name", "model", "kind": "cross-encoder" | "qwen3", "max_length", ...}
  queries.jsonl  {"id", "query"}                 câu GỐC tiếng Việt, đúng câu reranker local dùng
  docs.jsonl     {"doc_id", "text"}              text đã index (data/doc_texts.jsonl)
  pairs.jsonl    {"id", "doc_ids": [...]}        cặp cần chấm của từng câu
  check.jsonl    {"id", "doc_id", "expected"}    (tuỳ chọn) cặp đã chấm ở local, để kiểm điểm Kaggle cùng thang

Kaggle có thể giữ nguyên file zip hoặc tự giải nén khi nạp dataset; --job nhận cả hai
(đường dẫn .zip, thư mục đã giải nén, hoặc chỉ tên job thì tự tìm dưới /kaggle/input).

Ra: mỗi dòng {"id", "scores": {doc_id: điểm}}, ghi tới đâu flush tới đó. Chạy lại cùng --out
thì bỏ qua câu đã chấm. --shard k/n chia câu theo id để chạy song song trên nhiều GPU.
--max-hours dừng gọn trước giới hạn 12 giờ của Kaggle, phần đã chấm vẫn nằm trong --out.

    python rerank_job.py --job t14_qwen3r4b --out /kaggle/working/scores_t14_qwen3r4b.s0.jsonl --shard 0/2
"""
import argparse
import glob
import io
import json
import math
import os
import sys
import time
import zipfile

FILES = ("meta.json", "queries.jsonl", "docs.jsonl", "pairs.jsonl")
QWEN3_INSTRUCTION = ("Given a medical question written in Vietnamese, judge whether the document, "
                     "which may be in Vietnamese, English or Chinese, is relevant to the question")
QWEN3_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
                "the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
                "<|im_start|>user\n")
QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


# ------------------------------------------------------------------ đọc job
class Job:
    """Đọc job từ file zip hoặc thư mục, cùng một giao diện."""

    def __init__(self, path):
        self.path = path
        self.zip = zipfile.ZipFile(path) if path.endswith(".zip") else None
        if self.zip:
            names = self.zip.namelist()
        else:   # thư mục giải nén: tên file ở gốc, cộng <thư mục con>/meta.json một tầng
            names = os.listdir(path) + [f"{d}/meta.json" for d in os.listdir(path)
                                        if os.path.isfile(os.path.join(path, d, "meta.json"))]
        # zip có thể bọc thêm một thư mục cùng tên job
        self.prefix = ""
        if "meta.json" not in names:
            inner = [n for n in names if n.endswith("/meta.json")]
            if not inner:
                raise SystemExit(f"{path}: không thấy meta.json")
            self.prefix = inner[0][: -len("meta.json")]
        self.meta = json.loads(self.read("meta.json"))

    def has(self, name):
        if self.zip:
            return self.prefix + name in self.zip.namelist()
        return os.path.exists(os.path.join(self.path, self.prefix, name))

    def read(self, name):
        if self.zip:
            return self.zip.read(self.prefix + name).decode("utf-8")
        with open(os.path.join(self.path, self.prefix, name), encoding="utf-8") as fh:
            return fh.read()

    def jsonl(self, name):
        if self.zip:
            with self.zip.open(self.prefix + name) as fh:
                for line in io.TextIOWrapper(fh, encoding="utf-8"):
                    if line.strip():
                        yield json.loads(line)
        else:
            with open(os.path.join(self.path, self.prefix, name), encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)


def find_job(job, roots=("/kaggle/input",)):
    """Tên job -> đường dẫn. Nhận sẵn đường dẫn .zip / thư mục; còn không thì tìm dưới roots."""
    if os.path.exists(job):
        return job
    for root in roots:
        hits = (glob.glob(os.path.join(root, "**", f"{job}.zip"), recursive=True)
                + [os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", job, "meta.json"), recursive=True)])
        if hits:
            return sorted(hits, key=len)[0]
        # Kaggle giải nén nhưng đổi tên thư mục: so tên trong meta.json
        for p in glob.glob(os.path.join(root, "**", "meta.json"), recursive=True):
            try:
                if json.load(open(p, encoding="utf-8")).get("name") == job:
                    return os.path.dirname(p)
            except (OSError, ValueError):
                continue
    raise SystemExit(f"không tìm thấy job {job} dưới {roots}")


def parse_shard(text):
    k, n = (int(x) for x in text.split("/"))
    if not 0 <= k < n:
        raise SystemExit(f"--shard {text}: cần 0 <= k < n")
    return k, n


def load_work(job, shard, done, limit=0):
    """-> (queries, docs, [(qid, [doc_id])]) của shard này, bỏ câu đã chấm, đủ text."""
    k, n = shard
    queries = {int(r["id"]): r["query"] for r in job.jsonl("queries.jsonl")}
    pairs = [(int(r["id"]), [str(d) for d in r["doc_ids"]]) for r in job.jsonl("pairs.jsonl")]
    pairs = [(q, ds) for q, ds in sorted(pairs) if q % n == k and q not in done and ds]
    if limit:
        pairs = pairs[:limit]
    need = {d for _, ds in pairs for d in ds}
    docs = {}
    for r in job.jsonl("docs.jsonl"):
        d = str(r["doc_id"])
        if d in need:
            docs[d] = r["text"]
    missing = need - docs.keys()
    if missing:
        raise SystemExit(f"{len(missing)} doc trong pairs không có text trong docs.jsonl, vd {sorted(missing)[:5]}")
    return queries, docs, pairs


def read_done(out):
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(int(json.loads(line)["id"]))
                except (ValueError, KeyError):
                    continue        # dòng cuối ghi dở khi bị ngắt
    return done


# ------------------------------------------------------------------ model
def half_dtype(torch, device):
    if not device.startswith("cuda"):
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def dtype_kw(dtype):
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


class CrossEncoderScorer:
    """Giống CrossEncoderReranker của repo: sentence-transformers CrossEncoder, fp16 trên GPU,
    predict() mặc định (sigmoid với model 1 nhãn như bge-reranker-v2-m3)."""

    def __init__(self, model, max_length, device, dtype, batch_size):
        import torch
        from sentence_transformers import CrossEncoder
        dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype) or half_dtype(torch, device)
        kw = {"model_kwargs": {"torch_dtype": dt}} if dt != torch.float32 else {}
        self.model = CrossEncoder(model, device=device, max_length=max_length, **kw)
        self.batch_size = batch_size or 32
        self.dtype = str(dt)

    def score(self, pairs):
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        out = [0.0] * len(pairs)
        for i, s in zip(order, self.model.predict([pairs[i] for i in order], batch_size=self.batch_size,
                                                   show_progress_bar=False)):
            out[i] = float(s)
        return out


class Qwen3Scorer:
    """Giống Qwen3Reranker của repo (template model card): điểm = P(yes) / (P(yes) + P(no))."""

    def __init__(self, model, max_length, device, dtype, batch_size, instruction=QWEN3_INSTRUCTION):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype) or half_dtype(torch, device)
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model, **dtype_kw(dt)).to(device).eval()
        self.device, self.max_length, self.instruction = device, max_length, instruction
        self.batch_size = batch_size or 8
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.prefix = self.tok.encode(QWEN3_PREFIX, add_special_tokens=False)
        self.suffix = self.tok.encode(QWEN3_SUFFIX, add_special_tokens=False)
        self.dtype = str(dt)

    def encode(self, query, doc):
        ids = self.tok.encode(f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}",
                              add_special_tokens=False)[: self.max_length]
        return self.prefix + ids + self.suffix

    def score(self, pairs):
        torch = self.torch
        ids = [self.encode(q, d) for q, d in pairs]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        out = [float("nan")] * len(ids)
        with torch.no_grad():
            for b in range(0, len(order), self.batch_size):
                idx = order[b: b + self.batch_size]
                batch = self.tok.pad({"input_ids": [ids[i] for i in idx]}, padding=True,
                                     return_tensors="pt").to(self.device)
                last = self.model(**batch).logits[:, -1, :]
                two = torch.stack([last[:, self.no], last[:, self.yes]], dim=1).float()
                for i, p in zip(idx, torch.softmax(two, dim=1)[:, 1].tolist()):
                    out[i] = float(p)
        return out


def make_scorer(meta, args, device):
    kind = args.kind or meta.get("kind") or ("qwen3" if "qwen3-reranker" in (args.model or meta["model"]).lower()
                                              else "cross-encoder")
    model, max_length = args.model or meta["model"], args.max_length or meta.get("max_length", 1024)
    cls = Qwen3Scorer if kind == "qwen3" else CrossEncoderScorer
    return cls(model, max_length, device, args.dtype, args.batch_size), kind, model


# ------------------------------------------------------------------ chạy
def check_scale(job, scorer, tol):
    """Chấm lại các cặp local đã chấm: điểm Kaggle phải cùng thang mới gộp với cache được."""
    if not job.has("check.jsonl"):
        return None
    rows = list(job.jsonl("check.jsonl"))
    queries = {int(r["id"]): r["query"] for r in job.jsonl("queries.jsonl")}
    need = {str(r["doc_id"]) for r in rows}
    docs = {str(r["doc_id"]): r["text"] for r in job.jsonl("docs.jsonl") if str(r["doc_id"]) in need}
    rows = [r for r in rows if str(r["doc_id"]) in docs and int(r["id"]) in queries]
    got = scorer.score([(queries[int(r["id"])], docs[str(r["doc_id"])]) for r in rows])
    diffs = [abs(g - r["expected"]) for g, r in zip(got, rows)]
    worst = max(diffs) if diffs else 0.0
    print(f"[check] {len(rows)} cặp đã chấm ở local: lệch lớn nhất {worst:.5f}, trung bình "
          f"{sum(diffs) / max(len(diffs), 1):.5f} (ngưỡng {tol})", flush=True)
    if worst > tol:
        raise SystemExit("[check] điểm Kaggle KHÁC thang điểm local: không gộp được với cache. "
                         "Kiểm phiên bản sentence-transformers / model, hoặc chạy với --check-tol lớn hơn nếu chấp nhận.")
    return worst


def run(args):
    job = Job(find_job(args.job))
    meta = job.meta
    done = read_done(args.out)
    for path in glob.glob(args.skip_done, recursive=True) if args.skip_done else []:
        if os.path.abspath(path) != os.path.abspath(args.out):
            done |= read_done(path)
    shard = parse_shard(args.shard)
    queries, docs, work = load_work(job, shard, done, args.limit_queries)
    n_pairs = sum(len(ds) for _, ds in work)
    print(f"[job] {meta.get('name')} | shard {args.shard} | đã có {len(done)} câu | còn {len(work)} câu, "
          f"{n_pairs:,} cặp -> {args.out}", flush=True)
    if not work:
        return 0
    scorer, kind, model = make_scorer(meta, args, args.device)
    print(f"[job] {kind} {model} | {scorer.dtype} | max_length {args.max_length or meta.get('max_length', 1024)} "
          f"| {args.device}", flush=True)
    if not done and shard[0] == 0:
        check_scale(job, scorer, args.check_tol)
    t0, scored, nan = time.time(), 0, 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fh:
        for i, (q, ds) in enumerate(work, 1):
            scores = scorer.score([(queries[q], docs[d]) for d in ds])
            row = {}
            for d, s in zip(ds, scores):
                if math.isnan(s):
                    nan += 1
                    row[d] = None
                else:
                    row[d] = round(s, 6)
            fh.write(json.dumps({"id": q, "scores": row}) + "\n")
            fh.flush()
            scored += len(ds)
            el = time.time() - t0
            if i % 10 == 0 or i == len(work):
                rate = scored / max(el, 1e-9)
                print(f"[job] {i}/{len(work)} câu | {rate:.1f} cặp/s | còn ~{(n_pairs - scored) / max(rate, 1e-9) / 3600:.2f} giờ"
                      + (f" | {nan} điểm NaN" if nan else ""), flush=True)
            if args.max_hours and el > args.max_hours * 3600:
                print(f"[job] dừng sau {args.max_hours} giờ, đã chấm {i}/{len(work)} câu; chạy lại cùng --out để làm tiếp",
                      flush=True)
                break
    if nan:
        print(f"[job] CẢNH BÁO: {nan} điểm NaN (thường do fp16 tràn số). Chạy lại các câu đó với --dtype fp32.",
              flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="tên job, file .zip, hoặc thư mục đã giải nén")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1", help="k/n: chấm các câu có id %% n == k")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default=None, help="ghi đè model trong meta.json")
    ap.add_argument("--kind", choices=["cross-encoder", "qwen3"], default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    ap.add_argument("--limit-queries", type=int, default=0, help="chỉ N câu đầu của shard, để đo tốc độ")
    ap.add_argument("--skip-done", default=None, help="glob các file điểm cũ (lượt chạy trước): câu có trong đó bỏ qua")
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--check-tol", type=float, default=0.02)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
