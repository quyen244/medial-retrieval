"""Sinh nhãn train reranker cấp doc, thang 3 mức, trên 1.200 câu thật đã sửa. Logic ở doc_labels.py.

    python -m src.rag.labeling.run_doc_labels select     # 60 doc/câu + text 8k cắt còn phần reranker len 2048 thấy (không cần Qdrant)
    python -m src.rag.labeling.run_doc_labels embed      # vector bge-m3 của câu hỏi, cho few-shot (cần GPU/CPU)
    python -m src.rag.labeling.run_doc_labels agree      # LLM chấm các cặp đã gán tay, đo độ khớp (cần vLLM)
    python -m src.rag.labeling.run_doc_labels judge --limit 50
    python -m src.rag.labeling.run_doc_labels judge
    python -m src.rag.labeling.run_doc_labels build --partial-label 0.4

Trên Kaggle (2 x T4, mỗi GPU một server vLLM Qwen3-8B, embed + agree + judge trong một kernel):
    python -m src.rag.labeling.run_doc_labels kaggle --owner <kaggle-user>

Chạy `agree` và đọc báo cáo TRƯỚC khi `judge` đủ 1.200 câu: đó là cổng kiểm prompt.
`agree` và `judge` ghi file theo mã băm prompt (`agree_<sha>.jsonl`, `judged_<sha>.jsonl`):
sửa prompt là ra file mới, không trộn nhãn của hai bản prompt. Chạy lại cùng lệnh là làm tiếp.

Prompt: prompts/judge_doc_3level.txt, biến MLflow hai ngoặc {{...}}:
  {{query}}        câu hỏi đã sửa chính tả, tiếng Việt. Căn cứ cuối cùng khi gợi ý mâu thuẫn.
  {{few_shots}}    ví dụ từ các câu gán tay giống nhất (không bao giờ là chính câu đang chấm):
                   câu hỏi + vài doc cắt 500 ký tự + điểm 2/1/0 của người gán (kèm ghi chú nếu có).
  {{documents}}    mảng JSON [{"doc_id": 1, "text": "..."}], doc_id cục bộ 1..3, không có hạng/điểm/ngôn ngữ.
Đặt {{documents}} CUỐI prompt: phần trước nó giống nhau ở mọi lô của một câu, vLLM dùng lại prefix cache.
Output model phải là mảng JSON [{"doc_id": int, "score": 0|1|2}], mỗi doc đầu vào đúng một phần tử;
trường thêm (same_subject, answers...) được chấp nhận và bỏ qua.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

from src.rag.labeling import doc_labels as dl
from src.rag.labeling import prompts as prompt_store
from src.rag.labeling.config import LabelingConfig

# Pool bge-reranker-v2-m3 len 2048 trên text 8k, đủ 1.200 câu (567 chấm ở server + 633 trên Kaggle, 2026-11-03).
POOL = "data/T28_rrf_bge_q3e/w0.6_qc_vi90_bge_q3etext2048_full_pool.jsonl"
TEXTS = "data/doc_texts_8k.jsonl"
RR_TOKENIZER = "BAAI/bge-reranker-v2-m3"
# Phần doc reranker nhìn thấy ở len 2048: 2048 - 4 token đặc biệt - 114 token của câu hỏi dài nhất.
# LLM chấm và tập train đều dùng đúng phần này: nhãn không dựa vào chữ mà reranker không bao giờ thấy.
DOC_TOKENS = 1930
TEXT_SRC = f"8k_bge{DOC_TOKENS}"
CORRECTED = "data/spelling/corrected_queries.jsonl"
HUMAN = "artifact/labeling/labels_query_*.json"
OUT = "data/labeling_doc_qcorr"
PROMPT = "judge_doc_3level"


def human_labels(args):
    paths = sorted(glob.glob(args.human))
    human = dl.load_human_labels(paths)
    n = {s: sum(1 for h in human.values() for v in h["labels"].values() if v[0] == s) for s in dl.SCORES}
    print(f"[human] {len(human)} câu từ {len(paths)} file | mức 2: {n[2]}, 1: {n[1]}, 0: {n[0]}")
    return human


def load_texts(out):
    path = Path(out) / "texts.jsonl"
    if not path.exists():
        raise SystemExit(f"Không thấy {path}. Chạy `select` trước.")
    return {str(r["doc_id"]): r["text"] for r in dl.read_jsonl(path)}


def trim_texts(texts, max_tokens=DOC_TOKENS):
    """Cắt đuôi mỗi doc còn `max_tokens` token của tokenizer reranker (chỉ tokenizer, không nạp model)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(RR_TOKENIZER)
    out, cut = {}, 0
    for d, t in texts.items():
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) > max_tokens:
            t, cut = tok.decode(ids[:max_tokens]), cut + 1
        out[d] = t
    return out, cut


def cmd_select(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ranks = dl.pick_ranks(300, args.top, args.spread)
    pool = list(dl.read_jsonl(args.pool))
    rows = dl.select_candidates(pool, ranks)
    corrected = {int(r["id"]): r.get("corrected_query") or r["query"] for r in dl.read_jsonl(CORRECTED)}
    bad = [r["id"] for r in rows if corrected.get(r["id"]) != r["query"]]
    if len(rows) != 1200 or bad:
        raise SystemExit(f"pool có {len(rows)} câu, {len(bad)} câu không phải corrected_query (vd {bad[:5]})")
    with open(out / "candidates.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            dl.append_jsonl(fh, r)
    human = human_labels(args)
    # Nhãn tay gán trên trang 300 doc của cùng pool (thứ tự 1024 hay 2048 đều cùng tập doc): doc ngoài pool
    # nghĩa là file nhãn của pool khác.
    pool_docs = {int(r["id"]): {str(c["doc_id"]) for c in r["candidates"]} for r in pool}
    for qid, h in human.items():
        stray = set(h["labels"]) - pool_docs.get(qid, set())
        if stray:
            raise SystemExit(f"câu {qid}: {len(stray)} doc gán tay không có trong pool {args.pool} (vd {sorted(stray)[:3]})")
    need = {c["doc_id"] for r in rows for c in r["candidates"]} | {d for h in human.values() for d in h["labels"]}
    texts_path = out / "texts.jsonl"
    have = set()
    if texts_path.exists():
        for r in dl.read_jsonl(texts_path):
            if r.get("src") != TEXT_SRC:
                raise SystemExit(f"{texts_path} có text nguồn {r.get('src')}, khác {TEXT_SRC}: xoá file rồi chạy lại")
            have.add(str(r["doc_id"]))
    print(f"[select] {len(rows)} câu x {len(ranks)} hạng -> {out / 'candidates.jsonl'} | "
          f"{len(need):,} doc cần text, đã có {len(need & have):,}", flush=True)
    if need - have:
        from scripts.kaggle_rerank_vllm import read_texts
        got, cut = trim_texts(read_texts(args.texts, need - have))
        with open(texts_path, "a", encoding="utf-8") as fh:
            for d in sorted(got, key=int):
                dl.append_jsonl(fh, {"doc_id": d, "src": TEXT_SRC, "text": got[d]})
        print(f"[select] {len(got):,} doc từ {args.texts}, {cut:,} doc cắt còn {DOC_TOKENS} token")
    print(f"[select] text -> {texts_path}")


def cmd_embed(args):
    import numpy as np
    from src.rag.embedding import BgeM3Embedder
    rows = [(int(r["id"]), r.get("corrected_query") or r["query"]) for r in dl.read_jsonl(CORRECTED)]
    enc = BgeM3Embedder("BAAI/bge-m3", max_length=512)
    vecs = np.asarray(enc.encode([q for _, q in rows], batch_size=32), dtype=np.float32)
    vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
    np.savez(Path(args.out) / "query_vecs.npz", ids=np.array([i for i, _ in rows]), vecs=vecs)
    print(f"[embed] {len(rows)} câu -> {Path(args.out) / 'query_vecs.npz'}")


def judge_setup(args):
    """Mọi thứ `agree` và `judge` dùng chung: prompt, text, nhãn tay, biến của từng câu."""
    import numpy as np
    template = prompt_store.load(PROMPT)
    sha = dl.prompt_sha(template)
    texts = load_texts(args.out)
    human = human_labels(args)
    if not human:
        raise SystemExit(f"Không có nhãn tay nào ở {args.human}: few-shot cần ít nhất một câu.")
    vec_path = Path(args.out) / "query_vecs.npz"
    if not vec_path.exists():
        raise SystemExit(f"Không thấy {vec_path}. Chạy `embed` trước.")
    z = np.load(vec_path)
    vecs = dict(zip(z["ids"].tolist(), z["vecs"]))
    vars_of = dl.make_vars_of(human, texts, vecs, args.shots, args.per_label, args.shot_chars)

    cfg = LabelingConfig()
    cfg.llm_url = args.llm_url or cfg.llm_url
    cfg.llm_model = args.llm_model or cfg.llm_model
    meta = {"prompt": PROMPT, "prompt_sha": sha, "llm_model": cfg.llm_model}
    print(f"[judge] prompt {PROMPT} sha {sha} | {cfg.llm_model} @ {cfg.llm_url} | {args.shots} câu ví dụ x "
          f"{args.per_label} doc mỗi mức | lô {args.batch_size} doc")
    return template, sha, texts, human, vars_of, cfg, meta


def run(args, rows, doc_ids_of, out_path, setup):
    template, sha, texts, human, vars_of, cfg, meta = setup
    chat = dl.make_chat(cfg.llm_model, cfg.llm_url.split(","), cfg.judge_temperature, cfg.judge_max_tokens, cfg.seed)
    st = dl.run_judge(rows, doc_ids_of, vars_of, texts, template, dl.fill, chat, out_path,
                      meta, batch_size=args.batch_size, retries=cfg.llm_max_retries, seed=cfg.seed,
                      concurrency=args.concurrency or cfg.llm_concurrency)
    if st:
        print(f"[judge] lượt này: {dict(st)}")


def cmd_agree(args):
    setup = judge_setup(args)
    sha, human = setup[1], setup[3]
    rows = dl.agree_rows(human, dl.read_jsonl(Path(args.out) / "candidates.jsonl"))
    out_path = Path(args.out) / f"agree_{sha}.jsonl"
    run(args, rows, lambda r: [c["doc_id"] for c in r["candidates"]], out_path, setup)
    rep = dl.agreement(dl.read_jsonl(out_path), human)
    (Path(args.out) / f"agree_{sha}_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                                             encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


def cmd_judge(args):
    setup = judge_setup(args)
    rows = list(dl.read_jsonl(Path(args.out) / "candidates.jsonl"))
    if args.limit:
        rows = rows[: args.limit]
    run(args, rows, lambda r: [c["doc_id"] for c in r["candidates"]], Path(args.out) / f"judged_{setup[1]}.jsonl",
        setup)


KAGGLE_ROOT = "data/kaggle"
KAGGLE_DATASET = f"{KAGGLE_ROOT}/doclabel_dataset"
JOB_SRC = "kaggle/doc_label_job.py"
LIB_SRC = "src/rag/labeling/doc_labels.py"
KERNEL_RUN = r'''"""Kaggle kernel: nhãn doc 3 mức cho job __JOB__ bằng Qwen3-8B, mỗi GPU một server vLLM.
Sinh bởi src/rag/labeling/run_doc_labels.py (lệnh kaggle); đừng sửa tay. Code nhúng ngay trong file:
dataset chỉ chứa dữ liệu, sửa code thì sinh lại và push lại kernel."""
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
sh([sys.executable, os.path.join(code, "doc_label_job.py"), "run", "--job", JOB, "--out", W, *ARGS])
'''


def cmd_kaggle(args):
    """Zip job (dữ liệu) + hai kernel: `_smoke` (agree + 50 câu, đo tốc độ) và bản chạy đủ."""
    import zipfile
    from scripts.kaggle_jobs import ensure_dataset_meta
    out = Path(args.out)
    template = prompt_store.load(PROMPT)
    sha = dl.prompt_sha(template)
    human = human_labels(args)
    if not human:
        raise SystemExit("Chưa có nhãn tay nào: few-shot và cổng agree cần nhãn 10 câu.")
    texts = load_texts(out)
    miss = sorted({d for h in human.values() for d in h["labels"]} - texts.keys())
    if miss:
        raise SystemExit(f"{len(miss)} doc gán tay chưa có text (vd {miss[:5]}): chạy lại `select`.")
    name = args.name or f"doc_labels_{sha}"
    Path(KAGGLE_DATASET).mkdir(parents=True, exist_ok=True)
    zpath = Path(KAGGLE_DATASET) / f"{name}.zip"
    queries = [{"id": int(r["id"]), "query": r.get("corrected_query") or r["query"]} for r in dl.read_jsonl(CORRECTED)]
    with zipfile.ZipFile(str(zpath) + ".part", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("meta.json", json.dumps({"name": name, "prompt": PROMPT, "prompt_sha": sha, "pool": args.pool,
                                            "n_human": len(human)}, ensure_ascii=False, indent=2))
        z.writestr("prompt.txt", template)
        z.write(out / "candidates.jsonl", "candidates.jsonl")
        z.write(out / "texts.jsonl", "texts.jsonl")
        z.writestr("human.json", json.dumps({str(q): {"query": h["query"], "labels": h["labels"]}
                                             for q, h in human.items()}, ensure_ascii=False))
        z.writestr("queries.jsonl", "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queries))
        if (out / "query_vecs.npz").exists():
            z.write(out / "query_vecs.npz", "query_vecs.npz")
    Path(str(zpath) + ".part").replace(zpath)

    dataset_id = ensure_dataset_meta(args.owner, args.dataset_slug, KAGGLE_DATASET)
    owner = args.owner or dataset_id.split("/")[0]
    src = {"doc_label_job.py": open(JOB_SRC, encoding="utf-8").read(),
           "doc_labels.py": open(LIB_SRC, encoding="utf-8").read()}
    kernels = []
    for suffix, extra in (("_smoke", ["--steps", "embed,serve,agree,judge", "--limit", "50", "--max-hours", "3"]),
                          ("", ["--steps", "embed,serve,agree,judge"])):
        kdir = Path(KAGGLE_ROOT) / f"kernel_{name}{suffix}"
        kdir.mkdir(parents=True, exist_ok=True)
        code = (KERNEL_RUN.replace("__JOB__", name).replace("__ARGS__", json.dumps(extra + args.job_arg))
                .replace("__VLLM__", args.vllm_spec).replace("__SRC__", repr(src)))
        (kdir / "run.py").write_text(code, encoding="utf-8")
        slug = f"vibiomir-{name.replace('_', '-')}{suffix.replace('_', '-')}"
        json.dump({"id": f"{owner}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
                   "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
                   "machine_shape": "NvidiaTeslaT4", "dataset_sources": [dataset_id],
                   "competition_sources": [], "kernel_sources": [], "model_sources": []},
                  open(kdir / "kernel-metadata.json", "w", encoding="utf-8"), indent=2)
        kernels.append((os.path.normpath(kdir), f"{owner}/{slug}"))

    # kaggle CLI trên Windows hỏng với "/" trong -p, nên in theo dấu của hệ điều hành
    ds, od = os.path.normpath(KAGGLE_DATASET), os.path.normpath(out)
    print(f"[kaggle] {zpath} ({zpath.stat().st_size / 1e6:.1f} MB) | prompt sha {sha} | {len(human)} câu gán tay")
    print("Bước tiếp (máy có kaggle CLI):")
    print(f"  kaggle datasets create -p {ds}       # lần đầu; lần sau: kaggle datasets version -p {ds} -m \"{name}\"")
    for kdir, kid in kernels:
        print(f"  kaggle kernels push -p {kdir}")
        print(f"  kaggle kernels status {kid}")
        print(f"  kaggle kernels output {kid} -p {od} --file-pattern \"(agree|judged)_.*|query_vecs.*|vllm_gpu.*log\"")


def cmd_build(args):
    out = Path(args.out)
    judged = args.judged or (sorted(glob.glob(str(out / "judged_*.jsonl")), key=lambda p: Path(p).stat().st_mtime)
                             or [None])[-1]
    if not judged:
        raise SystemExit(f"Không thấy judged_*.jsonl trong {out}. Chạy `judge` trước.")
    rows = list(dl.read_jsonl(judged))
    partial = None if args.partial_label in ("drop", "none") else float(args.partial_label)
    split, st = dl.build_pairs(rows, human_labels(args), load_texts(out), partial, args.val_frac, args.seed)
    for name, items in split.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for it in items:
                dl.append_jsonl(fh, it)
    print(f"[build] từ {judged} ({len(rows)} câu) | mức 1 -> {partial}")
    print(f"[build] train {len(split['train']):,} cặp, val {len(split['val']):,} cặp | {dict(sorted(st.items()))}")
    print(f"[build] -> {out / 'train.jsonl'}, {out / 'val.jsonl'}")


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default=OUT)
    common.add_argument("--pool", default=POOL)
    common.add_argument("--human", default=HUMAN, help="glob file nhãn xuất từ trang gán nhãn")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("select", parents=[common])
    p.add_argument("--top", type=int, default=20, help="số hạng đầu chấm hết")
    p.add_argument("--spread", type=int, default=40, help="số hạng rải đều trên phần còn lại của pool")
    p.add_argument("--texts", default=TEXTS)
    p.set_defaults(fn=cmd_select)

    sub.add_parser("embed", parents=[common]).set_defaults(fn=cmd_embed)

    judge_common = argparse.ArgumentParser(add_help=False)
    judge_common.add_argument("--llm-url", default=None)
    judge_common.add_argument("--llm-model", default=None)
    judge_common.add_argument("--batch-size", type=int, default=3,
                              help="số doc mỗi lời gọi LLM (doc tới ~1.930 token: 3 doc + few-shot vừa 12k)")
    judge_common.add_argument("--concurrency", type=int, default=None)
    judge_common.add_argument("--shots", type=int, default=2, help="số câu ví dụ mỗi prompt")
    judge_common.add_argument("--per-label", type=int, default=1, help="số doc ví dụ mỗi mức 2/1/0")
    judge_common.add_argument("--shot-chars", type=int, default=500, help="cắt doc ví dụ còn bấy nhiêu ký tự")
    sub.add_parser("agree", parents=[common, judge_common]).set_defaults(fn=cmd_agree)
    p = sub.add_parser("judge", parents=[common, judge_common])
    p.add_argument("--limit", type=int, default=0, help="chỉ N câu đầu")
    p.set_defaults(fn=cmd_judge)

    p = sub.add_parser("kaggle", parents=[common])
    p.add_argument("--owner", default=None, help="kaggle username (lần đầu bắt buộc)")
    p.add_argument("--dataset-slug", default="vibiomir-doc-labels")
    p.add_argument("--name", default=None, help="mặc định doc_labels_<sha prompt>")
    p.add_argument("--vllm-spec", default="vllm", help="gói pip cài vLLM trên Kaggle")
    p.add_argument("--job-arg", action="append", default=[], help="tham số thêm cho doc_label_job.py run")
    p.set_defaults(fn=cmd_kaggle)

    p = sub.add_parser("build", parents=[common])
    p.add_argument("--judged", default=None, help="mặc định judged_*.jsonl mới nhất trong --out")
    p.add_argument("--partial-label", default="0.4", help="nhãn cho mức 1: số trong [0,1], hoặc 'drop' để bỏ")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_build)

    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
