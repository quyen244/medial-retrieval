"""So vLLM với HF (kaggle/embed_job.py) trên doc đã có vector: tốc độ + cosine từng doc.

Đưa vào vLLM ĐÚNG chuỗi token HF đã dùng (cắt max_tokens-1 + <|endoftext|>), nên cosine chỉ đo
sai khác do engine. Chỉ trộn shard vLLM với shard HF khi cosine min đủ cao (vd. >= 0.999).
Chạy bằng venv có vLLM, GPU phải rảnh:

    .venv-vllm/bin/python -m scripts.bench_vllm_embed --shard 24 --n 500
"""
import argparse
import io
import json
import time
import zipfile

import numpy as np

MODEL = "Qwen/Qwen3-Embedding-4B"


def load_docs(job, shard, shard_size, n):
    lo = shard * shard_size
    with zipfile.ZipFile(job) as z, io.TextIOWrapper(z.open("docs.jsonl"), encoding="utf-8") as f:
        rows = []
        for i, line in enumerate(f):
            if i >= lo + n:
                break
            if i >= lo:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="data/kaggle/embed_dataset/emb_full_q3e4b.zip")
    ap.add_argument("--ref", default="data/kaggle/out/emb_full_q3e4b_p0of8/q3e4b", help="thư mục docs_*.npy của HF")
    ap.add_argument("--shard", type=int, default=24)
    ap.add_argument("--shard-size", type=int, default=4000)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.inputs import TokensPrompt

    rows = load_docs(args.job, args.shard, args.shard_size, args.n)
    ref_ids = np.load(f"{args.ref}/docs_{args.shard:05d}.ids.npy")[: len(rows)]
    ref = np.load(f"{args.ref}/docs_{args.shard:05d}.npy")[: len(rows)].astype(np.float32)
    assert [r["doc_id"] for r in rows] == ref_ids.tolist(), "thứ tự doc lệch với vector HF"

    # Cùng cách cắt với Encoder.ids trong kaggle/embed_job.py
    tok = AutoTokenizer.from_pretrained(MODEL)
    eod = tok.convert_tokens_to_ids("<|endoftext|>")
    ids = []
    for x in tok([r["text"] for r in rows], add_special_tokens=True)["input_ids"]:
        if x and x[-1] == eod:
            x = x[:-1]
        ids.append(x[: args.max_tokens - 1] + [eod])
    n_tok = sum(map(len, ids))
    print(f"{len(ids)} doc shard {args.shard}, TB {n_tok / len(ids):.0f} token/doc", flush=True)

    t = time.time()
    try:
        llm = LLM(model=MODEL, runner="pooling", dtype=args.dtype, max_model_len=args.max_tokens,
                  gpu_memory_utilization=args.gpu_mem)
    except TypeError:   # vLLM cũ
        llm = LLM(model=MODEL, task="embed", dtype=args.dtype, max_model_len=args.max_tokens,
                  gpu_memory_utilization=args.gpu_mem)
    print(f"nạp model {time.time() - t:.0f}s", flush=True)

    llm.embed([TokensPrompt(prompt_token_ids=ids[0])])   # khởi động, không tính giờ
    t = time.time()
    outs = llm.embed([TokensPrompt(prompt_token_ids=x) for x in ids])
    dt = time.time() - t
    v = np.array([o.outputs.embedding for o in outs], dtype=np.float32)

    cos = (v * ref).sum(1) / (np.linalg.norm(v, axis=1) * np.linalg.norm(ref, axis=1))
    print(f"vLLM: {dt:.0f}s, {len(ids) / dt:.1f} doc/s, {n_tok / dt:,.0f} token/s")
    print(f"cosine vLLM vs HF: min {cos.min():.5f}, p1 {np.percentile(cos, 1):.5f}, mean {cos.mean():.5f}")


if __name__ == "__main__":
    main()
