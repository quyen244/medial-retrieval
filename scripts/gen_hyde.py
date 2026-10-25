"""Sinh đoạn y văn giả định (HyDE) cho từng câu hỏi, offline, cache ra jsonl.

Biến thể truy xuất `hyde` của scripts/phase1_doc_retrieve.py: câu hỏi người bệnh
viết bằng giọng kể, tài liệu viết bằng giọng y văn - tìm bằng một đoạn văn giả
định kéo vector query về cùng phân phối với doc. Đoạn này là một lượt tìm RIÊNG,
gộp RRF với các biến thể khác, không ghép vào chuỗi câu gốc.

Đầu vào prompt (prompts/hyde_doc.txt), mọi thứ đều tiếng Việt:
  - query: câu ĐÃ SỬA chính tả (data/spelling/corrected_queries.jsonl)
  - retrieval_targets: cột vi của các target augment_v3, bỏ trùng
  - diseases / symptoms / medications: thực thể augment_v3 (sinh từ câu đã sửa)

    python -m scripts.gen_hyde --sample 20 --out data/hyde/smoke20.jsonl --show
    python -m scripts.gen_hyde                     # cả 1.200 câu, chạy tiếp được
"""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.augment.config import AugmentConfig
from src.rag.augment.steps import json_object, make_chat
from src.rag.labeling import prompts as prompt_store

CORRECTED = "data/spelling/corrected_queries.jsonl"
AUGMENTED = "data/augment_v3_corrected/augmented.jsonl"
PROMPT = "hyde_doc"


def targets_of(rec):
    out = []
    for u in rec.get("units", []):
        for t in u.get("targets", []):
            vi = (t.get("vi") or "").strip()
            if vi and vi not in out:
                out.append(vi)
    return out


def build_prompt(template, query, rec):
    ents = {k: [t["vi"] for t in rec.get(k, []) if t.get("vi")]
            for k in ("diseases", "symptoms", "medications")}
    dumps = lambda v: json.dumps(v, ensure_ascii=False)
    return prompt_store.fill(template, query=query, retrieval_targets=dumps(targets_of(rec)),
                             **{k: dumps(v) for k, v in ents.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/hyde/hyde_v1.jsonl")
    ap.add_argument("--corrected", default=CORRECTED)
    ap.add_argument("--augmented", default=AUGMENTED)
    ap.add_argument("--sample", type=int, default=0,
                    help="lấy N câu ngẫu nhiên, nửa ngắn nửa dài, để soát bằng mắt")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--llm-url", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--show", action="store_true", help="in từng câu kèm đoạn HyDE")
    args = ap.parse_args()

    cfg = AugmentConfig()
    if args.llm_url:
        cfg.llm_url = args.llm_url
    chat = make_chat(cfg)
    template = prompt_store.load(PROMPT)
    augmented = {int(r["qid"]): r for r in map(json.loads, open(args.augmented, encoding="utf-8"))}
    queries = [json.loads(l) for l in open(args.corrected, encoding="utf-8")]

    if args.sample:
        rnd = random.Random(args.seed)
        long_ = [q for q in queries if augmented.get(int(q["id"]), {}).get("is_long")]
        short = [q for q in queries if not augmented.get(int(q["id"]), {}).get("is_long")]
        half = args.sample // 2
        queries = rnd.sample(short, args.sample - half) + rnd.sample(long_, half)

    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8")}
    todo = [q for q in queries if q["id"] not in done]
    print(f"[hyde] {len(queries)} câu, đã có {len(done)}, còn {len(todo)} | {cfg.llm_model} @ {cfg.llm_url}")

    def one(q):
        rec = augmented.get(int(q["id"]), {})
        query = q.get("corrected_query") or q["query"]
        t = time.time()
        raw = chat(build_prompt(template, query, rec), args.max_tokens)
        obj = json_object(raw) or {}
        doc = (obj.get("hyde_document") or "").strip()
        flags = [] if doc else ["parse_fail" if not obj else "empty"]
        return {"id": q["id"], "query": query, "hyde_document": doc,
                "n_words": len(doc.split()), "is_long": bool(rec.get("is_long")),
                "flags": flags, "latency_s": round(time.time() - t, 2),
                "raw": None if obj else raw}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0, n, n_flag = time.time(), 0, 0
    with open(args.out, "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for r in ex.map(one, todo):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            n_flag += bool(r["flags"])
            if args.show:
                print(f"\n--- id={r['id']} {'DÀI' if r['is_long'] else 'ngắn'} | {r['n_words']} từ "
                      f"| {r['latency_s']}s {r['flags'] or ''}")
                print(f"Q:    {r['query']}")
                print(f"HYDE: {r['hyde_document']}")
            elif n % 50 == 0:
                print(f"  {n}/{len(todo)} | {n / (time.time() - t0):.2f} câu/s | gắn cờ {n_flag}", flush=True)

    print(f"\n[hyde] xong {n} câu trong {time.time() - t0:.0f}s | gắn cờ {n_flag} -> {args.out}")


if __name__ == "__main__":
    main()
