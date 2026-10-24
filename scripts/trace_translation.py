"""Trace một lần dịch query: input, prompt gửi LLM, output thô, kết quả đã bóc.

Mặc định chạy đúng các câu smoke 20 đang dịch SAI (docs: so sánh có/không thực thể):

    824  u nhú thực quản           -> leiomyoma / varices
    358  đi tiểu đau và dắt        -> dyspareunia
    645  mận và xoài               -> pineapple, mất câu hỏi
    563  tự nhiên ... cà nhắc      -> naturally, staggering gait
    93   nhau chưa bong hết        -> blood clot
    997  lang thận                 -> renal ptosis
    309  lệch 1 bên má             -> leaning to one side (bản không thực thể)

    python -m scripts.trace_translation                    # các câu trên, có thực thể
    python -m scripts.trace_translation --no-entities      # cùng câu, bỏ thực thể
    python -m scripts.trace_translation --ids 824,358      # chỉ vài câu
    python -m scripts.trace_translation --dry-run          # chỉ in input + prompt, không gọi LLM
    python -m scripts.trace_translation --out data/translate/trace.jsonl   # lưu lại để so sau khi sửa prompt

Cần vLLM đang chạy (trừ --dry-run). Prompt dựng bằng chính build_prompt() của
scripts/translate_queries.py, nên trace phản ánh đúng code chạy thật.
"""
import argparse
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.augment.config import AugmentConfig
from src.rag.augment.steps import json_object
from src.rag.labeling import prompts as prompt_store
from scripts.translate_queries import (AUGMENTED, ICD_EMB, PROMPT, QUERIES, IcdShots, build_prompt,
                                       check, load_embedder, retrieve_shots)

RULE = "─" * 100
WRONG_IDS = [824, 358, 645, 563, 93, 997, 309]


def call(cfg, prompt, max_tokens):
    """Giống make_chat() (nhiệt độ 0, seed, tắt thinking) nhưng trả cả JSON
    phản hồi để in được số token."""
    body = {
        "model": cfg.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg.temperature,
        "max_tokens": max_tokens,
        "seed": cfg.seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(f"{cfg.llm_url}/chat/completions", json=body, timeout=cfg.llm_timeout)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=",".join(map(str, WRONG_IDS)))
    ap.add_argument("--no-entities", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="không gọi LLM")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--llm-url", default=None)
    ap.add_argument("--out", default=None, help="ghi trace ra jsonl")
    args = ap.parse_args()

    cfg = AugmentConfig()
    if args.llm_url:
        cfg.llm_url = args.llm_url
    template = prompt_store.load(PROMPT)
    icd = IcdShots()
    if icd.emb is None:
        raise SystemExit(f"Chưa có {ICD_EMB}. Chạy: python -m scripts.translate_queries --build-icd-index")
    augmented = {int(json.loads(l)["qid"]): json.loads(l) for l in open(AUGMENTED, encoding="utf-8")}
    queries = {int(json.loads(l)["id"]): json.loads(l) for l in open(QUERIES, encoding="utf-8")}
    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    shots = retrieve_shots(icd, [queries[i] for i in ids], augmented, load_embedder())

    out = open(args.out, "a", encoding="utf-8") if args.out else None
    for qid in ids:
        q, rec = queries[qid], augmented.get(qid, {})
        prompt, ents = build_prompt(template, q["query"], rec, shots[qid], args.no_entities)

        print(f"\n{RULE}\nid={qid} | {'DÀI' if rec.get('is_long') else 'ngắn'} | "
              f"{'KHÔNG thực thể' if args.no_entities else 'có thực thể'}\n{RULE}")
        print("[INPUT] query:")
        print(f"    {q['query']}")
        print(f"[INPUT] medical_entities (tiếng Việt): {json.dumps(ents, ensure_ascii=False)}")
        print("[INPUT] few_shots - cặp ICD retrieve được (điểm cosine BGE-M3):")
        for vi, en, zh, s in shots[qid]:
            print(f"    {s:.3f}  {vi or '-'} | {en} | {zh or '-'}")
        print(f"\n[PROMPT] {len(prompt)} ký tự\n{prompt}")

        if args.dry_run:
            continue

        t0 = time.time()
        resp = call(cfg, prompt, args.max_tokens)
        el = time.time() - t0
        raw = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        obj = json_object(raw) or {}
        en, zh = (obj.get("query_en") or "").strip(), (obj.get("query_zh") or "").strip()
        flags = check(en, zh, q["query"]) if obj else ["parse_fail"]

        print(f"\n[OUTPUT thô] {el:.1f}s | prompt {usage.get('prompt_tokens')} tok, "
              f"output {usage.get('completion_tokens')} tok | finish={resp['choices'][0].get('finish_reason')}")
        print(raw)
        print("\n[KẾT QUẢ]")
        print(f"    EN: {en}")
        print(f"    ZH: {zh}")
        print(f"    cờ: {flags or 'không'}")

        if out:
            out.write(json.dumps({"id": qid, "no_entities": args.no_entities, "query": q["query"],
                                  "entities": ents, "few_shots": shots[qid], "prompt": prompt, "raw": raw,
                                  "query_en": en, "query_zh": zh, "flags": flags, "usage": usage,
                                  "latency_s": round(el, 2)}, ensure_ascii=False) + "\n")
    if out:
        out.close()


if __name__ == "__main__":
    main()
