"""In ra input/output của từng stage augment cho một query ngắn và một query dài.

    python -m scripts.trace_augment
    python -m scripts.trace_augment --short-id 2 --long-id 1
    python -m scripts.trace_augment --show-prompts      # in cả prompt gửi LLM

Cần vLLM đang chạy (bash scripts/serve_vllm.sh). Không ghi file, không log mlflow.

Cách hoạt động: bọc các hàm bước trong `src.rag.augment.steps` rồi gọi đúng
`steps.build`, nên trace phản ánh code thật chứ không phải bản chép lại.
"""
import argparse
import json
import sys
from typing import Any, Callable, List

from src.rag.augment import steps
from src.rag.augment.config import AugmentConfig
from src.rag.augment.lookup import load_tables
from src.rag.augment.run_augment import read_queries
from src.rag.labeling import prompts as prompt_store

RULE = "─" * 100


def short(s: Any, n: int = 400) -> str:
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False, default=lambda o: o.__dict__)
    s = s.replace("\n", "⏎ ")
    return s if len(s) <= n else s[:n] + f"… (+{len(s) - n} ký tự)"


def fmt_terms(terms) -> str:
    if not terms:
        return "      (rỗng)"
    return "\n".join(
        f"      - vi={t.vi!r}  en={t.en!r} ({t.source_en})  zh={t.zh!r}" for t in terms
    )


def fmt_targets(targets) -> str:
    if not targets:
        return "      (rỗng)"
    return "\n".join(
        f"      - [{t.aspect}] vi={t.vi!r} | en={t.en!r} | zh={t.zh!r}" for t in targets
    )


def fmt_output(name: str, out: Any) -> str:
    if name == "extract_entities":
        d, m, s, failed = out
        return f"disease={d}\n      medication={m}\n      symptom={s}\n      failed={failed}"
    if name == "resolve_diseases":
        return f"n_translate_fail={out[1]}\n{fmt_terms(out[0])}"
    if name == "resolve_medications":
        return fmt_terms(out)
    if name == "resolve_symptoms":
        return f"failed={out[1]}\n{fmt_terms(out[0])}"
    if name == "decompose":
        subs, failed = out
        return f"failed={failed}\n" + "\n".join(f"      {i}. {s}" for i, s in enumerate(subs, 1))
    if name == "retrieval_targets":
        return f"failed={out[1]}\n{fmt_targets(out[0])}"
    return short(out)


def fmt_input(name: str, args: tuple) -> str:
    # Bỏ chat/prompts/cfg/lookup khỏi phần hiển thị: chỉ giữ dữ liệu thật của bước.
    if name == "extract_entities":
        return short(args[3])
    if name == "resolve_diseases":
        return f"names={args[3]}"
    if name == "resolve_medications":
        return f"names={args[0]}"
    if name == "resolve_symptoms":
        return f"names={args[3]}"
    if name in ("decompose", "retrieval_targets"):
        return short(args[3])
    return short(args)


def install_tracing(show_prompts: bool) -> Callable[[], None]:
    """Bọc các bước trong steps. Trả về hàm khôi phục."""
    originals = {}
    stage_no = {"n": 0}

    def wrap(name: str):
        fn = getattr(steps, name)
        originals[name] = fn

        def traced(*args, **kwargs):
            stage_no["n"] += 1
            print(f"\n  [{stage_no['n']}] {name}")
            print(f"    INPUT : {fmt_input(name, args)}")
            out = fn(*args, **kwargs)
            print(f"    OUTPUT: {fmt_output(name, out)}")
            return out

        setattr(steps, name, traced)

    for name in ("extract_entities", "resolve_diseases", "resolve_medications",
                 "resolve_symptoms", "decompose", "retrieval_targets"):
        wrap(name)

    def restore():
        for name, fn in originals.items():
            setattr(steps, name, fn)

    return restore


def traced_chat(chat: steps.ChatFn, show_prompts: bool) -> steps.ChatFn:
    n = {"calls": 0}

    def inner(prompt: str, max_tokens: int):
        n["calls"] += 1
        if show_prompts:
            print(f"      ┌ LLM#{n['calls']} prompt (max_tokens={max_tokens}): {short(prompt, 1200)}")
        raw = chat(prompt, max_tokens)
        print(f"      └ LLM#{n['calls']} raw: {short(raw, 500) if raw is not None else None}")
        return raw

    inner.calls = n
    return inner


def trace_one(label: str, row, chat, prompts, cfg, icd, rx, show_prompts: bool) -> None:
    print(f"\n{RULE}\n{label}  id={row['id']}  ({steps.word_count(row['query'])} từ, "
          f"ngưỡng dài = {cfg.narrative_word_threshold})\n{RULE}")
    print(f"QUERY GỐC: {row['query']}")

    restore = install_tracing(show_prompts)
    tc = traced_chat(chat, show_prompts)
    try:
        aq = steps.build(tc, prompts, cfg, int(row["id"]), row["query"], icd, rx)
    finally:
        restore()

    print(f"\n  [compose] → {len(aq.units)} unit, nhánh {'DÀI' if aq.is_long else 'NGẮN'}, "
          f"{tc.calls['calls']} lời gọi LLM, failures={aq.failures or 'không'}")
    for i, u in enumerate(aq.units, 1):
        text = u.retrieval_text()
        print(f"\n    unit {i}/{len(aq.units)} kind={u.kind}  ({len(text)} ký tự)")
        print("    ┌" + "─" * 60)
        for line in text.split("\n"):
            print(f"    │ {line}")
        print("    └" + "─" * 60)
    print(f"\n  stats: {json.dumps(aq.stats, ensure_ascii=False)}")


def pick(rows: List[dict], long: bool, threshold: int):
    for r in rows:
        if (steps.word_count(r["query"]) > threshold) == long:
            return r
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", default="data/queries/real_queries.jsonl")
    p.add_argument("--short-id", type=int, default=None, help="id query ngắn (mặc định: query ngắn đầu tiên)")
    p.add_argument("--long-id", type=int, default=None, help="id query dài (mặc định: query dài đầu tiên)")
    p.add_argument("--show-prompts", action="store_true")
    p.add_argument("--llm-url", default=None)
    p.add_argument("--llm-model", default=None)
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cfg = AugmentConfig()
    if args.llm_url:
        cfg.llm_url = args.llm_url
    if args.llm_model:
        cfg.llm_model = args.llm_model

    rows = read_queries(args.queries)
    by_id = {int(r["id"]): r for r in rows}
    short_row = by_id.get(args.short_id) if args.short_id is not None else pick(rows, False, cfg.narrative_word_threshold)
    long_row = by_id.get(args.long_id) if args.long_id is not None else pick(rows, True, cfg.narrative_word_threshold)
    if short_row is None or long_row is None:
        print(f"Không chọn được đủ 2 query từ {args.queries} (short={short_row is not None}, long={long_row is not None})")
        return 1

    prompts = prompt_store.sync_all(prompt_store.AUGMENT_PROMPTS)
    icd, rx = load_tables(cfg)
    chat = steps.make_chat(cfg)

    trace_one("QUERY NGẮN", short_row, chat, prompts, cfg, icd, rx, args.show_prompts)
    trace_one("QUERY DÀI", long_row, chat, prompts, cfg, icd, rx, args.show_prompts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
