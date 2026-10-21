"""Chạy bước 4-5 của pipeline sinh nhãn, có mlflow ghi lại thí nghiệm.

    python -m src.rag.labeling.run_judge --pilot
    python -m src.rag.labeling.run_judge

Input: data/labeling/gen_queries_kept.jsonl (bước 3) và source_chunks.jsonl
(bước 1, dùng để chèn chunk nguồn vào pool mà không truy Qdrant thêm).
Output: candidates.jsonl (bước 4), judged.jsonl + judged_kept.jsonl (bước 5).

Tách khỏi run_generate.py chứ không gộp vào: doc chốt experiment riêng
(`labeling_judging`), và bước 4-5 chạy lại được độc lập trên cùng một tập query
đã sinh - gộp vào nghĩa là muốn chấm lại thì phải sinh lại query.

Cổng chặn số 2 trong TODO.md là soát tay 30-50 bản ghi judged.jsonl; script ghi
mẫu đó ra file và log làm artifact.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import torch

from src.rag.embedding import BgeM3Embedder
from src.rag.labeling import prompts as prompt_store
from src.rag.labeling.candidates import fetch_candidates, lang_targets
from src.rag.labeling.judge import (
    TIER1_MODEL,
    CrossEncoderScorer,
    finalize,
    judge_tier2,
    judged_rows,
    tier1_score,
)
from src.rag.labeling.config import LabelingConfig
from src.rag.labeling.steps import write_jsonl
from src.rag.vectorstore import VectorStore


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true", help="chỉ chấm 300 query đầu")
    p.add_argument("--limit", type=int, default=None, help="giới hạn số query (ghi đè --pilot)")
    p.add_argument("--queries", default=None, help="mặc định <out_dir>/gen_queries_kept.jsonl")
    p.add_argument("--source-chunks", default=None, help="mặc định <out_dir>/source_chunks.jsonl")
    p.add_argument("--reranker-model", default=TIER1_MODEL)
    p.add_argument("--llm-url", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--qdrant-url", default=None)
    p.add_argument("--collection", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-name", default=None)
    return p


def apply_overrides(cfg: LabelingConfig, args) -> LabelingConfig:
    for attr, val in [
        ("llm_url", args.llm_url),
        ("llm_model", args.llm_model),
        ("qdrant_url", args.qdrant_url),
        ("corpus_collection", args.collection),
        ("out_dir", args.out_dir),
        ("run_name", args.run_name),
    ]:
        if val is not None:
            setattr(cfg, attr, val)
    cfg.experiment = "labeling_judging"
    return cfg


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Không thấy {path}. Chạy run_generate.py trước.")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    # Console Windows mặc định cp1252: một dòng print tiếng Việt là đủ ném
    # UnicodeEncodeError và giết cả run nhiều tiếng (cảnh báo parse hỏng của
    # bước 5b nằm trong vòng lặp nóng). Ghi đè ở entrypoint, không rải
    # try/except quanh từng print trong thư viện.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args = build_argparser().parse_args()
    cfg = apply_overrides(LabelingConfig(), args)
    out = Path(cfg.out_dir)

    # Prompt nạp TRƯỚC khi mở model hay Qdrant, cùng lý do như run_generate:
    # prompt sai thì hỏng ở giây thứ nhất, không phải sau khi nạp xong weights.
    print("[prompt] đồng bộ với mlflow registry...")
    loaded = prompt_store.sync_all(["judge_tier2"])
    for name, pv in loaded.items():
        print(f"  {name}: version {pv.version}")

    records = read_jsonl(Path(args.queries) if args.queries else out / "gen_queries_kept.jsonl")
    limit = args.limit if args.limit is not None else (cfg.n_pilot if args.pilot else None)
    if limit:
        records = records[:limit]

    src_rows = read_jsonl(Path(args.source_chunks) if args.source_chunks
                          else out / "source_chunks.jsonl")
    src_lookup = {r["chunk_id"]: r for r in src_rows}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = BgeM3Embedder("BAAI/bge-m3", device=device, use_fp16=True, max_length=1024)
    vectorstore = VectorStore(encoder, url=cfg.qdrant_url, collection_name=cfg.corpus_collection)
    scorer = CrossEncoderScorer(args.reranker_model, device=device)

    targets = lang_targets(cfg)
    print(f"[setup] {len(records)} query | hạn ngạch ứng viên {targets} "
          f"| tầng 2 {cfg.tier2_per_query}/{cfg.candidates_per_query} "
          f"= {cfg.tier2_per_query // cfg.tier2_batch_size} lời gọi/query | device={device}")

    mlflow.set_experiment(cfg.experiment)
    run_name = cfg.run_name or f"{'pilot' if args.pilot else 'full'}_judge_{int(time.time())}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg.to_dict())
        mlflow.set_tag("pilot", str(args.pilot))
        mlflow.log_param("tier1_model", args.reranker_model)
        for name, pv in loaded.items():
            mlflow.log_param(f"prompt_{name}_uri", pv.uri)
            mlflow.log_param(f"prompt_{name}_version", pv.version)

        # --- bước 4 ---
        t = time.perf_counter()
        rows, b4 = fetch_candidates(vectorstore, records, cfg, src_lookup)
        d4 = time.perf_counter() - t
        print(f"[b4] {b4['b4_n_query']} query, pool trung bình "
              f"{b4['b4_pool_trung_binh']} trong {d4:.0f}s | {b4}")
        mlflow.log_metrics({"b4_giay": d4,
                            **{k: v for k, v in b4.items() if isinstance(v, (int, float))}})
        write_jsonl(out / "candidates.jsonl", rows)

        # --- bước 5a ---
        t = time.perf_counter()
        b5a = tier1_score(scorer, rows)
        d5a = time.perf_counter() - t
        print(f"[b5a] chấm {b5a['b5a_n_cap']} cặp trong {d5a:.0f}s | {b5a}")
        mlflow.log_metrics({"b5a_giay": d5a,
                            **{k: v for k, v in b5a.items() if isinstance(v, (int, float))}})

        # --- bước 5b ---
        t = time.perf_counter()
        b5b = judge_tier2(rows, loaded["judge_tier2"], cfg)
        d5b = time.perf_counter() - t
        print(f"[b5b] {b5b.get('b5b_n_llm_call', 0)} lời gọi LLM trong {d5b:.0f}s | {b5b}")
        mlflow.log_metrics({"b5b_giay": d5b,
                            **{k: v for k, v in b5b.items() if isinstance(v, (int, float))}})

        # --- tổng kết ---
        kept, final = finalize(rows)
        all_out = judged_rows(rows)
        write_jsonl(out / "judged.jsonl", all_out)
        write_jsonl(out / "judged_kept.jsonl", judged_rows(kept))
        print(f"[cuối] giữ {len(kept)}/{len(rows)} query | {final}")
        mlflow.log_metrics({k: v for k, v in final.items() if isinstance(v, (int, float))})

        # Mẫu để soát tay - cổng chặn số 2 trong TODO.md. Mẫu KÈM nội dung chunk
        # (judged.jsonl không có) vì không đọc được văn bản thì không soát được
        # điểm nào đúng điểm nào sai.
        rng = random.Random(cfg.seed)
        sample_rows = rng.sample(rows, min(40, len(rows)))
        sample = [
            {
                "qid": r["qid"], "query": r["query"], "kept": r["kept"],
                "candidates": sorted(
                    [
                        {**{k: c[k] for k in ("chunk_id", "language", "t1_score",
                                              "t2_score", "t2_judged", "weight", "is_src")},
                         "content": c["content"][:600]}
                        for c in r["candidates"] if c["t2_judged"]
                    ],
                    key=lambda c: -c["t1_score"],
                ),
            }
            for r in sample_rows
        ]
        sample_path = out / "sample_judged_de_soat.json"
        sample_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(sample_path))
        mlflow.log_artifact(str(out / "judged_kept.jsonl"))

        print("\n" + "=" * 64)
        print(f"Việc tiếp theo: soát tay {len(sample)} bản ghi ở {sample_path}")
        print(f"  - ty_le_src_thang = {final['ty_le_src_thang']:.1%}: gần 100% nghĩa là")
        print("    trộn ẩn danh không phá được lợi thế trùng từ vựng, xem lại prompt bước 2")
        print("  - t2_positive_rate_zh gần 0 nghĩa là model có thể đang chấm thấp")
        print("    chỉ vì thấy tiếng Trung, xem lại candidate_lang_quota")
        print(f"  - b5b_json_fail = {b5b.get('b5b_json_fail', 0)}: >0 thì prompt tầng 2")
        print("    đang trả JSON không ổn định, sửa prompt trước khi chạy full")
        print("=" * 64)


if __name__ == "__main__":
    main()
