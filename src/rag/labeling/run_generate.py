"""Chạy bước 1-2-3 của pipeline sinh nhãn, có mlflow ghi lại thí nghiệm.

Pilot (việc 4 trong TODO.md):
    python -m src.rag.labeling.run_generate --pilot

Chạy full (việc 7):
    python -m src.rag.labeling.run_generate

Mỗi lần chạy sinh ra một mlflow run chứa: toàn bộ LabelingConfig, URI của đúng
version prompt đã dùng, các chỉ số của từng bước, và 30 query mẫu để soát tay.

Cổng chặn: script DỪNG nếu keep rate của round-trip dưới ngưỡng. Dưới ngưỡng
nghĩa là prompt sinh query đang có vấn đề, chạy tiếp chỉ tạo thêm nhãn rác.
"""
import argparse
import json
import random
import time
from pathlib import Path

import mlflow
import torch

from src.ingestion.utils import get_dataset
from src.rag.embedding import BgeM3Embedder
from src.rag.labeling.config import LabelingConfig
from src.rag.labeling import prompts as prompt_store
from src.rag.labeling.steps import (
    FewShotPool,
    IcdTerms,
    generate_queries,
    roundtrip_filter,
    select_source_chunks,
    word_count,
    write_jsonl,
)
from src.rag.vectorstore import VectorStore


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true", help="chạy 300 chunk thay vì toàn bộ")
    p.add_argument("--n-chunks", type=int, default=None, help="ghi đè n_source_chunks")
    p.add_argument("--llm-url", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--qdrant-url", default=None)
    p.add_argument("--collection", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-name", default=None)
    return p


def apply_overrides(cfg: LabelingConfig, args) -> LabelingConfig:
    if args.pilot:
        cfg.n_source_chunks = cfg.n_pilot
    for attr, val in [
        ("n_source_chunks", args.n_chunks),
        ("llm_url", args.llm_url),
        ("llm_model", args.llm_model),
        ("qdrant_url", args.qdrant_url),
        ("corpus_collection", args.collection),
        ("out_dir", args.out_dir),
        ("run_name", args.run_name),
    ]:
        if val is not None:
            setattr(cfg, attr, val)
    return cfg


def mode_distribution(records):
    """Phân phối độ dài thực tế của query sinh ra, để so với 66/34 mục tiêu.

    Đây là chỉ số quan trọng nhất của bước 2: khớp phân phối độ dài chính là
    lý do bước này tồn tại dưới dạng hai chế độ thay vì một prompt chung.
    """
    if not records:
        return {}
    n = len(records)
    long_ = sum(1 for r in records if r["n_words"] >= 30)
    return {
        "n_query": n,
        "ty_le_dai_thuc_te": round(long_ / n, 4),
        "do_dai_trung_vi": sorted(r["n_words"] for r in records)[n // 2],
        "do_dai_max": max(r["n_words"] for r in records),
    }


def main():
    args = build_argparser().parse_args()
    cfg = apply_overrides(LabelingConfig(), args)
    out = Path(cfg.out_dir)

    # Prompt nạp TRƯỚC khi mở model hay Qdrant: nếu chưa viết prompt thì hỏng
    # ngay ở giây thứ nhất, không phải sau khi nạp xong 2GB weights.
    print("[prompt] đồng bộ với mlflow registry...")
    # Chỉ prompt của bước 1-3. judge_tier2 cũng nằm trong REQUIRED_VARS nhưng
    # thuộc bước 5, run_judge.py tự nạp - không để một run sinh query hỏng vì
    # prompt của bước nó không dùng.
    loaded = prompt_store.sync_all(["gen_query_short", "gen_query_narrative"])
    for name, pv in loaded.items():
        print(f"  {name}: version {pv.version}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = BgeM3Embedder("BAAI/bge-m3", device=device, use_fp16=False, max_length=1024)
    vectorstore = VectorStore(encoder, url=cfg.qdrant_url, collection_name=cfg.corpus_collection)
    query_store = VectorStore(encoder, url=cfg.qdrant_url, collection_name=cfg.query_collection)

    real_queries = [r["query"] for r in get_dataset("query")["train"]]
    icd = IcdTerms(cfg.icd_table)
    few_shots = FewShotPool(real_queries, cfg)
    print(f"[setup] {len(real_queries)} query thật | ICD {len(icd.en2vi)} cặp en->vi | device={device}")

    mlflow.set_experiment(cfg.experiment)
    run_name = cfg.run_name or f"{'pilot' if args.pilot else 'full'}_{int(time.time())}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg.to_dict())
        mlflow.set_tag("pilot", str(args.pilot))
        for name, pv in loaded.items():
            mlflow.log_param(f"prompt_{name}_uri", pv.uri)
            mlflow.log_param(f"prompt_{name}_version", pv.version)

        # --- bước 1 ---
        t = time.perf_counter()
        chunks = select_source_chunks(vectorstore, real_queries, cfg)
        d1 = time.perf_counter() - t
        by_lang = {l: sum(1 for c in chunks if c["language"] == l) for l in cfg.lang_quota}
        print(f"[b1] {len(chunks)} chunk trong {d1:.0f}s | {by_lang}")
        mlflow.log_metrics({"b1_n_chunk": len(chunks), "b1_giay": d1,
                            **{f"b1_chunk_{l}": v for l, v in by_lang.items()}})
        write_jsonl(out / "source_chunks.jsonl", chunks)

        # --- bước 2 ---
        t = time.perf_counter()
        records, gen_stats = generate_queries(vectorstore, chunks, loaded, icd, few_shots, cfg)
        d2 = time.perf_counter() - t
        dist = mode_distribution(records)
        print(f"[b2] {len(records)} query trong {d2:.0f}s | {gen_stats}")
        print(f"     phân phối: {dist}  (mục tiêu tỷ lệ dài = {cfg.mode_ratio['narrative']})")
        mlflow.log_metrics({"b2_n_query": len(records), "b2_giay": d2,
                            **{f"b2_{k}": v for k, v in gen_stats.items()},
                            **{f"b2_{k}": v for k, v in dist.items() if isinstance(v, (int, float))}})
        write_jsonl(out / "gen_queries.jsonl", records)

        # --- bước 3 ---
        t = time.perf_counter()
        kept, rt_stats = roundtrip_filter(vectorstore, records, cfg)
        d3 = time.perf_counter() - t
        print(f"[b3] giữ {len(kept)}/{len(records)} = {rt_stats['keep_rate']:.1%} trong {d3:.0f}s")
        mlflow.log_metrics({"b3_n_giu": len(kept), "b3_giay": d3,
                            **{f"b3_{k}": v for k, v in rt_stats.items()}})
        write_jsonl(out / "gen_queries_kept.jsonl", kept)

        # Mẫu để soát tay - đây là cổng chặn số 1 trong TODO.md.
        sample = random.Random(cfg.seed).sample(records, min(30, len(records)))
        sample_path = out / "sample_de_soat.json"
        sample_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(sample_path))
        mlflow.log_artifact(str(out / "gen_queries_kept.jsonl"))

        keep_rate = rt_stats["keep_rate"]
        mlflow.set_tag("cong_chan_roundtrip", "DAT" if keep_rate >= cfg.roundtrip_min_keep_rate else "HONG")

        print("\n" + "=" * 64)
        if keep_rate < cfg.roundtrip_min_keep_rate:
            print(f"CỔNG CHẶN HỎNG: keep rate {keep_rate:.1%} < {cfg.roundtrip_min_keep_rate:.0%}")
            print("Prompt sinh query đang có vấn đề. Sửa prompt rồi chạy lại,")
            print("ĐỪNG chạy full với prompt này.")
            print(f"Soát 30 mẫu ở: {sample_path}")
            print("=" * 64)
            raise SystemExit(1)

        print(f"Keep rate {keep_rate:.1%} đạt ngưỡng.")
        print(f"Việc tiếp theo: soát tay 30 query ở {sample_path}")
        print("  - phân phối dài/ngắn có khớp 66/34 không?")
        print("  - có bê nguyên thuật ngữ lâm sàng từ chunk sang query không?")
        print("  - có câu nào nhắc tới tài liệu ('theo đoạn trên') không?")
        print("=" * 64)


if __name__ == "__main__":
    main()
