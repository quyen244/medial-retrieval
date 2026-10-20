"""Chạy pha augment câu truy vấn, có mlflow ghi lại thí nghiệm.

    python -m src.rag.augment.run_augment --queries data/queries/dev.jsonl
    python -m src.rag.augment.run_augment --limit 20

Input: file jsonl mỗi dòng một query, cần hai khoá `id` và `query`.
Output: <out_dir>/augmented.jsonl, mỗi dòng một AugmentedQuery.

Chạy TÁCH khỏi pipeline retrieval là có chủ ý: mỗi query tốn 4-7 lời gọi LLM,
nhét vào pipeline thì đổi một tham số retrieval là trả lại toàn bộ chi phí đó.
Augment một lần, mọi thí nghiệm retrieval sau đọc lại cùng một file.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import mlflow

from src.rag.augment.config import AugmentConfig
from src.rag.augment.lookup import load_tables
from src.rag.augment.steps import build, make_chat
from src.rag.labeling import prompts as prompt_store


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", default=None, help="jsonl có khoá id và query")
    p.add_argument("--limit", type=int, default=None, help="chỉ chạy N query đầu")
    p.add_argument("--llm-url", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-name", default=None)
    return p


def apply_overrides(cfg: AugmentConfig, args) -> AugmentConfig:
    for attr, value in (
        ("queries_file", args.queries),
        ("llm_url", args.llm_url),
        ("llm_model", args.llm_model),
        ("concurrency", args.concurrency),
        ("out_dir", args.out_dir),
        ("run_name", args.run_name),
    ):
        if value is not None:
            setattr(cfg, attr, value)
    return cfg


def read_queries(path: str, limit: int = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append({"id": row["id"], "query": row["query"]})
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def aggregate(results) -> Dict[str, Any]:
    """Số liệu của cả run. Chỉ các đại lượng đo được từ kết quả thật."""
    n = len(results)
    if not n:
        return {"n_query": 0}
    n_long = sum(1 for a in results if a.is_long)
    lens = sorted(a.stats.get("len_max", 0) for a in results)
    n_dis = sum(a.stats.get("n_disease", 0) for a in results)
    n_med = sum(a.stats.get("n_medication", 0) for a in results)
    return {
        "n_query": n,
        "n_query_dai": n_long,
        "ty_le_dai": round(n_long / n, 4),
        "n_unit_tong": sum(len(a.units) for a in results),
        "n_disease": n_dis,
        "n_medication": n_med,
        "n_symptom": sum(a.stats.get("n_symptom", 0) for a in results),
        # Bảng chuẩn phủ được bao nhiêu phần: phần còn lại là model tự dịch,
        # tức phần có rủi ro sai thuật ngữ cao hơn.
        "ty_le_disease_tu_icd": round(
            sum(a.stats.get("n_disease_from_icd", 0) for a in results) / n_dis, 4
        ) if n_dis else 0.0,
        "ty_le_medication_tu_rxnorm": round(
            sum(a.stats.get("n_medication_from_rxnorm", 0) for a in results) / n_med, 4
        ) if n_med else 0.0,
        # Phân rã làm rụng thực thể: chỉ đo, không tự sửa.
        "n_entity_rung": sum(a.stats.get("n_entity_missing", 0) for a in results),
        "n_query_co_loi": sum(1 for a in results if a.failures),
        # Độ dài chuỗi truy vấn, tính bằng KÝ TỰ. Encoder cắt ở 1024 TOKEN và
        # cắt im lặng, nên cần số thật ở đây để biết có chạm trần hay không.
        "len_p50": lens[len(lens) // 2],
        "len_p95": lens[min(len(lens) - 1, int(len(lens) * 0.95))],
        "len_max": lens[-1],
    }


def main(argv=None) -> int:
    # Console Windows mặc định cp1252: một dòng print tiếng Việt là đủ ném
    # UnicodeEncodeError và giết cả run.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = build_argparser().parse_args(argv)
    cfg = apply_overrides(AugmentConfig(), args)

    queries = read_queries(cfg.queries_file, args.limit)
    print(f"Đọc {len(queries)} query từ {cfg.queries_file}")

    prompts = prompt_store.sync_all(prompt_store.AUGMENT_PROMPTS)
    icd, rx = load_tables(cfg)
    print(f"ICD: {len(icd.vi2en)} cụm tra được, {icd.n_ambiguous} cụm bị loại vì mơ hồ")
    print(f"RxNorm: {len(rx.name2canonical)} dạng viết")

    chat = make_chat(cfg)

    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_name=cfg.run_name or None):
        mlflow.log_params(cfg.to_dict())
        for name, version in prompts.items():
            mlflow.log_param(f"prompt_{name}_uri", getattr(version, "uri", str(version)))

        def one(row):
            return build(chat, prompts, cfg, int(row["id"]), row["query"], icd, rx)

        with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
            results = list(ex.map(one, queries))

        # Giữ đúng thứ tự file đầu vào: ex.map đã bảo toàn thứ tự, nhưng sắp
        # lại theo qid cho chắc để hai lần chạy ra file giống nhau từng dòng.
        results.sort(key=lambda a: a.qid)

        out_dir = Path(cfg.out_dir)
        write_jsonl([a.to_dict() for a in results], out_dir / "augmented.jsonl")

        metrics = aggregate(results)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        mlflow.log_artifact(str(out_dir / "augmented.jsonl"))

    print(f"-> {out_dir / 'augmented.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
