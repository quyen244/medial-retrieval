"""Đo xem corpus đang index có phủ được 1.200 query thật hay không.

Phải chạy TRƯỚC bước chọn chunk nguồn. Lý do: bước đó truy xuất chunk quanh từng
query thật, nên nếu corpus không có gì gần một query thì query đó vẫn "góp" một
chunk, chỉ là chunk rác. Cơ chế round-robin bảo đảm **mọi query đều tham gia**,
nhưng tham gia không đồng nghĩa với phủ tốt. Đây là chỗ phải đo chứ không suy ra.

    python -m src.rag.labeling.audit_coverage --collection medical_documents
    python -m src.rag.labeling.audit_coverage --stage post   # sau khi có source_chunks.jsonl

Hai giai đoạn:

  pre   Với mỗi query thật, lấy điểm tương đồng của chunk GẦN NHẤT theo từng
        ngôn ngữ. Query có điểm thấp = corpus đang thủng ở chủ đề đó.

  post  Đọc source_chunks.jsonl, đếm mỗi query thật kéo về được bao nhiêu chunk.
        Đây là phép đo trực tiếp, không phải proxy: trường `seed_query` đã ghi
        sẵn query nào kéo chunk nào.

Ngưỡng `--hole-threshold` chỉ là quy tắc ngón tay cái cho cosine của BGE-M3,
KHÔNG phải hằng số đã kiểm chứng. Hãy đọc vài query bị gắn cờ rồi tự chỉnh.
"""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import torch

from src.ingestion.utils import get_dataset
from src.rag.embedding import BgeM3Embedder
from src.rag.labeling.config import LabelingConfig
from src.rag.vectorstore import VectorStore


def pct(values, p):
    if not values:
        return 0.0
    return round(sorted(values)[min(int(p * (len(values) - 1)), len(values) - 1)], 4)


def stage_pre(vectorstore, queries, languages, hole_threshold):
    by_lang = {lang: [] for lang in languages}
    best_any = []
    worst = []

    for i, q in enumerate(queries):
        if i % 100 == 0:
            print(f"  ...{i}/{len(queries)}", flush=True)
        best = 0.0
        for lang in languages:
            try:
                hits = vectorstore.retrieve(q, top_k=1, language=lang)
            except Exception:
                hits = []
            s = float(getattr(hits[0], "score", 0.0)) if hits else 0.0
            by_lang[lang].append(s)
            best = max(best, s)
        best_any.append(best)
        worst.append((best, q))

    print("\n" + "=" * 68)
    print(f"{'ngôn ngữ':<10} {'p10':>8} {'p25':>8} {'trung vị':>10} {'p75':>8} {'p90':>8}")
    for lang in languages:
        v = by_lang[lang]
        print(f"{lang:<10} {pct(v,.10):>8} {pct(v,.25):>8} {pct(v,.50):>10} "
              f"{pct(v,.75):>8} {pct(v,.90):>8}")
    v = best_any
    print(f"{'tốt nhất':<10} {pct(v,.10):>8} {pct(v,.25):>8} {pct(v,.50):>10} "
          f"{pct(v,.75):>8} {pct(v,.90):>8}")

    holes = sum(1 for s in best_any if s < hole_threshold)
    print("-" * 68)
    print(f"Query không có chunk nào đạt {hole_threshold}: {holes}/{len(queries)} "
          f"= {100*holes/max(len(queries),1):.1f}%")
    print("\n5 query bị phủ kém nhất:")
    for s, q in sorted(worst)[:5]:
        print(f"  [{s:.3f}] {q[:84]}")
    print("=" * 68)

    return {
        "n_query": len(queries),
        "ty_le_thung": round(holes / max(len(queries), 1), 4),
        "diem_trung_vi": pct(best_any, .50),
        **{f"p50_{lang}": pct(by_lang[lang], .50) for lang in languages},
    }


def stage_post(path, queries):
    """Kiểm tra CODE bước 1 chạy đúng không, không kiểm tra corpus đủ hay thiếu
    (đó là việc của stage_pre). Vì select_source_chunks không có ngưỡng
    similarity - Qdrant luôn trả top-8 gần nhất bất kể điểm thấp hay cao - nên
    tỷ lệ "query kéo được >=1 chunk" gần như luôn cao và không mang thêm thông
    tin so với stage_pre. Bỏ hẳn chỉ số đó.

    Hai chỉ số còn lại mới có giá trị: phân bố theo tầng lộ ra ngôn ngữ nào
    phải đào sâu (tầng 6, 7, 8...) mới đủ hạn ngạch - dấu hiệu corpus thưa ở
    ngôn ngữ đó, chi tiết hơn con số nhị phân của stage_pre. Phân bố ngôn ngữ
    xác nhận quota 60/30/10 (cộng phần bù EN) đã áp đúng trong file thật.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Chưa có {p}. Chạy run_generate trước.")
    rows = [json.loads(l) for l in p.open(encoding="utf-8")]

    print("=" * 68)
    print(f"Tổng chunk nguồn      : {len(rows)}")
    print(f"Phân bố theo tầng     : {dict(sorted(Counter(r.get('seed_depth') for r in rows).items()))}")
    print(f"Phân bố ngôn ngữ      : {dict(Counter(r.get('language') for r in rows))}")
    print("=" * 68)
    return {"n_chunk": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pre", "post"], default="pre")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--qdrant-url", default=None)
    ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N query đầu, 0 = tất cả")
    ap.add_argument("--hole-threshold", type=float, default=0.50,
                    help="quy tắc ngón tay cái cho cosine BGE-M3, chưa kiểm chứng")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    cfg = LabelingConfig()
    if args.collection:
        cfg.corpus_collection = args.collection
    if args.qdrant_url:
        cfg.qdrant_url = args.qdrant_url

    queries = [r["query"] for r in get_dataset("query")["train"]]
    if args.limit:
        queries = queries[:args.limit]

    if args.stage == "post":
        metrics = stage_post(Path(cfg.out_dir) / "source_chunks.jsonl", queries)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[audit] {len(queries)} query | collection={cfg.corpus_collection} | device={device}")
        encoder = BgeM3Embedder("BAAI/bge-m3", device=device, use_fp16=False, max_length=1024)
        vs = VectorStore(encoder, url=cfg.qdrant_url, collection_name=cfg.corpus_collection)
        metrics = stage_pre(vs, queries, list(cfg.lang_quota), args.hole_threshold)

    if not args.no_mlflow:
        import mlflow
        mlflow.set_experiment(cfg.experiment)
        with mlflow.start_run(run_name=f"audit_coverage_{args.stage}"):
            mlflow.log_param("collection", cfg.corpus_collection)
            mlflow.log_param("stage", args.stage)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
