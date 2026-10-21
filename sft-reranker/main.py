"""Fine-tune reranker đa ngữ cho ViBioMIR.

    python sft-reranker/main.py --data-dir data/labeling_full
    python sft-reranker/main.py --data-dir data/labeling_full --epochs 1 --batch-size 4 --grad-accum 4
    python sft-reranker/main.py --data-dir data/labeling_full --dry-run   # chỉ đọc dữ liệu, không train

Dữ liệu: <data-dir>/train.jsonl và val.jsonl, mỗi dòng {"query", "passage", "label"},
do `python -m src.rag.labeling.build_train_set --dir <data-dir>` sinh ra.
Model lưu vào --out (mặc định ./models/medical-reranker), dùng lúc nộp bài bằng
`run_full_submission --reranker-model ./models/medical-reranker`.

Backbone: BAAI/bge-reranker-v2-m3 (XLM-RoBERTa, vocab 250.002, đa ngữ). Không dùng
cross-encoder/ms-marco-MiniLM-L-6-v2 như trước: vocab WordPiece tiếng Anh thuần,
không làm được cặp câu hỏi tiếng Việt - tài liệu tiếng Anh.

Loss: BinaryCrossEntropyLoss với nhãn mềm (1.0 / 0.4 / 0.0). TODO (việc 8 trong
TODO.md): listwise KL theo nhóm câu hỏi khi đã chốt được dữ liệu.

GPU: cần gần trọn 16 GB (bf16 + gradient checkpointing). Tắt vLLM trước khi train.
"""
import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict

MODEL_ID = "BAAI/bge-reranker-v2-m3"

# 1024 chứ không phải 512 mặc định: câu hỏi kể bệnh dài nhất 281 từ, và câu hỏi
# thật sự nằm ở CUỐI đoạn kể. Cắt ở 512 token cắt mất đúng câu hỏi. Khớp với
# reranker_max_length lúc nộp bài.
MAX_LENGTH = 1024


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def describe(name, rows):
    labels = defaultdict(int)
    for r in rows:
        labels[r["label"]] += 1
    qs = len({r["query"] for r in rows})
    print(f"[data] {name}: {len(rows)} cặp, {qs} câu hỏi, nhãn {dict(sorted(labels.items()))}")


def pairwise_accuracy(model, rows, batch_size):
    """Trong cùng một câu hỏi, tỉ lệ cặp (positive, negative) mà model chấm
    positive cao hơn. 0.5 = đoán mò. Đo nhanh trên val, không cần nhãn người."""
    scores = model.predict([(r["query"], r["passage"]) for r in rows], batch_size=batch_size,
                           show_progress_bar=False)
    by_q = defaultdict(lambda: ([], []))
    for r, s in zip(rows, scores):
        (by_q[r["query"]][0] if r["label"] > 0 else by_q[r["query"]][1]).append(float(s))
    win = total = 0
    for pos, neg in by_q.values():
        for p in pos:
            for n in neg:
                total += 1
                win += p > n
    return win / total if total else float("nan")


def smoke(model):
    """Đúng luồng chính của cuộc thi: câu hỏi tiếng Việt, tài liệu đa ngữ."""
    query = "Metformin có tác dụng phụ nào thường gặp?"
    docs = ["Common side effects include nausea and diarrhea.",
            "二甲双胍常见的副作用包括恶心和腹泻。",
            "Hypertension is high blood pressure."]
    for doc, s in sorted(zip(docs, model.predict([(query, d) for d in docs])), key=lambda x: -float(x[1])):
        print(f"  {float(s):.4f} | {doc}")


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/labeling_full")
    p.add_argument("--out", default="./models/medical-reranker")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch-size", type=int, default=4, help="cặp mỗi bước trên GPU")
    p.add_argument("--grad-accum", type=int, default=4, help="batch hiệu dụng = batch-size × grad-accum")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="chỉ đọc và thống kê dữ liệu")
    a = p.parse_args(argv)

    train_path = os.path.join(a.data_dir, "train.jsonl")
    val_path = os.path.join(a.data_dir, "val.jsonl")
    if not os.path.exists(train_path):
        print(f"Không thấy {train_path}. Chạy trước: python -m src.rag.labeling.build_train_set --dir {a.data_dir}")
        return 1
    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path) if os.path.exists(val_path) else []
    describe("train", train_rows)
    if val_rows:
        describe("val", val_rows)
    steps = math.ceil(len(train_rows) / (a.batch_size * a.grad_accum) * a.epochs)
    print(f"[plan] {a.epochs} epoch, batch hiệu dụng {a.batch_size * a.grad_accum}, ~{steps} bước tối ưu")
    if a.dry_run:
        return 0

    import torch
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    model = CrossEncoder(MODEL_ID, num_labels=1, max_length=a.max_length, cache_folder="models")

    print("\n[trước khi train]")
    smoke(model)
    base_acc = pairwise_accuracy(model, val_rows, 32) if val_rows else None
    if base_acc is not None:
        print(f"  val pairwise accuracy (bản gốc): {base_acc:.4f}")

    cols = lambda rows: Dataset.from_dict({"query": [r["query"] for r in rows],
                                           "passage": [r["passage"] for r in rows],
                                           "label": [float(r["label"]) for r in rows]})
    has_val = bool(val_rows)
    args = CrossEncoderTrainingArguments(
        output_dir="./outputs/bge-reranker-ft",
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size * 2,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        warmup_ratio=a.warmup_ratio,
        # bf16 chứ không phải fp16: XLM-R large hay tràn số khi train ở fp16.
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        # 568M tham số ở seq 1024 không vừa 16 GB nếu không bật.
        gradient_checkpointing=True,
        eval_strategy="epoch" if has_val else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss" if has_val else None,
        greater_is_better=False,
        logging_steps=20,
        seed=a.seed,
        report_to="none",
    )
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=cols(train_rows),
        eval_dataset=cols(val_rows) if has_val else None,
        loss=BinaryCrossEntropyLoss(model),
    )
    trainer.train()
    model.save(a.out)
    print(f"\n[đã lưu] {a.out}")

    print("\n[sau khi train]")
    smoke(model)
    if val_rows:
        acc = pairwise_accuracy(model, val_rows, 32)
        print(f"  val pairwise accuracy: {base_acc:.4f} (gốc) -> {acc:.4f} (fine-tune)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
