"""Fine-tune reranker đa ngữ cho ViBioMIR.

Backbone: BAAI/bge-reranker-v2-m3 (XLM-RoBERTa, vocab 250.002, đa ngữ).

Vì sao KHÔNG dùng cross-encoder/ms-marco-MiniLM-L-6-v2 như trước:
    nó là BertForSequenceClassification với vocab 30.522, tức WordPiece tiếng Anh
    thuần. Query của cuộc thi 100% tiếng Việt còn tài liệu đa số tiếng Anh, nên
    cặp vi->en không thể hoạt động với backbone đó dù dữ liệu train tốt đến đâu.
    (Đã xác minh bằng config.json của cả hai model.)

TODO (việc 8 trong TODO.md): đổi BinaryCrossEntropyLoss sang listwise KL khi đã
có dữ liệu nhóm từ bước sinh nhãn. BCE chấm từng cặp độc lập nên không học được
thứ tự tương đối, trong khi F2 cần nhiều positive mỗi query và cần điểm số hiệu
chỉnh được để đặt ngưỡng cutoff. Chưa đổi lúc này vì train.jsonl hiện còn ở dạng
cặp phẳng, chưa có nhóm.
"""
import os

import torch
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from datasets import load_dataset

MODEL_ID = "BAAI/bge-reranker-v2-m3"

# 1024 chứ không phải 512 mặc định: 12% query thật dài hơn 60 từ (dài nhất 281),
# và câu hỏi thật sự nằm ở CUỐI đoạn bệnh nhân kể bệnh. Cắt đuôi ở 512 token sẽ
# cắt mất đúng câu hỏi. Model hỗ trợ tới 8194 nên 1024 vẫn an toàn.
MAX_LENGTH = 1024

# Dữ liệu thật do `python -m src.rag.labeling.build_train_set` sinh ra. Chưa có
# thì rơi về 2 dòng mẫu trong thư mục này, chỉ đủ để chạy thử luồng.
TRAIN_FILE = "data/labeling/train.jsonl"
VAL_FILE = "data/labeling/val.jsonl"
if not os.path.exists(TRAIN_FILE):
    print(f"[!] Không thấy {TRAIN_FILE}, dùng dữ liệu mẫu sft-reranker/train.jsonl")
    TRAIN_FILE, VAL_FILE = "sft-reranker/train.jsonl", None
HAS_VAL = bool(VAL_FILE) and os.path.exists(VAL_FILE) and os.path.getsize(VAL_FILE) > 0

data_files = {"train": TRAIN_FILE}
if HAS_VAL:
    data_files["validation"] = VAL_FILE
dataset = load_dataset("json", data_files=data_files)
print(f"[data] train {len(dataset['train'])} cặp" + (f", val {len(dataset['validation'])} cặp" if HAS_VAL else ", không có val"))


model = CrossEncoder(
     MODEL_ID,
     num_labels=1,
     max_length=MAX_LENGTH,
     cache_folder='models',
)

def train():
    loss = BinaryCrossEntropyLoss(model)

    args = CrossEncoderTrainingArguments(
        output_dir="./outputs/bge-reranker-ft",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=2e-5,
        # bf16 chứ không phải fp16: XLM-R large hay tràn số khi train ở fp16.
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        # 568M tham số ở seq 1024 không vừa thoải mái trên 16GB nếu không bật.
        gradient_checkpointing=True,
        # save_strategy="best" cần eval; không có val thì lưu theo epoch.
        eval_strategy="epoch" if HAS_VAL else "no",
        save_strategy="best" if HAS_VAL else "epoch",
        metric_for_best_model="eval_loss" if HAS_VAL else None,
        greater_is_better=False,
        report_to="none",
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if HAS_VAL else None,
        loss=loss,
    )

    trainer.train()
    trainer.save_model("./models/medical-reranker")

def inference():
    """Smoke test đúng luồng truy xuất chính của cuộc thi: query tiếng Việt,
    tài liệu tiếng Anh. Backbone cũ không làm được việc này."""
    query = "Metformin có tác dụng phụ nào thường gặp?"
    documents = [
        "Common side effects include nausea and diarrhea.",
        "Metformin treats type 2 diabetes.",
        "Hypertension is high blood pressure.",
    ]

    scores = model.predict([(query, doc) for doc in documents])

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    for doc, score in ranked:
        print(f"{float(score):.4f} | {doc}")

def main():
    print("Before training")

    inference()

    print("After training")

    train()

    inference()

if __name__ == "__main__":
    main()
