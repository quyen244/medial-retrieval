import torch
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from datasets import load_dataset

dataset = load_dataset(
    'json',
    data_files={
        'train' : 'sft-reranker/train.jsonl'
    }
)


model = CrossEncoder(
     "cross-encoder/ms-marco-MiniLM-L-6-v2", 
     cache_folder = 'models'
)

def train():
    loss = BinaryCrossEntropyLoss(model)

    args = CrossEncoderTrainingArguments(
        output_dir="./outputs/bge-reranker-ft",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        eval_strategy="no",
        save_strategy="best",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        loss=loss,
    )

    trainer.train()
    trainer.save_model("./models/medical-reranker")

def inference():
    query = "Metformin có tác dụng phụ nào thường gặp?"
    documents = [
        "Common side effects include nausea and diarrhea.",
        "Metformin treats type 2 diabetes.",
        "Hypertension is high blood pressure.",
    ]

    scores = model.predict(
        [(query, doc) for doc in documents],
            apply_softmax=False,
    )
    
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

