"""Tải reranker đã fine-tune lên một model repo trên Hugging Face Hub.

    python scripts/upload_reranker.py                                  # mặc định quyenpro/vibiomir-medical-reranker, private
    python scripts/upload_reranker.py --repo-id <ten>/<repo> --public
    python scripts/upload_reranker.py --model-dir ./models/medical-reranker --message "epoch 2, batch 16"

Token: đọc HF_TOKEN từ biến môi trường hoặc file .env (đã nằm trong .gitignore).
KHÔNG dán token vào script hay dòng lệnh.

Dùng lại sau khi tải:
    python -m src.rag.run_full_submission ... --reranker-model <ten>/<repo>
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

NEEDED = ["config.json", "model.safetensors", "modules.json", "tokenizer_config.json"]


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="./models/medical-reranker")
    p.add_argument("--repo-id", default="quyenpro/vibiomir-medical-reranker")
    p.add_argument("--public", action="store_true", help="mặc định private")
    p.add_argument("--message", default="Upload fine-tuned reranker")
    a = p.parse_args(argv)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Chưa có HF_TOKEN (biến môi trường hoặc file .env).")
        return 1
    d = Path(a.model_dir)
    missing = [f for f in NEEDED if not (d / f).exists()]
    if missing:
        print(f"{d} thiếu {missing}. Train xong chưa? (python sft-reranker/main.py ...)")
        return 1

    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    api = HfApi(token=token)
    who = api.whoami()["name"]
    print(f"Tài khoản: {who} | tải {d} ({size / 1e9:.2f} GB) lên {a.repo_id} "
          f"({'public' if a.public else 'private'})")

    api.create_repo(a.repo_id, repo_type="model", private=not a.public, exist_ok=True)
    info = api.upload_folder(folder_path=str(d), repo_id=a.repo_id, repo_type="model",
                             commit_message=a.message)
    print(f"Xong: https://huggingface.co/{a.repo_id}")
    print(f"commit: {getattr(info, 'oid', info)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
