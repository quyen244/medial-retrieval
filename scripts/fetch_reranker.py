"""Tải lại reranker đã fine-tune từ Hugging Face về ./models/medical-reranker.

Checkpoint gốc trên máy đã mất (chỉ còn sft-reranker/main.py và train.jsonl),
nhưng nó đã được đẩy lên Hub bằng scripts/upload_reranker.py. Không có nó thì
mọi so sánh với các bài nộp cũ (D_ft, N3) đều lẫn thêm biến reranker.

Token đọc từ HF_TOKEN trong môi trường hoặc file .env (.env nằm trong .gitignore).
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="quyenpro/vibiomir-medical-reranker")
    p.add_argument("--local-dir", default="models/medical-reranker")
    a = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Chưa có HF_TOKEN trong môi trường hay .env")
        return 1

    try:
        path = snapshot_download(repo_id=a.repo_id, local_dir=a.local_dir, token=token)
    except Exception as e:
        print(f"LỖI tải {a.repo_id}: {type(e).__name__}: {str(e)[:400]}")
        return 1

    files = sorted(Path(path).rglob("*"))
    total = sum(f.stat().st_size for f in files if f.is_file())
    print(f"đã tải {a.repo_id} -> {path}  ({total/1e6:.0f} MB)")
    for f in files:
        if f.is_file() and f.stat().st_size > 1000:
            print(f"   {f.relative_to(path)}  {f.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
