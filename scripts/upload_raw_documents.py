"""Tải data/raw_documents lên một dataset repo trên Hugging Face Hub.

Dùng upload_folder() - KHÔNG dùng upload_large_folder(), hàm đó đã bị
huggingface_hub đánh dấu deprecated (xem cảnh báo in ra nếu gọi nó). Lý do
đổi: kể từ khi thư viện có gói hf_xet (đã cài trong venv này, kiểm bằng
`is_xet_available()`), upload_folder() tự chạy theo đường streamed - vừa
kiểm tra vừa tải, tự chia nhiều commit khi folder lớn, và resumable: tắt
giữa chừng rồi gọi lại chính lệnh này sẽ tiếp tục từ chỗ dở dang, không tải
lại từ đầu. Không cần tham số num_workers nữa - hf_xet tự quản lý việc đó,
không còn kiểu "mỗi worker tự gửi commit" từng gây dồn 429 Too Many Requests
như ở upload_large_folder.

KHÔNG dán token vào đây hay vào dòng lệnh - nó sẽ nằm trong lịch sử shell và
dễ lọt vào log. Đặt HF_TOKEN bằng một trong hai cách (file .env đã có sẵn
trong .gitignore của repo, không lọt vào git):

    .env       :  HF_TOKEN=hf_...   (đúng pattern src/ingestion/__init__.py)
    PowerShell :  $env:HF_TOKEN = "hf_..."
    Git Bash   :  export HF_TOKEN="hf_..."

rồi chạy:

    python scripts/upload_raw_documents.py --repo-id <ten-ban>/vibiomir-raw-documents

Mặc định tạo repo private.
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

# huggingface_hub không tự bật handler cho logger của nó. Không có dòng này,
# một số thông báo tiến trình bị nuốt im lặng và màn hình trông như đứng yên
# dù vẫn đang chạy.
logging.basicConfig(level=logging.INFO, format="%(message)s")

ROOT = Path(__file__).parent.parent
DEFAULT_FOLDER = ROOT / "data" / "raw_documents"


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="vd: <ten-ban>/vibiomir-raw-documents")
    p.add_argument("--folder", default=str(DEFAULT_FOLDER))
    p.add_argument("--public", action="store_true",
                   help="tạo repo công khai (mặc định private)")
    p.add_argument("--max-retries", type=int, default=20,
                   help="số lần tự chạy lại nếu dính lỗi mạng/rate-limit. "
                        "An toàn vì tiến độ lưu local, chạy lại chỉ tiếp tục.")
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Thiếu biến môi trường HF_TOKEN.", file=sys.stderr)
        print("  PowerShell : $env:HF_TOKEN = \"hf_...\"", file=sys.stderr)
        print("  Git Bash   : export HF_TOKEN=\"hf_...\"", file=sys.stderr)
        return 1

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Không thấy {folder}", file=sys.stderr)
        return 1

    print(f"Nguồn: {folder}")
    print(f"-> repo: {args.repo_id} (repo_type=dataset, private={not args.public})")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id, repo_type="dataset",
        private=not args.public, exist_ok=True,
    )

    # Resumable qua hf_xet: nếu bị ngắt giữa chừng (Ctrl+C, mất mạng,
    # rate-limit), chạy lại đúng lệnh này sẽ tự bỏ qua phần đã commit và dữ
    # liệu đã tải lên, không tải lại từ đầu. Retry ở đây vì vậy chỉ đơn giản
    # là gọi lại chính hàm này.
    attempt = 0
    while True:
        attempt += 1
        try:
            api.upload_folder(
                repo_id=args.repo_id,
                repo_type="dataset",
                folder_path=str(folder),
            )
            break
        except Exception as e:
            if attempt >= args.max_retries:
                print(f"Hết {args.max_retries} lần thử, dừng lại. Lỗi cuối: {e}",
                      file=sys.stderr)
                print("Chạy lại đúng lệnh này để tiếp tục - tiến độ đã lưu local.",
                      file=sys.stderr)
                return 1
            wait_s = 65  # qua khung 60s của rate-limit trước khi thử lại
            print(f"[lần {attempt}/{args.max_retries}] Lỗi: {e}")
            print(f"Đợi {wait_s}s rồi tự chạy lại (tiến độ không mất)...")
            time.sleep(wait_s)

    print("Xong. Trên server, tải lại bằng:")
    print(f"  huggingface-cli download {args.repo_id} --repo-type dataset "
          f"--local-dir data/raw_documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
