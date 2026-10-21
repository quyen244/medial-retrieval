"""Nén mỗi domain trong data/raw_documents thành một .zip, gom vào zip_folder/.

Lý do gộp: upload từng file một (~1 triệu file nhỏ) bị kẹt - đúng điều HF đã
cảnh báo ("exceeds the recommended limit... Consider using fewer, larger
files"). Gộp theo domain biến ~1 triệu file thành vài chục file zip.

Giải nén lại trên server, đúng cấu trúc mà src/rag/ingest_documents.py mong
đợi (data/raw_documents/{domain}/{doc_id}.json):

    mkdir -p data/raw_documents
    for f in zip_folder/*.zip; do
        unzip -q "$f" -d data/raw_documents/
    done
"""
import argparse
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_SRC = ROOT / "data" / "raw_documents"
DEFAULT_OUT = ROOT / "zip_folder"


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(DEFAULT_SRC))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Bỏ thư mục ẩn: data/raw_documents/.cache/huggingface/upload là trạng
    # thái nội bộ do upload_large_folder() (đã bỏ, xem upload_raw_documents.py)
    # tự ghi vào bên trong thư mục dữ liệu ở các lần chạy trước - không phải
    # dữ liệu crawl, sẽ làm lẫn số liệu và lẫn cả vào gói zip nếu không lọc.
    domains = sorted(d for d in src.iterdir() if d.is_dir() and not d.name.startswith("."))
    print(f"{len(domains)} domain can nen -> {out}")

    t_total = time.time()
    for i, domain_dir in enumerate(domains, 1):
        dest = out / f"{domain_dir.name}.zip"
        if dest.exists():
            print(f"[{i}/{len(domains)}] {domain_dir.name}: da co, bo qua")
            continue

        files = [f for f in domain_dir.rglob("*") if f.is_file()]
        t0 = time.time()
        # Nen ra file tam roi rename: lo bi ngat giua chung thi khong de lai
        # mot .zip do dang bi tuong la da xong o lan chay sau.
        tmp = dest.with_suffix(".zip.part")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for f in files:
                zf.write(f, arcname=f"{domain_dir.name}/{f.relative_to(domain_dir)}")
        tmp.rename(dest)

        elapsed = time.time() - t0
        size_mb = dest.stat().st_size / 1_000_000
        print(f"[{i}/{len(domains)}] {domain_dir.name}: {len(files)} file -> "
              f"{size_mb:.1f} MB trong {elapsed:.1f}s")

    print(f"Xong toan bo trong {time.time() - t_total:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
