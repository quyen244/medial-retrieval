"""Nén data/raw_documents thành một .tar.gz mỗi domain, trước khi upload.

LÝ DO: upload từng file một (dù qua upload_large_folder hay upload_folder)
với ~1 triệu file nhỏ (~3,6 KB/file trung bình) bị kẹt - đo thực tế thấy
20+ phút không commit thêm file nào. Đây đúng là điều Hugging Face đã cảnh
báo ngay từ dòng log đầu ("exceeds the recommended limit... Consider using
fewer, larger files"), không phải lỗi script. Gộp theo domain biến ~1 triệu
file thành vài chục file lớn, đúng hình dạng mà HF khuyến nghị.

Mức nén dùng compresslevel=1, KHÔNG phải mặc định (6) hay 9: đo thực tế trên
domain bachmai.gov.vn (2.772 file) - mức 6 mất 26,8s, mức 1 chỉ 1,4s, trong
khi dung lượng chỉ lớn hơn ~25% (6,0 MB so với 4,8 MB). Thời gian đóng gói ở
đây đổi lấy dung lượng là đổi không đáng - 1 triệu file ở mức 6 ngoại suy
~2,7 giờ chỉ để nén, ở mức 1 chỉ ~8-9 phút.

Giải nén lại trên server (khớp đúng cấu trúc mà src/rag/ingest_documents.py
mong đợi: data/raw_documents/{domain}/{doc_id}.json):

    mkdir -p data/raw_documents
    for f in data/raw_documents_packed/*.tar.gz; do
        tar xzf "$f" -C data/raw_documents/
    done
"""
import argparse
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_SRC = ROOT / "data" / "raw_documents"
DEFAULT_OUT = ROOT / "data" / "raw_documents_packed"


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(DEFAULT_SRC))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--compresslevel", type=int, default=1,
                   help="1 = nhanh nhất còn nén được đáng kể (đo thực tế, xem docstring)")
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    domains = sorted(d for d in src.iterdir() if d.is_dir())
    print(f"{len(domains)} domain cần nén -> {out}")

    t_total = time.time()
    for i, domain_dir in enumerate(domains, 1):
        dest = out / f"{domain_dir.name}.tar.gz"
        if dest.exists():
            print(f"[{i}/{len(domains)}] {domain_dir.name}: đã có, bỏ qua")
            continue

        n_files = sum(1 for _ in domain_dir.rglob("*") if _.is_file())
        t0 = time.time()
        # Nén ra file tạm rồi rename: lỡ bị ngắt giữa chừng thì không để lại
        # một .tar.gz dở dang bị tưởng là đã xong ở lần chạy sau.
        tmp = dest.with_suffix(".tar.gz.part")
        with tarfile.open(tmp, "w:gz", compresslevel=args.compresslevel) as tar:
            tar.add(domain_dir, arcname=domain_dir.name)
        tmp.rename(dest)

        elapsed = time.time() - t0
        size_mb = dest.stat().st_size / 1_000_000
        print(f"[{i}/{len(domains)}] {domain_dir.name}: {n_files} file -> "
              f"{size_mb:.1f} MB trong {elapsed:.1f}s")

    print(f"Xong toàn bộ trong {time.time() - t_total:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
