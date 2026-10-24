#!/usr/bin/env bash
# Giải nén zip_folder/*.zip (tải từ HF dataset quyenpro/vibiomir-raw-documents)
# vào data/raw_documents/{domain}/{doc_id}.json, rồi dựng data/raw_documents_index.json.
# Ngược lại với scripts/zip_raw_documents.py. Chạy từ thư mục gốc repo:
#     bash scripts/unzip_raw_documents.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=zip_folder
DST=data/raw_documents
mkdir -p "$DST"

n=$(ls "$SRC"/*.zip | wc -l)
i=0
for f in "$SRC"/*.zip; do
  i=$((i + 1))
  echo "[$i/$n] $(basename "$f")"
  # -n: không ghi đè file đã có, chạy lại sau khi đứt giữa chừng vẫn an toàn.
  unzip -q -n "$f" -d "$DST"
done

echo
echo "== Dựng index doc_id -> domain =="
# Gọi thẳng build_index(): `phase0_assemble --build-index` dựng xong còn đọc
# tiếp dump D_ft_top50.jsonl, máy không có dump thì báo lỗi ở cuối.
venv/bin/python -c "
from scripts.phase0_assemble import build_index
idx = build_index()
print(f'{len(idx):,} doc, {len(set(idx.values()))} domain -> data/raw_documents_index.json')
"
