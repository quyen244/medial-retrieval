#!/usr/bin/env bash
# Xoá collection "medical_documents" cũ (bản nhỏ, 4105 point) trên Qdrant đang
# chạy ở localhost:6333, rồi restore bản đầy đủ (3,927,800 point) từ bundle
# backup. Chạy từ thư mục gốc repo.
#
# LƯU Ý: bundle ở data/medical_documents-20261022-011147 (không có "backups/")
# THIẾU file medical_documents.snapshot (27GB) - chỉ có ledger.sqlite +
# manifest.json. Bundle ĐẦY ĐỦ nằm ở data/backups/medical_documents-20261022-011147.
set -euo pipefail
cd "$(dirname "$0")/.."

URL=http://localhost:6333
BUNDLE=data/backups/medical_documents-20261022-011147

echo "== Qdrant hiện có =="
curl -s "$URL/collections" | python -m json.tool

echo
echo "== Collection medical_documents TRƯỚC khi xoá =="
curl -s "$URL/collections/medical_documents" | python -c \
  "import json,sys; r=json.load(sys.stdin).get('result'); print('points_count:', r.get('points_count') if r else '(không tồn tại)')"

read -p "Xoá collection 'medical_documents' hiện tại và restore từ $BUNDLE? (gõ 'yes' để tiếp tục) " confirm
if [ "$confirm" != "yes" ]; then
  echo "Huỷ."
  exit 1
fi

echo
echo "== Xoá collection cũ =="
curl -s -X DELETE "$URL/collections/medical_documents" | python -m json.tool

echo
echo "== Restore bundle đầy đủ ($BUNDLE) =="
source venv/bin/activate 2>/dev/null || true
python -m scripts.qdrant_backup --url "$URL" restore --bundle "$BUNDLE"

echo
echo "== Xong. Collection sau restore =="
curl -s "$URL/collections/medical_documents" | python -c \
  "import json,sys; r=json.load(sys.stdin)['result']; print('points_count:', r.get('points_count')); print('vectors:', r['config']['params'].get('vectors')); print('sparse_vectors:', r['config']['params'].get('sparse_vectors'))"
