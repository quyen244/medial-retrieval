#!/usr/bin/env bash
# Sao lưu collection doc-level `medical_docs` thành bundle mang đi được
# (snapshot + manifest có sha256, số point), qua scripts/qdrant_backup.py.
#
#     bash scripts/backup_medical_docs.sh
#     HF_REPO=quyenpro/vibiomir-medical-docs bash scripts/backup_medical_docs.sh   # đẩy luôn lên HF
#
# Chặn backup khi ingest chưa xong: snapshot giữa chừng vẫn hợp lệ nhưng thiếu
# doc, restore ra rồi chạy retrieval sẽ âm thầm mất recall.
# Khôi phục: bash scripts/restore_medical_docs.sh
set -euo pipefail
cd "$(dirname "$0")/.."

URL=${QDRANT_URL:-http://localhost:6333}
COLL=medical_docs
OUT=data/backups

expected=$(wc -l < data/doc_texts.jsonl)
read -r points status indexed < <(curl -s "$URL/collections/$COLL" | python3 -c \
  "import json,sys; r=json.load(sys.stdin)['result']; print(r['points_count'], r['status'], r.get('indexed_vectors_count') or 0)")
echo "$COLL: $points point (doc_texts.jsonl: $expected), status=$status, indexed=$indexed"

if [ "$points" -lt "$expected" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "Ingest chưa xong ($points < $expected). Đợi xong rồi chạy lại, hoặc FORCE=1 để backup bản dở." >&2
  exit 1
fi
if [ "$status" != "green" ]; then
  echo "Collection đang $status (còn build HNSW). Backup vẫn đúng dữ liệu, nhưng restore xong sẽ phải build index lại." >&2
fi

venv/bin/python -m scripts.qdrant_backup --url "$URL" backup --collection "$COLL" --out-dir "$OUT"
bundle=$(ls -dt "$OUT/$COLL"-* | head -1)

if [ -n "${HF_REPO:-}" ]; then
  echo "== Đẩy $bundle lên HF dataset $HF_REPO (private) =="
  set -a; [ -f .env ] && . ./.env; set +a
  venv/bin/python - "$bundle" "$HF_REPO" <<'EOF'
import sys
from pathlib import Path
from huggingface_hub import HfApi
bundle, repo = Path(sys.argv[1]), sys.argv[2]
api = HfApi()
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path=str(bundle), path_in_repo=bundle.name, repo_id=repo, repo_type="dataset")
print(f"đã đẩy -> {repo}/{bundle.name}")
EOF
fi
