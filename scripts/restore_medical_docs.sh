#!/usr/bin/env bash
# Khôi phục collection `medical_docs` từ bundle do scripts/backup_medical_docs.sh tạo.
#
#     bash scripts/restore_medical_docs.sh                                 # bundle mới nhất trong data/backups
#     bash scripts/restore_medical_docs.sh data/backups/medical_docs-2026... # bundle chỉ định
#     OVERWRITE=1 bash scripts/restore_medical_docs.sh                     # ghi đè collection đang có
#     HF_REPO=quyenpro/vibiomir-medical-docs bash scripts/restore_medical_docs.sh   # tải từ HF trước
#
# qdrant_backup kiểm sha256 trước khi upload và đối chiếu số point sau khi nạp.
set -euo pipefail
cd "$(dirname "$0")/.."

URL=${QDRANT_URL:-http://localhost:6333}
OUT=data/backups

if [ -n "${HF_REPO:-}" ] && [ -z "${1:-}" ]; then
  echo "== Tải bundle mới nhất từ HF dataset $HF_REPO =="
  set -a; [ -f .env ] && . ./.env; set +a
  venv/bin/python - "$HF_REPO" "$OUT" <<'EOF'
import sys
from huggingface_hub import HfApi, snapshot_download
repo, out = sys.argv[1], sys.argv[2]
names = sorted({f.split("/")[0] for f in HfApi().list_repo_files(repo, repo_type="dataset")
                if f.startswith("medical_docs-")})
if not names:
    sys.exit(f"{repo} không có bundle medical_docs-*")
snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{names[-1]}/*"], local_dir=out)
print(f"đã tải {names[-1]} -> {out}")
EOF
fi

bundle=${1:-$(ls -dt "$OUT"/medical_docs-* 2>/dev/null | head -1)}
if [ -z "$bundle" ] || [ ! -f "$bundle/manifest.json" ]; then
  echo "Không thấy bundle medical_docs-* trong $OUT (hoặc thiếu manifest.json)." >&2
  exit 1
fi
echo "Bundle: $bundle"

extra=()
[ "${OVERWRITE:-0}" = "1" ] && extra+=(--overwrite)
venv/bin/python -m scripts.qdrant_backup --url "$URL" restore --bundle "$bundle" "${extra[@]}"
