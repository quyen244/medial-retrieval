# Dùng chung cho các script nộp bài 1200 câu. Chạy từ thư mục gốc repo.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source venv/bin/activate 2>/dev/null || true

OUT=tmp/sub1200
mkdir -p "$OUT"
AUG=data/augment_v2_full/augmented.jsonl
FT=./models/medical-reranker
R="python -m src.rag.run_full_submission --collection medical_documents"

need_ft() {
  if [ ! -f "$FT/config.json" ]; then
    echo "Chưa có model fine-tune ở $FT. Đợi train xong (tmux attach -t train)." >&2
    exit 1
  fi
}
need_top50() {
  if [ ! -f "$OUT/$1_top50.jsonl" ] || [ ! -f "$OUT/$1.json" ]; then
    echo "Thiếu $OUT/$1.json hoặc $OUT/$1_top50.jsonl. Chạy script tạo $1 trước." >&2
    exit 1
  fi
}
