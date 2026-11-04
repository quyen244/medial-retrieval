#!/usr/bin/env bash
# Chạy pipeline nhãn doc 3 mức trên server có GPU (vd RTX 5060 Ti), không cần Kaggle.
# Zip job dựng ở máy local: python -m src.rag.labeling.run_doc_labels kaggle --owner <user>
#
#   bash scripts/run_doc_labels_server.sh doc_labels_71b3b840a577.zip           # agree: đo tốc độ + độ khớp nhãn tay
#   bash scripts/run_doc_labels_server.sh doc_labels_71b3b840a577.zip judge     # chấm đủ 1.200 câu (chạy lại là làm tiếp)
#   bash scripts/run_doc_labels_server.sh doc_labels_71b3b840a577.zip judge --limit 50
#
# Kết quả nằm ở $OUT (mặc định out_labels): agree_<sha>_report.json, judged_<sha>.jsonl, run.log.
set -euo pipefail

ZIP="$(realpath "${1:?cần đường dẫn zip job}")"
MODE="${2:-agree}"
OUT="${OUT:-out_labels}"
cd "$(dirname "$0")/.."
mkdir -p "$OUT"

python -c "import vllm" 2>/dev/null || pip install -q vllm
python -c "import sentence_transformers" 2>/dev/null || pip install -q sentence-transformers
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| GPU', \
[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

case "$MODE" in
  agree) STEPS=embed,serve,agree ;;
  judge) STEPS=embed,serve,judge ;;
  all)   STEPS=embed,serve,agree,judge ;;
  *) echo "MODE phải là agree | judge | all" >&2; exit 1 ;;
esac

# --max-hours 1000: server không có giới hạn phiên 12 giờ như Kaggle; tham số sau ($@) ghi đè được.
PYTHONPATH=src/rag/labeling python kaggle/doc_label_job.py run --job "$ZIP" --out "$OUT" --steps "$STEPS" \
  --max-hours 1000 "${@:3}" 2>&1 | tee -a "$OUT/run.log"
