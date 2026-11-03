#!/usr/bin/env bash
# ==============================================================================
# Script chạy embedding shard với backend vLLM cho Qwen3-Embedding-4B trên GPU server.
#
# Cách dùng:
#   1. Chạy phân vùng 0/8 (tiếp tục shard dở, bỏ qua shard đã có):
#      bash scripts/run_vllm_embed.sh
#      hoặc: bash scripts/run_vllm_embed.sh 0/8
#
#   2. Chạy lần lượt các phân vùng còn thiếu (0/8 -> 3/8 -> 4/8 -> 5/8 -> 6/8 -> 7/8 -> 1/8):
#      bash scripts/run_vllm_embed.sh loop 0/8 3/8 4/8 5/8 6/8 7/8 1/8
#
#   3. Chạy nền (nohup) ghi log vào logs/emb_vllm_server.log:
#      bash scripts/run_vllm_embed.sh bg 0/8 3/8 4/8 5/8
#
# Các biến môi trường tuỳ chỉnh:
#   PYTHON_BIN:       Đường dẫn python binary (mặc định dò .venv-vllm/bin/python)
#   DEVICE:           Thiết bị GPU (mặc định cuda:0)
#   GPU_MEM_UTIL:     Tỉ lệ VRAM cho vLLM (mặc định 0.9)
#   VLLM_BATCH_SIZE:  Số prompt tokenized mỗi sub-batch (mặc định 500)
#   JOB_ZIP:          Đường dẫn file job zip
#   OUT_DIR:          Thư mục lưu vector kết quả
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 1. Dò tìm Python binary trong .venv-vllm
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x ".venv-vllm/bin/python" ]]; then
  PY=".venv-vllm/bin/python"
elif [[ -x "venv/bin/python" ]]; then
  PY="venv/bin/python"
else
  PY="$(which python3)"
fi

echo "=== [vLLM Embed Runner] ==="
echo "Repo root : $ROOT_DIR"
echo "Python    : $PY"

# 2. Cấu hình mặc định
MODEL="${MODEL:-Qwen/Qwen3-Embedding-4B}"
JOB_ZIP="${JOB_ZIP:-data/kaggle/embed_dataset/emb_full_q3e4b.zip}"
OUT_DIR="${OUT_DIR:-data/kaggle/out/emb_full_q3e4b_server/q3e4b}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-fp16}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
BATCH_TOKENS="${BATCH_TOKENS:-12288}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-500}"
MAX_HOURS="${MAX_HOURS:-1000.0}"

DONE_DIRS=(
  "data/kaggle/out/emb_full_q3e4b_p0of8/q3e4b"
  "data/kaggle/out/emb_full_q3e4b_p2of8/q3e4b"
)

# Đảm bảo tắt FlashInfer sampler nếu thiếu CUDA Toolkit để tránh lỗi JIT
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$OUT_DIR")"
mkdir -p logs

run_single_part() {
  local part="$1"
  echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Bắt đầu embedding part: $part (backend=vLLM) <<<"
  "$PY" kaggle/embed_job.py embed \
    --backend vllm \
    --job "$JOB_ZIP" \
    --model "$MODEL" \
    --out "$OUT_DIR" \
    --part "$part" \
    --worker "0/1" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --max-tokens "$MAX_TOKENS" \
    --batch-tokens "$BATCH_TOKENS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --vllm-batch-size "$VLLM_BATCH_SIZE" \
    --max-hours "$MAX_HOURS" \
    --done-dirs "${DONE_DIRS[@]}"
  echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Hoàn thành hoặc đã xử lý xong part: $part <<<"
}

MODE="${1:-0/8}"

if [[ "$MODE" == "bg" ]]; then
  shift
  PARTS=("${@:-0/8 3/8 4/8 5/8 6/8 7/8 1/8}")
  LOG_FILE="logs/emb_vllm_server.log"
  echo "Chạy nền các part: ${PARTS[*]}"
  echo "Log chuyển vào: $LOG_FILE"
  nohup bash "$0" loop "${PARTS[@]}" > "$LOG_FILE" 2>&1 &
  echo "PID: $!"
  exit 0
elif [[ "$MODE" == "loop" ]]; then
  shift
  PARTS=("${@:-0/8 3/8 4/8 5/8 6/8 7/8 1/8}")
  echo "Chạy tuần tự các part: ${PARTS[*]}"
  for p in "${PARTS[@]}"; do
    run_single_part "$p"
  done
else
  # Chạy 1 part duy nhất (mặc định $MODE, vd: 0/8 hoặc 3/8)
  run_single_part "$MODE"
fi
