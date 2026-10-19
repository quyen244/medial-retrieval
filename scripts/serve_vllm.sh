#!/usr/bin/env bash
# Host Qwen3-8B-AWQ bang vLLM cho buoc sinh query va cham utility.
# Chay tren server GPU:  bash scripts/serve_vllm.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B-AWQ}"
PORT="${PORT:-8000}"

# --enable-prefix-caching: buoc cham tang 2 gui 8 request cung mot query,
#   prompt dat query TRUOC tai lieu nen phan instruction+query dung chung.
# --max-model-len 8192: query ke benh dai nhat 281 tu + tai lieu, 8192 du rong.
# --gpu-memory-utilization 0.88: chua ~5.5GB weights, con lai cho KV cache.
exec vllm serve "$MODEL" \
  --quantization awq \
  --port "$PORT" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --enable-prefix-caching \
  --disable-log-requests
