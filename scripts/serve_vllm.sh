#!/usr/bin/env bash
# Host Qwen3-8B-AWQ bang vLLM cho buoc sinh query va cham utility.
# Chay tren server GPU:  bash scripts/serve_vllm.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B-AWQ}"
PORT="${PORT:-8000}"

# FlashInfer sampler JIT-compile bang nvcc luc khoi dong; may khong co CUDA toolkit
# thi vLLM chet. Attention van chay FlashAttention 2, chi sampler doi sang PyTorch.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# --enable-prefix-caching: buoc cham tang 2 gui 8 request cung mot query,
#   prompt dat query TRUOC tai lieu nen phan instruction+query dung chung.
# --max-model-len 8192: query ke benh dai nhat 281 tu + tai lieu, 8192 du rong.
# --gpu-memory-utilization 0.6 (mac dinh): ~5.5GB weights + ~4GB KV cache, chua
#   ~6GB cho BGE-M3 + reranker chay cung GPU (pipeline sinh nhan). Chay vLLM mot
#   minh (vd. pha augment) muon nhanh hon: GPU_MEM_UTIL=0.88
exec vllm serve "$MODEL" \
  --quantization awq \
  --port "$PORT" \
  --max-model-len 8192 \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.6}" \
  --enable-prefix-caching \
  --no-enable-log-requests
