#!/usr/bin/env bash
# 08 (không cần GPU) Lắp lại 01 với 50 chunk/câu (toàn bộ tập giữ sau rerank).
source "$(dirname "$0")/common.sh"; need_top50 D_ft
python -m scripts.assemble_submission build v1 --k 50 \
   --top50 $OUT/D_ft_top50.jsonl --base $OUT/D_ft.json --out $OUT/D_ft_k50.json
