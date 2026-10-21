#!/usr/bin/env bash
# 07 (không cần GPU) Lắp lại 01 với 30 chunk/câu. Cần chạy 01 trước.
source "$(dirname "$0")/common.sh"; need_top50 D_ft
python -m scripts.assemble_submission build v1 --k 30 \
   --top50 $OUT/D_ft_top50.jsonl --base $OUT/D_ft.json --out $OUT/D_ft_k30.json
