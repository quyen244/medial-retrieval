#!/usr/bin/env bash
# 10 (không cần GPU) Lắp lại 01, số chunk thay đổi theo điểm rerank: tối thiểu 20,
# tối đa 40 (số chunk bài 01 đã nộp), giữ chunk có điểm trên phân vị 20.
source "$(dirname "$0")/common.sh"; need_top50 D_ft
python -m scripts.assemble_submission stats --top50 $OUT/D_ft_top50.jsonl
python -m scripts.assemble_submission build v2 --tau-pct 20 --min-k 20 --max-k 40 \
   --top50 $OUT/D_ft_top50.jsonl --base $OUT/D_ft.json --out $OUT/D_ft_adaptive.json
