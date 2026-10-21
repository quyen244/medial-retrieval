#!/usr/bin/env bash
# 01 (CHÍNH) Phân rã không augment + rerank theo sub-query + giữ chỗ, reranker
# fine-tune, nộp 40 chunk/câu. Câu ngắn tìm bằng câu gốc như A.
source "$(dirname "$0")/common.sh"; need_ft
$R --augmented $AUG --plain-units --reranker-model $FT --final-k 40 \
   --checkpoint $OUT/D_ft.jsonl --out $OUT/D_ft.json
