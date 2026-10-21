#!/usr/bin/env bash
# 06 (có augment) B' đầy đủ: phân rã + augment + giữ chỗ, reranker fine-tune, 40 chunk.
# Mốc để biết augment còn tệ không khi đã có reranker fine-tune.
source "$(dirname "$0")/common.sh"; need_ft
$R --augmented $AUG --reranker-model $FT --final-k 40 \
   --checkpoint $OUT/Bp_ft.jsonl --out $OUT/Bp_ft.json
