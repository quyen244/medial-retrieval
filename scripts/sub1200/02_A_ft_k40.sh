#!/usr/bin/env bash
# 02 Câu gốc (không phân rã, không augment), reranker fine-tune, 40 chunk.
# So với 01: phân rã có giúp ở câu dài không.
source "$(dirname "$0")/common.sh"; need_ft
$R --reranker-model $FT --final-k 40 \
   --checkpoint $OUT/A_ft.jsonl --out $OUT/A_ft.json
