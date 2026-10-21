#!/usr/bin/env bash
# 05 Như 02 nhưng tìm hybrid (dense + sparse). Bắt tên thuốc, mã xét nghiệm tốt hơn.
source "$(dirname "$0")/common.sh"; need_ft
$R --retriever qdrant-hybrid --reranker-model $FT --final-k 40 \
   --checkpoint $OUT/A_ft_hyb.jsonl --out $OUT/A_ft_hyb.json
