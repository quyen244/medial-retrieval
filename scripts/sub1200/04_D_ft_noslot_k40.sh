#!/usr/bin/env bash
# 04 Như 01 nhưng TẮT rerank theo sub-query + giữ chỗ. So với 01: giữ chỗ có giúp không.
source "$(dirname "$0")/common.sh"; need_ft
$R --augmented $AUG --plain-units --no-sub-rerank --reranker-model $FT --final-k 40 \
   --checkpoint $OUT/D_ft_noslot.jsonl --out $OUT/D_ft_noslot.json
