#!/usr/bin/env bash
# 13 Bản sao 01, MỌI bước theo căn bậc hai phân bố corpus: lấy 65 (vi 38 / en 16 /
# zh 11), giữ 50 (vi 29 / en 13 / zh 8), sub-query giữ 20 (vi 12 / en 5 / zh 3).
source "$(dirname "$0")/common.sh"; need_ft
$R --augmented $AUG --plain-units --reranker-model $FT --final-k 40 --branch-quota sqrt --sub-quota sqrt \
   --checkpoint $OUT/D_ft_allsqrt.jsonl --out $OUT/D_ft_allsqrt.json
