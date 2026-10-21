#!/usr/bin/env bash
# 09 (không cần GPU, ít augment) Câu ngắn từ 01 (câu gốc, không augment), câu dài
# từ 06 (phân rã có augment). Augment chỉ ở câu dài, nơi nó ít hại nhất.
source "$(dirname "$0")/common.sh"; need_top50 D_ft; need_top50 Bp_ft
python -m scripts.compose_submission mix --short $OUT/D_ft.json --long $OUT/Bp_ft.json \
   --out $OUT/mix_Dshort_Bplong.json
