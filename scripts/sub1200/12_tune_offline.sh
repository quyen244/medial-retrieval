#!/usr/bin/env bash
# 12 (không cần GPU, vài giây) Từ dump của 11: kiểm tra lắp lại đúng bài 11, rồi
# lắp lưới sub-query giữ 20 chunk chia theo
#   global = top 20 chung (như 01) | fixed = vi 10/en 7/zh 3 (tỉ lệ 26/16/8)
#   sqrt   = vi 12/en 5/zh 3 (căn bậc hai corpus) | corpus = vi 16/en 3/zh 1
# x K nộp 30 / 40 / 50  -> 12 bài D_tune_sub<kiểu>_k<K>.zip
source "$(dirname "$0")/common.sh"; need_top50 D_all
S="python -m scripts.simulate_subquota"
$S check --all $OUT/D_all_all.jsonl --base $OUT/D_all.json --final-k 40
$S grid --all $OUT/D_all_all.jsonl --base $OUT/D_all.json \
   --sub-quota global fixed sqrt corpus --final-k 30 40 50 --out-prefix $OUT/D_tune
