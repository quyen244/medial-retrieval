#!/usr/bin/env bash
# 11 (GPU, chạy MỘT lần) Cấu hình y hệt 01 nhưng ghi lại MỌI ứng viên kèm điểm
# (--dump-all -> D_all_all.jsonl, ~180 MB). Từ đó script 12 lắp offline mọi kiểu
# chia top-k sub-query x K nộp mà không chạy GPU lại.
source "$(dirname "$0")/common.sh"; need_ft
$R --augmented $AUG --plain-units --reranker-model $FT --final-k 40 --dump-all \
   --checkpoint $OUT/D_all.jsonl --out $OUT/D_all.json
