#!/usr/bin/env bash
# Thí nghiệm tune top-k (11-13), chạy SAU khi 01 xong (cùng dùng GPU).
#   tmux new -s tune   rồi   bash scripts/sub1200/run_tune.sh
# 11 và 13 cần GPU, mỗi bài ~27 phút; 12 chạy offline vài giây. Bị ngắt thì chạy lại: tiếp tục từ checkpoint.
set -uo pipefail
DIR="$(dirname "$0")"
for s in 11_D_all_dump 12_tune_offline 13_D_ft_allsqrt_k40; do
  echo; echo "================ $s ================"
  bash "$DIR/$s.sh" || echo "!!! $s LỖI, chuyển sang bài tiếp"
done
echo; echo "File nộp:"; ls -la tmp/sub1200/*.zip
