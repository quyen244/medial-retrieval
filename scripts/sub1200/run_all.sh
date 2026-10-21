#!/usr/bin/env bash
# Chạy cả 10, theo thứ tự: các bài cần GPU trước, các bài lắp offline sau.
# Chạy trong tmux:  tmux new -s sub1200  rồi  bash scripts/sub1200/run_all.sh
# Bị ngắt thì chạy lại đúng lệnh: mỗi bài tiếp tục từ checkpoint của nó.
set -uo pipefail
DIR="$(dirname "$0")"
for s in 01_D_ft_k40 02_A_ft_k40 03_A_base_k40 04_D_ft_noslot_k40 05_A_ft_hyb_k40 06_Bp_ft_k40 \
         07_D_ft_k30 08_D_ft_k50 09_mix_Dshort_Bplong 10_D_ft_adaptive; do
  echo; echo "================ $s ================"
  bash "$DIR/$s.sh" || echo "!!! $s LỖI, chuyển sang bài tiếp"
done
echo; echo "File nộp:"; ls -la tmp/sub1200/*.zip
