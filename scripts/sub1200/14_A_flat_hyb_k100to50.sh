#!/usr/bin/env bash
# 14 Không augment, không phân rã, KHÔNG chia ngôn ngữ (bỏ 3 nhánh vi/en/zh):
# 1 nhánh "all" duy nhất, hybrid (dense+sparse, RRF server-side) lấy 100 ứng
# viên, rerank GỐC lấy 50. So với 03 (A_base, chia ngôn ngữ): bỏ language lock
# có giúp không.
source "$(dirname "$0")/common.sh"
$R --retriever qdrant-hybrid-flat --flat-retrieve-k 100 --final-k 50 \
   --checkpoint $OUT/A_flat_hyb.jsonl --out $OUT/A_flat_hyb.json
