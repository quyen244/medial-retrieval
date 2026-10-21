#!/usr/bin/env bash
# 03 Như 02 nhưng reranker GỐC (bge-reranker-v2-m3). So với 02: fine-tune có giúp không.
source "$(dirname "$0")/common.sh"
$R --final-k 40 \
   --checkpoint $OUT/A_base.jsonl --out $OUT/A_base.json
