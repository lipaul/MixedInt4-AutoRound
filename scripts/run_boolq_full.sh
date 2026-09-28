#!/usr/bin/env bash
# Phase 0: full-size BoolQ (all 3270 docs) on both models for a paired test.
set -euo pipefail
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env.sh

echo "[boolq_full] quant starting $(date)"
./scripts/gpu_free.sh
"$PY" scripts/run_eval.py quant --output results/boolq_full/quant \
  --only boolq --limit 3270 \
  --max-num-seqs 32 --max-num-batched-tokens 16384 \
  --max-model-len 32768 --gpu-mem 0.85 --max-gen-toks 4096 \
  > logs/boolq_full_quant.log 2>&1
echo "[boolq_full] quant done $(date)"

echo "[boolq_full] base_fp8 starting $(date)"
./scripts/gpu_free.sh
"$PY" scripts/run_eval.py base_fp8 --output results/boolq_full/base \
  --only boolq --limit 3270 \
  --max-num-seqs 32 --max-num-batched-tokens 16384 \
  --max-model-len 32768 --gpu-mem 0.88 --max-gen-toks 4096 \
  > logs/boolq_full_base.log 2>&1
echo "[boolq_full] base done $(date)"

"$PY" scripts/mcnemar.py \
  --quant results/boolq_full/quant/boolq.json \
  --base  results/boolq_full/base/boolq.json
echo "[boolq_full] ALL DONE $(date)"
