#!/usr/bin/env bash
# BF16 drift check: bound how far the FP8 baseline is from true BF16.
# Deliberately tiny - the CPU-offloaded bf16 model is extremely slow.
# Run AFTER the fp8 baseline, using the SAME task/settings so the two are
# directly comparable on the same subset.
set -euo pipefail
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env.sh
./scripts/gpu_free.sh
exec "$PY" scripts/run_eval.py base \
  --output results/base_bf16_drift \
  --only mmlu,arc_challenge,piqa,winogrande \
  --limit "${DRIFT_LIMIT:-5}" \
  --offload-gb 20 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --max-model-len 8192 \
  --gpu-mem 0.88 \
  --tag bf16drift
