#!/usr/bin/env bash
# Full benchmark set for the FP8 base (stand-in for the bf16 baseline).
set -euo pipefail
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env.sh
./scripts/gpu_free.sh
exec "$PY" scripts/run_eval.py base_fp8 \
  --output results/base \
  --max-num-seqs 32 \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768 \
  --gpu-mem 0.90 \
  --max-gen-toks 4096
