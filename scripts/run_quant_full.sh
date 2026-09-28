#!/usr/bin/env bash
# Full benchmark set for the quantized model.
set -euo pipefail
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env.sh
./scripts/gpu_free.sh
exec "$PY" scripts/run_eval.py quant \
  --output results/quant \
  --max-num-seqs 32 \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768 \
  --gpu-mem 0.85 \
  --max-gen-toks 4096
