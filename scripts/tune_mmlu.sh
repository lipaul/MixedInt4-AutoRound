#!/usr/bin/env bash
set -euo pipefail
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env.sh
./scripts/gpu_free.sh
rm -rf results/tune
exec "$PY" scripts/run_eval.py quant \
  --output results/tune \
  --only mmlu \
  --limit 20 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768 \
  --gpu-mem 0.85
