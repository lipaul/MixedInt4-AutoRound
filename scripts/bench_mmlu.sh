#!/usr/bin/env bash
# Throughput benchmark: MMLU, limit 20, tuned batching. Quant model.
set -euo pipefail
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env.sh
./scripts/gpu_free.sh
rm -rf results/bench
exec "$PY" scripts/run_eval.py quant \
  --output results/bench \
  --only mmlu \
  --limit 20 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 65536 \
  --max-model-len 8192 \
  --gpu-mem 0.90
