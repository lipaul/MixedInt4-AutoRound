#!/usr/bin/env bash
# Load test + MMLU throughput benchmark for the BF16 base with CPU offload.
set -euo pipefail
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env.sh
./scripts/gpu_free.sh
rm -rf results/bench_base
exec "$PY" scripts/run_eval.py base \
  --output results/bench_base \
  --only mmlu \
  --limit 20 \
  --offload-gb "${OFFLOAD_GB:-20}" \
  --max-num-seqs 32 \
  --max-num-batched-tokens 16384 \
  --max-model-len 8192 \
  --gpu-mem 0.88
