#!/usr/bin/env bash
# FP8 base on the cheap benchmarks first, so a partial Recovery Rate can be
# computed without waiting for ARC/MMLU/GSM8K.
set -euo pipefail
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env.sh
./scripts/gpu_free.sh
exec "$PY" scripts/run_eval.py base_fp8 \
  --output results/base \
  --only piqa,winogrande,boolq,hellaswag \
  --max-num-seqs 32 \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768 \
  --gpu-mem 0.88 \
  --max-gen-toks 4096
