#!/usr/bin/env bash
# Evaluate full BoolQ on an arbitrary (custom) checkpoint.
# Usage: scripts/eval_boolq_custom.sh <model_path> <name>
set -euo pipefail
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env.sh

MODEL_PATH="${1:?model path}"
NAME="${2:?name}"

exec "$PY" scripts/run_eval.py custom \
  --model-path "$MODEL_PATH" \
  --quantization inc \
  --output "results/boolq_custom/${NAME}" \
  --only boolq --limit 3270 \
  --max-num-seqs 32 --max-num-batched-tokens 16384 \
  --max-model-len 32768 --gpu-mem 0.85 --max-gen-toks 4096
