#!/usr/bin/env bash
# Full BoolQ on the Intel Arc Pro B70 via vLLM-XPU.
# Usage: scripts/eval_boolq_b70.sh <model_path> <tag> [limit]
set -euo pipefail
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env_xpu.sh

MODEL_PATH="${1:?model path}"
TAG="${2:?tag}"
LIMIT="${3:-3270}"

# gpu_mem 0.95 / small batched tokens: the B70 reports 30.3 GiB and the
# activation headroom is large, so 0.85 leaves no room for KV cache.
exec "$PY_VLLM_XPU" scripts/run_eval.py custom \
  --model-path "$MODEL_PATH" \
  --quantization inc \
  --output "results/b70/${TAG}" \
  --only boolq --limit "$LIMIT" \
  --gpu-mem 0.95 --max-model-len 8192 \
  --max-num-seqs 16 --max-num-batched-tokens 4096
