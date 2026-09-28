#!/usr/bin/env bash
# Uniform BoolQ evaluation on the Intel Arc Pro B70 (vLLM-XPU).
#
# One eval config for every sweep point so numbers are directly comparable:
#   max_model_len 4096 (BoolQ longest prompt ~730 tok - changes no actual
#   processing, only the init-time KV capacity check), fixed KV size (skips
#   vLLM's profiling, which the 28 GiB FP8 weights cannot afford), small
#   chunked-prefill budget (bounds the per-chunk logits matrix at
#   768 x 151936 x 4B ~ 0.45 GiB).
#
# Usage:
#   scripts/eval_b70.sh <tag>                       # custom checkpoint
#   MODEL_KIND=base_fp8 scripts/eval_b70.sh <tag>   # on-the-fly FP8 of base
# Env: MODEL_PATH, LIMIT (default 3270), GPU_MEM (0.95), MODEL_LEN (4096),
#      MAX_SEQS (8), MAX_BATCH_TOK (768), KV_CACHE_MEMORY (800000000)
set -euo pipefail
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env_xpu.sh

TAG="${1:?tag}"
MODEL_KIND="${MODEL_KIND:-custom}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
LIMIT="${LIMIT:-3270}"

if [ "$MODEL_KIND" = "custom" ]; then
  KIND_ARGS=(custom --model-path "$MODEL_PATH" --quantization inc)
else
  KIND_ARGS=(base_fp8)
fi

exec "$PY_VLLM_XPU" scripts/run_eval.py "${KIND_ARGS[@]}" \
  --output "results/b70/${TAG}" \
  --only boolq --limit "$LIMIT" \
  --gpu-mem "${GPU_MEM:-0.95}" \
  --max-model-len "${MODEL_LEN:-4096}" \
  --max-num-seqs "${MAX_SEQS:-8}" \
  --max-num-batched-tokens "${MAX_BATCH_TOK:-768}" \
  --kv-cache-memory "${KV_CACHE_MEMORY:-800000000}"
