#!/usr/bin/env bash
# 8-bit arm: vLLM's own on-the-fly FP8 quantization of the BF16 base, on the B70.
# The BF16 base is 55.6 GB and converts to ~28 GB of FP8 weights, which just
# fits the B70's 30.3 GiB - but leaves no room for vLLM's profiling to find KV,
# so an explicit KV size is usually required (KV_CACHE_MEMORY=bytes).
set -euo pipefail
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env_xpu.sh

EXTRA=()
if [ -n "${KV_CACHE_MEMORY:-}" ]; then
  EXTRA+=(--kv-cache-memory "$KV_CACHE_MEMORY")
fi

exec "$PY_VLLM_XPU" scripts/run_eval.py base_fp8 \
  --output results/b70/fp8_vllm \
  --only boolq --limit "${LIMIT:-3270}" \
  --gpu-mem "${GPU_MEM:-0.95}" --max-model-len 8192 \
  --max-num-seqs "${MAX_SEQS:-16}" --max-num-batched-tokens "${MAX_BATCH_TOK:-4096}" \
  "${EXTRA[@]}"
