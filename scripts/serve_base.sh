#!/usr/bin/env bash
# Serve the BF16 base model (Qwen/Qwen3.8-27B) with CPU weight offload.
# 27.78B params * 2 bytes = ~55.6 GB weights, so ~18 GB must live in host RAM.
set -euo pipefail
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env.sh
OFFLOAD_GB="${OFFLOAD_GB:-18}"
PORT="${PORT:-8001}"
exec "$VENV/bin/vllm" serve "$BASE_MODEL" \
  --served-model-name base \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --cpu-offload-gb "$OFFLOAD_GB" \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --no-enable-log-requests
