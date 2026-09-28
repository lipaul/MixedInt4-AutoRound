#!/usr/bin/env bash
# Serve the MixedInt4-AutoRound quantized model with vLLM (text-only eval config).
set -euo pipefail
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env.sh
exec "$VENV/bin/vllm" serve "$QUANT_MODEL" \
  --served-model-name quant \
  --host 127.0.0.1 --port 8000 \
  --quantization inc \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --no-enable-log-requests
