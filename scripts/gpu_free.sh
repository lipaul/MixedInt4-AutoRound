#!/usr/bin/env bash
# Kill any orphaned vLLM engine/server processes and wait for GPU memory to free.
set -u
pkill -9 -f "[r]un_eval.py" 2>/dev/null
pkill -9 -f "[E]ngineCore" 2>/dev/null
pkill -9 -f "[v]llm serve" 2>/dev/null
for _ in $(seq 1 20); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  # chrome/desktop baseline is ~1.2 GiB
  if [ "$used" -lt 4000 ]; then break; fi
  sleep 1
done
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
