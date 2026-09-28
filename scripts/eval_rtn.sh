#!/usr/bin/env bash
# Evaluate BoolQ on the RTN W8/W4 checkpoints produced by quant_rtn.sh,
# locating the real model directory (AutoRound writes to a named subdir).
set -u
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env.sh

# shellcheck disable=SC2086
for BITS in ${ORDER:-8 4}; do
  ROOT="/home/acm/work/models/qwen38-w${BITS}-rtn"
  MODEL_DIR=$(find "$ROOT" -maxdepth 2 -name config.json -printf '%h\n' 2>/dev/null | head -1)
  if [ -z "$MODEL_DIR" ]; then
    echo "[eval_rtn] W${BITS}: no model at $ROOT (skipping)"
    continue
  fi
  # already evaluated?
  if [ -f "results/boolq_custom/w${BITS}_rtn/boolq.json" ]; then
    echo "[eval_rtn] W${BITS}: result exists, skipping"
    continue
  fi
  echo "[eval_rtn] W${BITS} -> $MODEL_DIR  ($(date))"
  ./scripts/gpu_free.sh
  ./scripts/eval_boolq_custom.sh "$MODEL_DIR" "w${BITS}_rtn" > "logs/boolq_w${BITS}.log" 2>&1
  echo "[eval_rtn] W${BITS} done $(date)"
done

"$PY" scripts/bits_summary.py
