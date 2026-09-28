#!/usr/bin/env bash
# Phase 2: does increasing bit width recover BoolQ?
#
# Holds the quantization *method* fixed (plain RTN, iters=0) and varies only the
# bit width, so W4 vs W8 isolates precision. Existing data points (the shipped
# AutoRound W4 checkpoint and the BF16/FP8 base) are added by the summary.
set -u
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env.sh

echo "[phase2] waiting for the boolq_full run to finish..."
while pgrep -f "[r]un_boolq_full" >/dev/null; do sleep 30; done
echo "[phase2] starting $(date)"

for BITS in 8 4; do
  OUT="/home/acm/work/models/qwen38-w${BITS}-rtn"
  echo "[phase2] === RTN W${BITS} quantize -> ${OUT} ==="
  BITS="$BITS" OUT="$OUT" ./scripts/quant_rtn.sh > "logs/quant_w${BITS}.log" 2>&1
  rc=$?
  MODEL_DIR=$(find "$OUT" -maxdepth 2 -name config.json -printf '%h\n' 2>/dev/null | head -1)
  if [ -z "$MODEL_DIR" ]; then
    echo "[phase2] RTN W${BITS} FAILED (rc=$rc); see logs/quant_w${BITS}.log"
    tail -30 "logs/quant_w${BITS}.log"
    continue
  fi
  echo "[phase2] RTN W${BITS} model dir: $MODEL_DIR"
  echo "[phase2] === RTN W${BITS} BoolQ eval ==="
  ./scripts/gpu_free.sh
  ./scripts/eval_boolq_custom.sh "$MODEL_DIR" "w${BITS}_rtn" > "logs/boolq_w${BITS}.log" 2>&1
  echo "[phase2] W${BITS} BoolQ done $(date)"
done

./scripts/gpu_free.sh
"$PY" scripts/bits_summary.py
echo "[phase2] ALL DONE $(date)"
