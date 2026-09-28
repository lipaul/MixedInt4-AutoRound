#!/usr/bin/env bash
# Quantize the BF16 base with AutoRound in plain RTN mode (iters=0) at a given
# bit width, matching the original checkpoint's block selection.
#
# Usage: BITS=8 OUT=/home/acm/work/models/qwen38-w8-rtn scripts/quant_rtn.sh
#
# RTN is used deliberately: it holds the *method* constant so that comparing
# W4 vs W8 isolates the effect of bit width. The shipped checkpoint is
# AutoRound-optimized, so it is also evaluated as a separate data point.
set -euo pipefail
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env.sh

BITS="${BITS:?set BITS}"
OUT="${OUT:?set OUT}"

./scripts/gpu_free.sh
rm -rf "$OUT"

"$VENV/bin/auto-round" quantize \
  --model_name "$BASE_MODEL" \
  --bits "$BITS" \
  --group_size 32 \
  --format auto_round \
  --algorithm rtn \
  --iters 0 \
  --nsamples 128 \
  --seqlen 2048 \
  --to_quant_block_names "model.language_model.layers,mtp.layers" \
  --output_dir "$OUT" \
  --device_map auto \
  --disable_torch_compile \
  2>&1 | tail -40

echo "[quant_rtn] W${BITS} written to $OUT"
du -sh "$OUT"
