#!/usr/bin/env bash
# AutoRound quantization on the Intel Arc Pro B70 (XPU).
# Usage: BITS=4 OUT=/home/acm/work/models/xpu-w4 ./scripts/quant_rtn_xpu.sh
set -euo pipefail
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env_xpu.sh

BITS="${BITS:?set BITS}"
OUT="${OUT:?set OUT}"

rm -rf "$OUT"

"$VENV_XPU/bin/auto-round" quantize \
  --model_name "$BASE_MODEL" \
  --bits "$BITS" \
  --group_size 32 \
  --format auto_round \
  --algorithm rtn \
  --iters 0 \
  --disable_opt_rtn \
  --nsamples 128 \
  --seqlen 2048 \
  --to_quant_block_names "model.language_model.layers,mtp.layers" \
  --output_dir "$OUT" \
  --device_map auto \
  --disable_torch_compile \
  2>&1 | tail -30

MODEL_DIR=$(find "$OUT" -maxdepth 2 -name config.json -printf '%h\n' 2>/dev/null | head -1)
echo "[quant_rtn_xpu] W${BITS} -> ${MODEL_DIR:-FAILED}"
[ -n "$MODEL_DIR" ] && du -sh "$MODEL_DIR"
