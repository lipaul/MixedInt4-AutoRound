#!/usr/bin/env bash
# Build an FP8 variant on the B70 and evaluate it, after the W4 run finishes.
# vLLM-XPU cannot evaluate int8 (XPU_WNA16_SUPPORTED_BITS = {2,4}), so FP8 is
# the high-precision arm of the sweep.
set -u
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env_xpu.sh

echo "[fp8] waiting for the W4 eval to finish..."
while pgrep -f "[r]un_eval.py" >/dev/null; do sleep 30; done
echo "[fp8] starting $(date)"

OUT=/home/acm/work/models/xpu-fp8
rm -rf "$OUT"
"$VENV_XPU/bin/auto-round" quantize \
  --model_name "$BASE_MODEL" \
  --scheme FP8_BLOCK \
  --format fp8 \
  --algorithm rtn --iters 0 --disable_opt_rtn \
  --to_quant_block_names "model.language_model.layers,mtp.layers" \
  --output_dir "$OUT" \
  --device_map auto \
  --disable_torch_compile \
  > logs/xpu_quant_fp8.log 2>&1
echo "[fp8] quantize rc=$?"

DIR=$(find "$OUT" -maxdepth 2 -name config.json -printf '%h\n' 2>/dev/null | head -1)
if [ -z "$DIR" ]; then
  echo "[fp8] FAILED - tail of log:"
  tail -25 logs/xpu_quant_fp8.log
  exit 1
fi
echo "[fp8] model dir: $DIR"
du -sh "$DIR"
"$VENV_XPU/bin/python" -c "
import json,sys
c=json.load(open('$DIR/config.json')); q=c.get('quantization_config',{})
print('bits:',q.get('bits'),'| data_type:',q.get('data_type'),'| group:',q.get('group_size'),'| packing:',q.get('packing_format'))
"

./scripts/eval_boolq_b70.sh "$DIR" fp8 3270 > logs/b70_fp8_full.log 2>&1
echo "[fp8] BoolQ done $(date)"
