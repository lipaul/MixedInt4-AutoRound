#!/usr/bin/env bash
# W2 uniform floor point for the B70 bits-vs-BoolQ curve.
# vLLM-XPU supports int bits {2,4}, so W2g32 (2 + 16/32 = 2.5 eff. bits) is
# runnable - unlike int8. Waits for the module-ablation sweep to finish.
set -u
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env_xpu.sh

echo "[w2] waiting for b70_sweep.sh to finish..."
while pgrep -f "[b]70_sweep.sh" >/dev/null; do sleep 30; done
echo "[w2] starting $(date +%H:%M:%S)"

OUT=/home/acm/work/models/xpu-variants/w2_rtn
rm -rf "$OUT"
"$VENV_XPU/bin/auto-round" quantize \
  --model_name "$BASE_MODEL" \
  --bits 2 --group_size 32 \
  --format auto_round \
  --algorithm rtn --iters 0 --disable_opt_rtn \
  --nsamples 128 --seqlen 2048 \
  --to_quant_block_names "model.language_model.layers,mtp.layers" \
  --output_dir "$OUT" \
  --device_map auto \
  --disable_torch_compile \
  > logs/sweep_quant_w2_rtn.log 2>&1
rc=$?
DIR=$(find "$OUT" -maxdepth 2 -name config.json -printf '%h\n' 2>/dev/null | head -1)
if [ -z "$DIR" ] || [ $rc -ne 0 ]; then
  echo "[w2] QUANT FAILED rc=$rc - see logs/sweep_quant_w2_rtn.log"
  exit 1
fi
echo "[w2] checkpoint: $DIR ($(du -sh "$DIR" | cut -f1))"

MODEL_PATH="$DIR" scripts/eval_b70.sh w2_rtn > logs/sweep_eval_w2_rtn.log 2>&1
echo "[w2] eval rc=$? ($(date +%H:%M:%S))"
