#!/usr/bin/env bash
# Module-level mixed-precision ablation on the Intel Arc Pro B70 (XPU only).
#
# Each variant = uniform W4g32 RTN (identical to the xpu-w4-rtn anchor) with
# one module GROUP promoted to BF16 via --layer_config. vLLM-XPU cannot load
# int8 (XPU INC supports int bits {2,4} only), so BF16 is the promotion
# target; the FP8 arm (vLLM on-the-fly, results/b70/fp8_vllm_u) is the 8-bit
# ceiling. All points are evaluated with scripts/eval_b70.sh - one uniform
# eval config - on the full 3270-doc BoolQ.
#
# Layer groups (module counts incl. the mtp layer where the substring
# matches):
#   v1_inprojab  in_proj_a + in_proj_b + mtp.fc        97  (mirrors the CUDA
#                W4 coverage / the shipped model's fp16 keep-set)
#   v2_outproj   linear_attn.out_proj + self_attn.o_proj  65
#   v3_downproj  mlp.down_proj                           65
#   v4_qkvz      linear_attn.in_proj_qkv + in_proj_z     96
#   v5_linattn   everything under linear_attn           240
set -u
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env_xpu.sh

MODELS=/home/acm/work/models/xpu-variants
mkdir -p "$MODELS"

wait_gpu() {
  while pgrep -f "[r]un_eval.py" >/dev/null || pgrep -f "[a]uto-round quantize" >/dev/null; do
    sleep 30
  done
}

verify_promoted() {  # $1 config.json  $2 expect-count  $3 patterns(csv)
  "$VENV_XPU/bin/python" - "$1" "$2" "$3" <<'PY'
import json, re, sys
cfg = json.load(open(sys.argv[1]))
q = cfg.get("quantization_config", {})
ec = q.get("extra_config", {})
patterns = [p.strip() for p in sys.argv[3].split(",") if p.strip()]
promoted = 0
for name, v in ec.items():
    if v.get("bits") == 16 and any(p and re.search(re.escape(p), name) for p in patterns):
        promoted += 1
expect = int(sys.argv[2])
print(f"[verify] promoted {promoted} / expected {expect} (extra_config total {len(ec)})")
sys.exit(0 if promoted >= expect else 1)
PY
}

run_variant() {  # $1 name  $2 layer_config JSON  $3 expect-count  $4 patterns(csv)
  local name=$1 layercfg=$2 expect=$3 patterns=$4
  wait_gpu
  local out="$MODELS/$name"
  rm -rf "$out"
  echo "[sweep] === $name: quantize ($(date +%H:%M:%S)) ==="
  "$VENV_XPU/bin/auto-round" quantize \
    --model_name "$BASE_MODEL" \
    --bits 4 --group_size 32 \
    --format auto_round \
    --algorithm rtn --iters 0 --disable_opt_rtn \
    --nsamples 128 --seqlen 2048 \
    --to_quant_block_names "model.language_model.layers,mtp.layers" \
    --layer_config "$layercfg" \
    --output_dir "$out" \
    --device_map auto \
    --disable_torch_compile \
    > "logs/sweep_quant_${name}.log" 2>&1
  local dir
  dir=$(find "$out" -maxdepth 2 -name config.json -printf '%h\n' 2>/dev/null | head -1)
  if [ -z "$dir" ]; then
    echo "[sweep] $name QUANT FAILED - see logs/sweep_quant_${name}.log"
    return 1
  fi
  if ! verify_promoted "$dir/config.json" "$expect" "$patterns"; then
    echo "[sweep] $name VERIFY FAILED (layer_config not honored?) - dir: $dir"
    return 1
  fi
  du -sh "$dir"
  echo "[sweep] === $name: eval full BoolQ ($(date +%H:%M:%S)) ==="
  MODEL_PATH="$dir" scripts/eval_b70.sh "$name" > "logs/sweep_eval_${name}.log" 2>&1
  local rc=$?
  echo "[sweep] $name eval rc=$rc ($(date +%H:%M:%S))"
  return $rc
}

echo "[sweep] waiting for GPU to be free..."
wait_gpu

# Re-run the W4 anchor under the uniform eval config for a fully consistent
# curve (its 83.73% was measured with the earlier 4096/16 settings).
echo "[sweep] === w4_rtn_u: re-eval anchor ($(date +%H:%M:%S)) ==="
rm -rf results/b70/w4_rtn_u
MODEL_PATH=/home/acm/work/models/xpu-w4-rtn scripts/eval_b70.sh w4_rtn_u \
  > logs/sweep_eval_w4_rtn_u.log 2>&1
echo "[sweep] w4_rtn_u rc=$? ($(date +%H:%M:%S))"
wait_gpu

run_variant v1_inprojab \
  '{"linear_attn.in_proj_a":"BF16","linear_attn.in_proj_b":"BF16","mtp.fc":"BF16"}' \
  97 "linear_attn.in_proj_a,linear_attn.in_proj_b,mtp.fc"
wait_gpu

run_variant v2_outproj \
  '{"linear_attn.out_proj":"BF16","self_attn.o_proj":"BF16"}' \
  65 "linear_attn.out_proj,self_attn.o_proj"
wait_gpu

run_variant v4_qkvz \
  '{"linear_attn.in_proj_qkv":"BF16","linear_attn.in_proj_z":"BF16"}' \
  96 "linear_attn.in_proj_qkv,linear_attn.in_proj_z"
wait_gpu

run_variant v3_downproj \
  '{"mlp.down_proj":"BF16"}' \
  65 "mlp.down_proj"
wait_gpu

run_variant v5_linattn \
  '{"linear_attn":"BF16"}' \
  240 "linear_attn."
wait_gpu

echo "[sweep] ALL DONE $(date +%H:%M:%S)"
