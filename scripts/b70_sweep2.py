#!/usr/bin/env python
"""Follow-up B70 sweep: variants that fit the ~23.9 GiB load budget.

The full-promotion variants v3 (down_proj, 26.3 GiB) and v5 (all linear_attn,
26.0 GiB) OOM during vLLM weight *loading* on the 30.3 GiB B70 (v4 at 23.9 GiB
is the empirical ceiling), so the big module groups are probed with
dose-response / split variants instead:

  v3h_downhalf  down_proj -> fp16 in every other language layer (32)   22.5 GiB
  v7_selfqkv    self_attn q/k/v -> fp16 (16 layers + mtp)               20.2 GiB
  v8_outonly    linear_attn.out_proj only -> fp16 (48)                 20.6 GiB
  v9_oonly      self_attn.o_proj only -> fp16 (16 + mtp)               19.3 GiB
  v10_gateupq   mlp gate+up -> fp16 in 16 layers (quarter dose)         22.5 GiB

Every variant keeps the uniform W4g32 RTN recipe + the uniform eval config.
Waits for any running GPU job first (e.g. the W2 floor point).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

ROOT = "/home/acm/paul_nv/MixedInt4-AutoRound"
VENV = f"{ROOT}/.venv-xpu"
MODELS = "/home/acm/work/models/xpu-variants"
BASE_MODEL = "/home/acm/work/models/Qwen3.8-27B"

EVEN_LAYERS = list(range(0, 64, 2))
GATEUP_LAYERS = list(range(0, 64, 4))  # quarter dose, spread across depth

VARIANTS = [
    ("v3h_downhalf",
     {f"model.language_model.layers.{i}.mlp.down_proj": "BF16" for i in EVEN_LAYERS},
     32, ["mlp.down_proj"]),
    ("v7_selfqkv",
     {"self_attn.q_proj": "BF16", "self_attn.k_proj": "BF16", "self_attn.v_proj": "BF16"},
     51, ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
    ("v8_outonly",
     {"linear_attn.out_proj": "BF16"},
     48, ["linear_attn.out_proj"]),
    ("v9_oonly",
     {"self_attn.o_proj": "BF16"},
     17, ["self_attn.o_proj"]),
    ("v10_gateupq",
     {f"model.language_model.layers.{i}.mlp.{p}": "BF16"
      for i in GATEUP_LAYERS for p in ("gate_proj", "up_proj")},
     32, ["mlp.gate_proj", "mlp.up_proj"]),
]


def gpu_busy() -> bool:
    for pat in ("run_eval.py", "auto-round quantize", "b70_w2.sh", "b70_sweep.sh"):
        r = subprocess.run(["pgrep", "-f", f"[{pat[0]}]{pat[1:]}"],
                            capture_output=True)
        if r.returncode == 0:
            return True
    return False


def wait_gpu():
    while gpu_busy():
        time.sleep(30)


def sh(cmd, log):
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=ROOT).returncode


def find_ckpt(out):
    for root, dirs, files in os.walk(out):
        if "config.json" in files:
            return root
    return None


def verify(config_json: str, patterns: list[str], expect: int) -> bool:
    q = json.load(open(config_json)).get("quantization_config", {})
    ec = q.get("extra_config", {})
    promoted = sum(
        1 for name, v in ec.items()
        if v.get("bits") == 16 and any(p in name for p in patterns)
    )
    print(f"[verify] promoted {promoted} / expected {expect} "
          f"(extra_config total {len(ec)})", flush=True)
    # >= : AutoRound also serializes the non-exact pattern keys themselves
    # into extra_config (sweep1: v1 counted 99 = 97 + 2 pattern keys), so the
    # raw count can legitimately exceed the number of promoted modules.
    return promoted >= expect


def main():
    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    print("[sweep2] waiting for GPU...", flush=True)
    wait_gpu()
    for name, lc, expect, patterns in VARIANTS:
        print(f"[sweep2] === {name}: quantize ({time.strftime('%H:%M:%S')}) ===",
              flush=True)
        out = f"{MODELS}/{name}"
        subprocess.run(["rm", "-rf", out], check=True)
        rc = sh([f"{VENV}/bin/auto-round", "quantize",
                 "--model_name", BASE_MODEL,
                 "--bits", "4", "--group_size", "32",
                 "--format", "auto_round",
                 "--algorithm", "rtn", "--iters", "0", "--disable_opt_rtn",
                 "--nsamples", "128", "--seqlen", "2048",
                 "--to_quant_block_names", "model.language_model.layers,mtp.layers",
                 "--layer_config", json.dumps(lc),
                 "--output_dir", out,
                 "--device_map", "auto",
                 "--disable_torch_compile"],
                f"{ROOT}/logs/sweep2_quant_{name}.log")
        ckpt = find_ckpt(out)
        if rc != 0 or ckpt is None:
            print(f"[sweep2] {name} QUANT FAILED rc={rc}", flush=True)
            continue
        if not verify(f"{ckpt}/config.json", patterns, expect):
            print(f"[sweep2] {name} VERIFY FAILED - skipping eval", flush=True)
            continue
        print(f"[sweep2] === {name}: eval full BoolQ ({time.strftime('%H:%M:%S')}) ===",
              flush=True)
        env = dict(os.environ, MODEL_PATH=ckpt)
        rc = subprocess.run([f"{ROOT}/scripts/eval_b70.sh", name],
                            cwd=ROOT, env=env,
                            stdout=open(f"{ROOT}/logs/sweep2_eval_{name}.log", "w"),
                            stderr=subprocess.STDOUT).returncode
        print(f"[sweep2] {name} eval rc={rc} ({time.strftime('%H:%M:%S')})", flush=True)
    print("[sweep2] ALL DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
