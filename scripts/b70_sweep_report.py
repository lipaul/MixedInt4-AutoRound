#!/usr/bin/env python
"""B70 mixed-precision sweep report: bits vs BoolQ (n=3270, uniform eval config).

Combines, for every completed sweep point:
  - effective avg bits (param-weighted; int4g32 = 4.5 bits/w, fp16 = 16)
  - full-BoolQ accuracy (+/- stderr)
  - paired McNemar vs the FP8 anchor
  - loaded weights GiB and eval throughput (from the eval logs)

Usage: scripts/b70_sweep_report.py [--json results/b70/sweep_report.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import struct

BASE = "/home/acm/work/models/Qwen3.8-27B"
QUANTIZABLE = (
    "linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z", "linear_attn.out_proj",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.down_proj", "mlp.gate_proj", "mlp.up_proj",
    "mtp.fc", "mtp.layers.",
)
# int4 symmetric, group 32, fp16 scale -> 4 + 16/32 bits per weight
INT4G32_BITS = 4.5
# fp8 e4m3 + fp32 per-output-channel scale (~4096-wide rows) -> ~8.008
FP8_BITS = 8.008

VARIANTS = [
    # tag, checkpoint (None = on-the-fly FP8 base), log, description
    ("w4_rtn_u", "/home/acm/work/models/xpu-w4-rtn", "logs/sweep_eval_w4_rtn_u.log",
     "uniform W4g32 (anchor, uniform eval cfg)"),
    ("w2_rtn", "/home/acm/work/models/xpu-variants/w2_rtn", "logs/sweep_eval_w2_rtn.log",
     "uniform W2g32 (floor point, 2.5 eff. bits)"),
    ("v1_inprojab", "/home/acm/work/models/xpu-variants/v1_inprojab", "logs/sweep_eval_v1_inprojab.log",
     "W4 + in_proj_a/b + mtp.fp -> fp16 (= CUDA W4 coverage)"),
    ("v2_outproj", "/home/acm/work/models/xpu-variants/v2_outproj", "logs/sweep_eval_v2_outproj.log",
     "W4 + out_proj/o_proj -> fp16"),
    ("v4_qkvz", "/home/acm/work/models/xpu-variants/v4_qkvz", "logs/sweep_eval_v4_qkvz.log",
     "W4 + in_proj_qkv/z -> fp16"),
    ("v3_downproj", "/home/acm/work/models/xpu-variants/v3_downproj", "logs/sweep_eval_v3_downproj.log",
     "OMITTED: 26.3 GiB weights OOM during vLLM load (30.3 GiB card); see v3h dose variant"),
    ("v5_linattn", "/home/acm/work/models/xpu-variants/v5_linattn", "logs/sweep_eval_v5_linattn.log",
     "OMITTED: 26.0 GiB weights OOM during vLLM load; ~ v1+v2+v4 composition"),
    ("fp8_vllm_u", None, "logs/b70_fp8_full3.log",
     "FP8 (vLLM on-the-fly) - 8-bit ceiling"),
    ("v3h_downhalf", "/home/acm/work/models/xpu-variants/v3h_downhalf", "logs/sweep2_eval_v3h_downhalf.log",
     "W4 + down_proj -> fp16 in every other layer (32/64, dose)"),
    ("v7_selfqkv", "/home/acm/work/models/xpu-variants/v7_selfqkv", "logs/sweep2_eval_v7_selfqkv.log",
     "W4 + self_attn q/k/v -> fp16"),
    ("v8_outonly", "/home/acm/work/models/xpu-variants/v8_outonly", "logs/sweep2_eval_v8_outonly.log",
     "W4 + linear_attn.out_proj only -> fp16 (splits v2)"),
    ("v9_oonly", "/home/acm/work/models/xpu-variants/v9_oonly", "logs/sweep2_eval_v9_oonly.log",
     "W4 + self_attn.o_proj only -> fp16 (splits v2)"),
    ("v10_gateupq", "/home/acm/work/models/xpu-variants/v10_gateupq", "logs/sweep2_eval_v10_gateupq.log",
     "OMITTED: gate+up 16-layer dose (22.7 GiB) loads all shards then OOMs in the post-load repack of the fp16 experts"),
    ("v10b_gateupd8", "/home/acm/work/models/xpu-variants/v10b_gateupd8", "logs/sweep2_eval_v10b_gateupd8.log",
     "W4 + mlp gate+up -> fp16 in 8 layers (1/8 dose)"),
]
EXTRA_ROWS = [
    ("w4_rtn", "/home/acm/work/models/xpu-w4-rtn", "logs/b70_w4_full.log",
     "uniform W4g32 (earlier eval cfg: len 8192 / seqs 16 / chunk 4096)"),
]
FP8_TAG = "fp8_vllm_u"


def load_shapes():
    idx = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
    shapes = {}
    for fn in sorted(set(idx.values())):
        with open(f"{BASE}/{fn}", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            shapes[k] = v["shape"]
    return {k: math.prod(v) for k, v in shapes.items() if k.endswith(".weight")}


def is_quantizable(name: str) -> bool:
    return any(p in name for p in QUANTIZABLE)


def effective_bits(ckpt_dir: str | None, shapes: dict):
    extra = {}
    fp8 = ckpt_dir is None
    qbits = None  # per-weight bits = q_bits + 16/group_size (sym, fp16 scale)
    if ckpt_dir:
        cfg = glob.glob(f"{ckpt_dir}/**/config.json", recursive=True)
        if not cfg:
            return None
        qc = json.load(open(cfg[0])).get("quantization_config", {})
        extra = {k: v for k, v in qc.get("extra_config", {}).items()}
        if qc.get("data_type") == "int":
            qbits = qc.get("bits") + 16 / qc.get("group_size", 32)
    tot = qtot = 0.0
    qt_params = 0
    for name, params in shapes.items():
        mod = name[: -len(".weight")]
        if is_quantizable(mod):
            b = FP8_BITS if fp8 else (16 if mod in extra and extra[mod].get("bits") == 16 else qbits)
            tot += b * params
            qtot += b * params
            qt_params += params
        else:
            tot += 16 * params
    return {
        "avg_bits_all": tot / sum(shapes.values()),
        "avg_bits_quantizable": qtot / qt_params if qt_params else None,
    }


def load_boolq(tag: str):
    path = f"results/b70/{tag}/boolq.json"
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        return None
    if "results" not in d:
        return None
    r = d["results"]["boolq"]
    samples = {str(s["doc_id"]): float(s.get("acc", 0.0)) > 0.5
               for s in d["samples"]["boolq"]}
    return {
        "acc": r["acc,none"],
        "stderr": r["acc_stderr,none"],
        "n": len(samples),
        "samples": samples,
        "seconds": d.get("seconds"),
    }


def mcnemar(q: dict, b: dict):
    ids = set(q) & set(b)
    b_only = sum(1 for i in ids if b[i] and not q[i])   # anchor ok, variant wrong
    c_only = sum(1 for i in ids if q[i] and not b[i])   # variant ok, anchor wrong
    if b_only + c_only == 0:
        return b_only, c_only, 1.0
    # exact binomial two-sided on discordant pairs
    k, n = min(b_only, c_only), b_only + c_only
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return b_only, c_only, min(1.0, p)


def log_stats(path: str):
    try:
        txt = open(path, "r", errors="ignore").read()
    except FileNotFoundError:
        return {}
    out = {}
    m = re.search(r"Model loading took ([\d.]+) GiB", txt)
    if m:
        out["weights_gib"] = float(m.group(1))
    speeds = re.findall(r"est\. speed input: ([\d.]+) toks/s", txt)
    if speeds:
        out["input_tok_s"] = float(speeds[-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    shapes = load_shapes()
    fp8 = load_boolq(FP8_TAG)
    if not fp8:
        raise SystemExit(f"FP8 anchor missing: results/b70/{FP8_TAG}/boolq.json")

    rows = []
    for tag, ckpt, log, desc in VARIANTS + EXTRA_ROWS:
        r = load_boolq(tag)
        if r is None:
            rows.append({"tag": tag, "desc": desc, "status": "pending"})
            continue
        eb = effective_bits(ckpt, shapes)
        bo, co, p = mcnemar(r["samples"], fp8["samples"])
        rows.append({
            "tag": tag, "desc": desc, "status": "done",
            "avg_bits_all": eb and round(eb["avg_bits_all"], 3),
            "avg_bits_quantizable": eb and round(eb["avg_bits_quantizable"], 3),
            "boolq_acc": round(r["acc"], 4),
            "boolq_stderr": round(r["stderr"], 4),
            "n": r["n"],
            "delta_vs_fp8_pp": round((r["acc"] - fp8["acc"]) * 100, 2),
            "mcnemar_discordant": f"{bo}/{co}",
            "mcnemar_p": p,
            "seconds": r["seconds"],
            **log_stats(log),
        })

    hdr = (f"{'tag':14s} {'q-bits':>7s} {'all-bits':>8s} {'BoolQ':>7s} {'+/-':>6s} "
           f"{'d vs FP8':>9s} {'discord':>9s} {'p':>10s} {'GiB':>6s} {'tok/s':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        if row["status"] != "done":
            print(f"{row['tag']:14s} ... pending ({row['desc']})")
            continue
        p = row["mcnemar_p"]
        pstr = f"{p:.2e}" if p < 0.01 else f"{p:.3f}"
        print(f"{row['tag']:14s} {row['avg_bits_quantizable']:7.3f} {row['avg_bits_all']:8.3f} "
              f"{row['boolq_acc']*100:6.2f}% {row['boolq_stderr']*100:5.2f}% "
              f"{row['delta_vs_fp8_pp']:+8.2f}p {row['mcnemar_discordant']:>9s} {pstr:>10s} "
              f"{row.get('weights_gib', float('nan')):6.2f} {row.get('input_tok_s', float('nan')):7.0f}")
    print(f"\nFP8 anchor: {fp8['acc']*100:.2f}% +/- {fp8['stderr']*100:.2f} (n={fp8['n']})")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"fp8_anchor": {k: fp8[k] for k in ("acc", "stderr", "n")},
                       "rows": rows}, f, indent=2)
        print(f"json -> {args.json}")


if __name__ == "__main__":
    main()
