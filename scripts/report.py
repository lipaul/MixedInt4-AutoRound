#!/usr/bin/env python
"""Compute Recovery Rate (quant / base) from lm-eval result JSONs.

Reads results/base/*.json and results/quant/*.json (as produced by
scripts/run_eval.py) and prints a comparison table with standard errors.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# benchmark -> (result key, metric name, filter)
# Values mirror the model card's choice of metric.
BENCH = {
    "mmlu": ("mmlu", "acc", None),
    "gsm8k_flex": ("gsm8k", "exact_match", "flexible-extract"),
    "gsm8k_strict": ("gsm8k", "exact_match", "strict-match"),
    "arc_challenge": ("arc_challenge", "acc_norm", None),
    "boolq": ("boolq", "acc", None),
    "hellaswag": ("hellaswag", "acc_norm", None),
    "piqa": ("piqa", "acc_norm", None),
    "winogrande": ("winogrande", "acc", None),
}

CATEGORIES = {
    "mmlu_humanities": "acc",
    "mmlu_other": "acc",
    "mmlu_social_sciences": "acc",
    "mmlu_stem": "acc",
}


def load_model_dir(d: Path) -> dict:
    """Merge every run file in a directory into {task_key: results_dict}."""
    merged: dict = {}
    for f in sorted(d.glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        merged.update(payload.get("results", {}))
    return merged


def pick(results: dict, task: str, metric: str, filt: str | None):
    """Return (value, stderr) for a metric, tolerating lm-eval key suffixes."""
    if task not in results:
        return None, None
    row = results[task]
    cands = [k for k in row if k.split(",")[0] == metric]
    if filt is not None:
        cands = [k for k in cands if len(k.split(",")) > 1 and k.split(",")[1] == filt]
    else:
        # prefer the ",none" filter when present
        none_c = [k for k in cands if k.endswith(",none")]
        if none_c:
            cands = none_c
    for key in cands:
        val = row.get(key)
        err = row.get(f"{metric}_stderr,{key.split(',', 1)[1]}" if "," in key else f"{metric}_stderr")
        if val is not None:
            return val, err
    return None, None


def fmt(v, e):
    if v is None:
        return "   n/a  "
    s = f"{v*100:6.2f}"
    if e is not None:
        s += f"±{e*100:.2f}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/base")
    ap.add_argument("--quant", default="results/quant")
    args = ap.parse_args()

    base = load_model_dir(Path(args.base))
    quant = load_model_dir(Path(args.quant))

    rows = []
    print(f"\n{'Benchmark':<16} {'Base %':>12} {'Quant %':>12} {'Δ pp':>8} {'Recovery':>10}")
    print("-" * 64)
    recs = []
    for label, (task, metric, filt) in BENCH.items():
        b, be = pick(base, task, metric, filt)
        q, qe = pick(quant, task, metric, filt)
        if b is None or q is None:
            print(f"{label:<16} {'-.--':>12} {'-.--':>12} {'':>8} {'n/a':>10}")
            continue
        d = (q - b) * 100
        rec = (q / b) * 100 if b else float("nan")
        sig = ""
        if be is not None and qe is not None:
            se = math.sqrt(be**2 + qe**2) * 100
            sig = "  (within noise)" if abs(d) < se else "  (*)"
        print(f"{label:<16} {fmt(b,be):>12} {fmt(q,qe):>12} {d:>+8.2f} {rec:>9.2f}%{sig}")
        recs.append((label, b, q, be, qe))

    if recs:
        # unweighted mean over the card's 7 core benchmarks
        core = [r for r in recs if r[0] in
                ("mmlu", "gsm8k_flex", "arc_challenge", "boolq", "hellaswag", "piqa", "winogrande")]
        bavg = sum(r[1] for r in core) / len(core)
        qavg = sum(r[2] for r in core) / len(core)
        print("-" * 64)
        print(f"{'AVERAGE(7)':<16} {bavg*100:>12.2f} {qavg*100:>12.2f} "
              f"{(qavg-bavg)*100:>+8.2f} {qavg/bavg*100:>9.2f}%")

    print(f"\n{'MMLU category':<20} {'Base %':>12} {'Quant %':>12} {'Δ pp':>8} {'Recovery':>10}")
    print("-" * 64)
    for task, metric in CATEGORIES.items():
        b, be = pick(base, task, metric, None)
        q, qe = pick(quant, task, metric, None)
        if b is None or q is None:
            print(f"{task:<20} {'-.--':>12} {'-.--':>12}")
            continue
        d = (q - b) * 100
        rec = (q / b) * 100 if b else float("nan")
        print(f"{task:<20} {fmt(b,be):>12} {fmt(q,qe):>12} {d:>+8.2f} {rec:>9.2f}%")

    print("\n(± = stderr; * marks |Δ| greater than the combined standard error)")


if __name__ == "__main__":
    main()
