#!/usr/bin/env python
"""Summarise BoolQ accuracy across precision variants.

Answers: does increasing bit width recover BoolQ?
"""
from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import binomtest

VARIANTS = [
    # (label, path, effective weight precision)
    ("BF16 base (card)", None, "16-bit"),
    ("FP8 base (measured)", "results/boolq_full/base/boolq.json", "8-bit fp8"),
    ("Shipped AutoRound W4", "results/boolq_full/quant/boolq.json", "4-bit mixed"),
    ("RTN W8", "results/boolq_custom/w8_rtn/boolq.json", "8-bit int"),
    ("RTN W4", "results/boolq_custom/w4_rtn/boolq.json", "4-bit int"),
]
CARD_BF16_BASE = 0.8664


def load(path: str) -> dict:
    payload = json.loads(Path(path).read_text())
    task = payload["run"]
    return {str(s["doc_id"]): float(s.get("acc", 0.0)) > 0.5
            for s in payload["samples"][task]}


def main():
    ref = None
    ref_label = None
    for label, path, _ in VARIANTS:
        if label.startswith("FP8") and path and Path(path).exists():
            ref = load(path)
            ref_label = label
            break

    print(f"{'variant':<22} {'precision':<12} {'BoolQ acc':>10} {'stderr':>8} {'vs ' + (ref_label or 'base'):>22}")
    print("-" * 80)
    for label, path, prec in VARIANTS:
        if path is None:
            print(f"{label:<22} {prec:<12} {CARD_BF16_BASE*100:>9.2f}% {'':>8} {'(published)':>22}")
            continue
        if not Path(path).exists():
            print(f"{label:<22} {prec:<12} {'n/a':>10} {'':>8} {'(not run)':>22}")
            continue
        d = load(path)
        n = len(d)
        acc = sum(d.values()) / n
        # binomial stderr
        import math
        se = math.sqrt(acc * (1 - acc) / n)
        extra = ""
        if ref and set(d) & set(ref):
            ids = set(d) & set(ref)
            b = sum(1 for i in ids if ref[i] and not d[i])
            c = sum(1 for i in ids if d[i] and not ref[i])
            if b + c:
                p = binomtest(b, b + c, 0.5).pvalue
                extra = f"p={p:.2g} ({b} vs {c})"
            else:
                extra = "no discordant"
        print(f"{label:<22} {prec:<12} {acc*100:>9.2f}% {se*100:>7.2f} {extra:>22}")
    print()
    print("Interpretation: if RTN W8 lands near the FP8/BF16 base and well above")
    print("RTN W4, then bit width - not the AutoRound recipe - is what BoolQ is")
    print("missing; if RTN W4 ~ shipped W4 and RTN W8 does not recover, the loss")
    print("is not purely a weight-precision effect.")


if __name__ == "__main__":
    main()
