#!/usr/bin/env python
"""Paired analysis of two BoolQ runs (same docs, same order-independent ids).

Unlike comparing two independent accuracies via stderr, pairing the per-question
outcomes gives a McNemar test, which is far more sensitive to a real regression.
Also stratifies by passage length, since BoolQ degrades are often long-context.

Usage:
  scripts/mcnemar.py --quant results/boolq_full/quant/boolq.json \
                     --base  results/boolq_full/base/boolq.json
"""
from __future__ import annotations

import argparse
import json

from scipy.stats import binomtest, chi2


def load(path: str) -> dict:
    """doc_id -> {correct: bool, len: int}"""
    payload = json.loads(open(path).read())
    task = payload["run"]
    out = {}
    for s in payload["samples"][task]:
        doc = s["doc"]
        passage = doc.get("passage", "") or ""
        # acc for boolq is 1.0/0.0
        correct = float(s.get("acc", 0.0)) > 0.5
        out[str(s["doc_id"])] = {"correct": correct, "len": len(passage)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", required=True)
    ap.add_argument("--base", required=True)
    args = ap.parse_args()

    q = load(args.quant)
    b = load(args.base)
    ids = sorted(set(q) & set(b), key=lambda x: int(x) if x.isdigit() else x)
    if not ids:
        raise SystemExit("no overlapping doc_ids")

    a = b_ = c = d = 0
    for i in ids:
        qc, bc = q[i]["correct"], b[i]["correct"]
        if qc and bc:
            a += 1
        elif bc and not qc:
            b_ += 1
        elif qc and not bc:
            c += 1
        else:
            d += 1

    n = len(ids)
    qacc = (a + c) / n
    bacc = (a + b_) / n

    print(f"paired docs           : {n}")
    print(f"base   acc            : {bacc*100:.2f}%")
    print(f"quant  acc            : {qacc*100:.2f}%")
    print(f"delta                 : {(qacc-bacc)*100:+.2f} pp")
    print()
    print("2x2 (rows=base, cols=quant):")
    print(f"  both correct        : {a}")
    print(f"  base ok, quant wrong: {b_}")
    print(f"  base wrong, quant ok: {c}")
    print(f"  both wrong          : {d}")
    print()
    # McNemar
    if b_ + c > 0:
        stat = (abs(b_ - c) - 1) ** 2 / (b_ + c)
        p_chi = chi2.sf(stat, df=1)
        p_exact = binomtest(b_, b_ + c, 0.5).pvalue
        print(f"McNemar chi2 (continuity-corrected) = {stat:.3f}, p = {p_chi:.4g}")
        print(f"exact binomial test on discordant pairs ({b_} vs {c}): p = {p_exact:.4g}")
        print(f"  -> regression is {'SIGNIFICANT' if p_exact < 0.05 else 'not significant'} at 0.05")
    else:
        print("no discordant pairs")
    print()

    # stratify by passage length quartiles
    lens = sorted(x["len"] for x in (q[i] for i in ids))
    cuts = [lens[int(len(lens) * f)] for f in (0.25, 0.5, 0.75)]
    print(f"passage-length quartile cuts (chars): {cuts}")
    print(f"{'bucket':<12} {'n':>6} {'base%':>8} {'quant%':>8} {'delta pp':>9} {'b ok/q bad':>11} {'b bad/q ok':>11}")
    buckets = [("Q1 short", 0, cuts[0]), ("Q2", cuts[0], cuts[1]),
               ("Q3", cuts[1], cuts[2]), ("Q4 long", cuts[2], 10**9)]
    for name, lo, hi in buckets:
        sub = [i for i in ids if lo <= q[i]["len"] < hi]
        if not sub:
            continue
        bb = sum(b[i]["correct"] for i in sub)
        qq = sum(q[i]["correct"] for i in sub)
        boqb = sum(1 for i in sub if b[i]["correct"] and not q[i]["correct"])
        bboq = sum(1 for i in sub if q[i]["correct"] and not b[i]["correct"])
        print(f"{name:<12} {len(sub):>6} {bb/len(sub)*100:>7.2f} {qq/len(sub)*100:>7.2f} "
              f"{(qq-bb)/len(sub)*100:>+8.2f} {boqb:>11} {bboq:>11}")


if __name__ == "__main__":
    main()
