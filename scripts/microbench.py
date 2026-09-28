#!/usr/bin/env python
"""Microbenchmark: prefill throughput, decode throughput, and prefix-cache reuse.

Answers two questions:
  1. How many prompt tokens/s can this model prefill (loglikelihood is prefill-bound)?
  2. Do requests sharing a long prefix benefit from prefix caching (4x for MMLU)?
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "/home/acm/work/models/Qwen3.8-27B-MixedInt4-AutoRound"


def make_prompt(tok, n_tokens, filler_word="the"):
    # build a prompt of approximately n_tokens
    body = (filler_word + " ") * n_tokens
    return body


def bench(llm, prompts, sampling, label, warm=False):
    t0 = time.time()
    outs = llm.generate(prompts, sampling, use_tqdm=False)
    dt = time.time() - t0
    n_in = sum(len(o.prompt_token_ids) for o in outs)
    n_out = sum(len(o.outputs[0].token_ids) for o in outs)
    print(
        f"{label:38s} prompts={len(prompts):4d} in_tok={n_in:8d} out_tok={n_out:6d} "
        f"time={dt:7.2f}s  prefill={n_in/dt:8.1f} tok/s  decode={n_out/dt:7.1f} tok/s",
        flush=True,
    )
    return dt, n_in, n_out


def main():
    llm = LLM(
        model=MODEL,
        quantization="inc",
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        max_num_seqs=32,
        max_num_batched_tokens=16384,
        disable_log_stats=True,
    )
    tok = llm.get_tokenizer()
    sp1 = SamplingParams(max_tokens=1, temperature=0.0)
    sp64 = SamplingParams(max_tokens=64, temperature=0.0)

    # ~1200-token prompts, distinct (no shared prefix)
    base = make_prompt(tok, 1200)
    distinct = [f"id{i} " + base for i in range(64)]
    # shared long prefix, differing only at the end (mirrors MMLU A/B/C/D choices)
    shared = [base + f"\nAnswer: {c}" for c in ["A", "B", "C", "D"]] * 16
    # same shared prompt repeated (best-case cache reuse)
    identical = [base + "\nAnswer: A"] * 64

    # warmup
    llm.generate(["warmup " + base[:200]], sp1, use_tqdm=False)

    bench(llm, distinct, sp1, "distinct 1200tok, 1 out tok")
    bench(llm, shared, sp1, "shared-prefix (MMLU-like), 1 out tok")
    bench(llm, identical, sp1, "identical prompts (cache best case), 1 tok")
    bench(llm, [base + " Explain briefly."] * 8, sp64, "decode throughput, 64 out tok")


if __name__ == "__main__":
    main()
