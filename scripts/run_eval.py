#!/usr/bin/env python
"""Run the Recovery-Rate benchmark set for one model with lm-eval in-process vLLM.

Loads the model once, then evaluates each task group with its conventional
few-shot count. Results (including per-sample logs) are written to --output.

Examples:
  scripts/run_eval.py quant --output results/quant
  scripts/run_eval.py base  --output results/base --offload-gb 18
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# CUDA_HOME is required for flashinfer's JIT to find cuda_runtime.h.
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

import lm_eval  # noqa: E402
from lm_eval.api.registry import get_model  # noqa: E402
from lm_eval.tasks import TaskManager  # noqa: E402

BASE_MODEL = "/home/acm/work/models/Qwen3.8-27B"
QUANT_MODEL = "/home/acm/work/models/Qwen3.8-27B-MixedInt4-AutoRound"

# name -> (tasks, num_fewshot, apply_chat_template, limit)
# Few-shot counts follow the conventional lm-eval protocol for each benchmark.
# Multiple-choice tasks are scored as plain continuations (no chat template);
# GSM8K is generative and needs the model's chat template.
#
# The limits are a cost control, not a methodological choice: this hybrid model
# runs at ~1.7k tok/s under the 300W cap, so the uncapped set would take ~45h
# per model. Each cap is applied identically to both models. None = full task.
TASK_RUNS = [
    # cheap -> expensive, so partial results accumulate quickly
    ("piqa", ["piqa"], 0, False, None),
    ("winogrande", ["winogrande"], 0, False, None),
    ("boolq", ["boolq"], 0, False, 500),
    ("hellaswag", ["hellaswag"], 10, False, 400),
    ("arc_challenge", ["arc_challenge"], 25, False, 250),
    ("mmlu", ["mmlu"], 5, False, 20),
    # GSM8K last: long thinking generations make it the bottleneck
    ("gsm8k", ["gsm8k"], 5, True, 250),
]


def build_model(model_key: str, args):
    is_quant = model_key == "quant"
    pretrained = QUANT_MODEL if is_quant else BASE_MODEL
    if model_key == "custom":
        pretrained = args.model_path
    model_args = {
        "pretrained": pretrained,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_mem,
        "seed": args.seed,
        "max_num_seqs": args.max_num_seqs,
        "enable_thinking": args.enable_thinking,
    }
    if is_quant or model_key == "custom":
        model_args["quantization"] = args.quantization
    if model_key == "base_fp8":
        # FP8 stand-in for the bf16 baseline: fits on the 48GB card without
        # CPU offload. Near-lossless, but a different denominator from the
        # card's bf16 baseline (bounded separately by the bf16 drift check).
        model_args["quantization"] = "fp8"
    if args.offload_gb:
        model_args["cpu_offload_gb"] = args.offload_gb
    if args.max_num_batched_tokens:
        model_args["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.kv_cache_memory:
        model_args["kv_cache_memory_bytes"] = args.kv_cache_memory
    print(f"[eval] loading model: {model_args['pretrained']}", flush=True)
    print(f"[eval] model_args: {model_args}", flush=True)
    lm = get_model("vllm").create_from_arg_string(
        "",
        {k: v for k, v in model_args.items() if v is not None},
    )
    return lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["base", "base_fp8", "quant", "custom"])
    ap.add_argument("--model-path", default=None,
                    help="checkpoint path for --model custom")
    ap.add_argument("--quantization", default="inc",
                    help="vLLM quantization for custom checkpoints (default: inc)")
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--limit", type=int, default=None, help="cap docs per task (debug)")
    ap.add_argument("--only", default=None, help="comma-separated subset of run names")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    ap.add_argument("--kv-cache-memory", type=int, default=None,
                    help="explicit KV cache bytes; skips vLLM memory profiling")
    ap.add_argument("--offload-gb", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-gen-toks", type=int, default=4096)
    ap.add_argument(
        "--apply-chat-template",
        choices=["false", "true"],
        default="false",
        help="apply the model chat template for loglikelihood tasks",
    )
    ap.add_argument(
        "--enable-thinking",
        choices=["default", "true", "false"],
        default="default",
        help="passed to the chat template; default leaves the model default (thinking ON)",
    )
    ap.add_argument("--tag", default="", help="suffix appended to result filenames")
    args = ap.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    enable_thinking = {
        "default": None,
        "true": True,
        "false": False,
    }[args.enable_thinking]
    args.enable_thinking = enable_thinking

    apply_ct = args.apply_chat_template == "true"  # kept for CLI back-compat
    runs = TASK_RUNS
    if args.only:
        keep = set(args.only.split(","))
        runs = [r for r in runs if r[0] in keep]

    lm = build_model(args.model, args)
    task_manager = TaskManager()

    summary = {}
    failed = []
    for name, tasks, nshot, run_chat, run_limit in runs:
        limit = args.limit if args.limit is not None else run_limit
        print(
            f"\n[eval] === {name} (tasks={tasks}, num_fewshot={nshot}, "
            f"chat={run_chat}, limit={limit}) ===",
            flush=True,
        )
        # Remove any stale result from an earlier run in this output dir so a
        # failed task can never be mistaken for a fresh result.
        fn = outdir / f"{name}{('_' + args.tag) if args.tag else ''}.json"
        fn.unlink(missing_ok=True)
        t0 = time.time()
        gen_kwargs = None
        if name == "gsm8k":
            gen_kwargs = {"max_gen_toks": args.max_gen_toks, "temperature": 0.0}
        try:
            res = lm_eval.simple_evaluate(
                model=lm,
                tasks=tasks,
                num_fewshot=nshot,
                batch_size="auto",
                apply_chat_template=run_chat,
                fewshot_as_multiturn=run_chat,
                gen_kwargs=gen_kwargs,
                log_samples=True,
                limit=limit,
                task_manager=task_manager,
                random_seed=args.seed,
                numpy_random_seed=args.seed,
                torch_random_seed=args.seed,
                fewshot_random_seed=args.seed,
            )
        except Exception:
            print(f"[eval] {name} FAILED:\n{traceback.format_exc()}", flush=True)
            summary[name] = {"error": traceback.format_exc()}
            failed.append(name)
            continue

        dt = time.time() - t0
        fn = outdir / f"{name}{('_' + args.tag) if args.tag else ''}.json"
        with open(fn, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "run": name,
                    "tasks": tasks,
                    "num_fewshot": nshot,
                    "apply_chat_template": run_chat,
                    "enable_thinking": args.enable_thinking,
                    "max_gen_toks": args.max_gen_toks,
                    "seed": args.seed,
                    "limit": limit,
                    "seconds": dt,
                    "results": res["results"],
                    "configs": {k: v for k, v in res.get("configs", {}).items()},
                    "samples": res.get("samples", {}),
                },
                f,
                default=str,
            )
        summary[name] = {
            "seconds": dt,
            "results": {k: v for k, v in res["results"].items() if not k.startswith("mmlu_") or True},
            "file": str(fn),
        }
        print(f"[eval] {name} done in {dt/60:.1f} min -> {fn}", flush=True)

    with open(outdir / f"_summary{('_' + args.tag) if args.tag else ''}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    if failed:
        print(f"\n[eval] FINISHED WITH FAILURES: {failed}", flush=True)
        return 1
    print(f"\n[eval] all done. summary -> {outdir}/_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
