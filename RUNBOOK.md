# Recovery Rate verification — runbook

Goal: independently measure Recovery Rate = quantized / bf16-base for
`Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound` on a single RTX 6000 Ada (48 GB).

## Environment
    source scripts/env.sh          # sets CUDA_HOME (required!), model paths, venv

`CUDA_HOME=/usr/local/cuda` is mandatory: without it flashinfer's JIT cannot
find `cuda_runtime.h` and engine init dies.

## Always free the GPU between runs
    ./scripts/gpu_free.sh

vLLM leaves an orphaned `VLLM::EngineCore` child holding ~40 GB when the parent
is killed. `gpu_free.sh` reaps it and waits for memory to drop.

## Models
- base  : /home/acm/work/models/Qwen3.8-27B  (27.78 B params, ~51.8 GiB bf16)
- quant : /home/acm/work/models/Qwen3.8-27B-MixedInt4-AutoRound (~19.4 GiB)

## Evaluation driver
    scripts/run_eval.py <base|quant> --output results/<dir> [options]

Task groups and few-shot counts (both models always identical):

| run           | tasks                | num_fewshot |
|---------------|----------------------|-------------|
| mmlu          | mmlu (+4 categories) | 5           |
| gsm8k         | gsm8k                | 5           |
| arc_challenge | arc_challenge        | 25          |
| hellaswag     | hellaswag            | 10          |
| boolq         | boolq                | 0           |
| piqa          | piqa                 | 0           |
| winogrande    | winogrande           | 0           |

Useful options:
- `--limit N`            cap docs per task (debug / subset)
- `--offload-gb N`       base only; CPU weight offload
- `--max-num-batched-tokens N`  lower it if you hit OOM during warmup
- `--enable-thinking`    default|true|false (loglikelihood tasks reject true)
- `--apply-chat-template` for chat-style scoring
- `--only a,b`           run a subset of the groups above

## Analysis
    scripts/report.py --base results/base --quant results/quant

Prints base, quant, Δ pp, Recovery %, with stderr, and flags |Δ| < stderr as
"within noise". Metric keys and the 7-benchmark average match the model card
(validated against synthetic data: base average reproduces 77.45%).

## Measured performance (quant model)
- prefill, no prefix sharing : ~2300 tok/s
- prefill, shared prefix     : ~6100 tok/s  (prefix caching works)
- decode (8 seqs)            : ~150 tok/s aggregate

Multiple-choice loglikelihood is prefill-bound and affordable. Generation
(GSM8K with thinking) is the scarce resource.

# B70 mixed-precision sweep — runbook (Intel Arc Pro B70, 30.3 GiB)

The sweep (W4-family variants + FP8 arm, full 3270-doc BoolQ) runs on the
B70 **only**. The RTX 6000 Ada is not used for any sweep number.

## Environment
    source scripts/env_xpu.sh   # .venv-xpu (auto-round), .venv-vllm-xpu (vLLM-XPU)

`env_xpu.sh` pins `ONEAPI_DEVICE_SELECTOR=level_zero:0`, sets
`CUDA_VISIBLE_DEVICES=""` (harmless: see platform patch below) and
`VLLM_DISABLE_CUDA_PLATFORM=1`.

## Two load-bearing local patches in `.venv-vllm-xpu`

1. `vllm/platforms/__init__.py` — `cuda_platform_plugin()` returns `None`
   when `VLLM_DISABLE_CUDA_PLATFORM=1`. vLLM-XPU 0.29.0 uses vendored pynvml
   for CUDA detection and **ignores** `CUDA_VISIBLE_DEVICES`, so with both
   GPUs present it aborts with "Only one platform plugin can be activated:
   ['cuda','xpu']".
2. `vllm/utils/mem_utils.py` — `MemorySnapshot.measure()` falls back to
   `torch.xpu.mem_get_info()` when XPU free memory reads 0. After
   `import vllm`, `torch.accelerator.get_memory_info(xpu)` wrongly reports
   0 free (native `torch.xpu.mem_get_info()` is correct) and engine init
   dies on "Free memory on device (0.0 GiB) is less than desired".

Both patches must be reapplied if `.venv-vllm-xpu` is recreated.

## Uniform BoolQ eval config (every sweep point)
    scripts/eval_b70.sh <tag>                     # custom checkpoint
    MODEL_KIND=base_fp8 MODEL_PATH=base scripts/eval_b70.sh <tag>

One config for all points so numbers are directly comparable:
`max_model_len 4096` (BoolQ's longest prompt is ~730 tok — this changes no
actual processing, it only relaxes vLLM's init-time KV capacity check),
`--kv-cache-memory 800000000` (skips vLLM memory profiling, which the 28 GiB
FP8 weights cannot afford), `max_num_batched_tokens 768` (bounds the
per-chunk logits matrix at 768 x 151936 x 4B ≈ 0.45 GiB), `max_num_seqs 8`.

The 768/8 batching matters: lm-eval sorts loglikelihood requests
**longest-first**, so a full run hits peak memory pressure immediately —
the 20-doc smoke passes while the full set OOMs
(`UR_RESULT_ERROR_OUT_OF_RESOURCES` in `mamba_hybrid.py add_request`).

## Quantization (B70)
Uniform W4g32 RTN (the anchor, `/home/acm/work/models/xpu-w4-rtn`):
    scripts/quant_rtn_xpu.sh         # BITS=4 OUT=... ; --disable_opt_rtn is required

Variant = anchor + one module group promoted to BF16:
    scripts/b70_sweep.sh             # runs anchor re-eval + v1..v5 sequentially
    # per-variant: --layer_config '{"linear_attn.out_proj":"BF16",...}'
    #   (keys are substrings/regex; "BF16" = keep module fp16)

Each variant is verified after quantization (promoted module count in
`extra_config` must match) before its eval is allowed to run.

## B70-specific findings
- Each variant checkpoint is 12-26 GiB, so `xpu-variants/` grows by >200 GiB
  over a sweep. `/` filled transiently once (v10 quantize died with
  `No space left on device`); check `df -h` before a sweep and prune
  `xpu-variants/` afterwards.
- `--disable_opt_rtn` is required: the default opt_rtn path crashes on XPU
  (torch-xpu-ops "index out of bounds" SYCL assertion).
- The model-free RTN route (what `--disable_opt_rtn` selects) quantizes **all**
  504 supported linear modules, including `in_proj_a/b` + `mtp.fc`; the
  opt_rtn route (CUDA) auto-keeps those in fp16 (shape-divisibility rule).
  The two "W4 RTN" checkpoints therefore differ in coverage — 79.39% (Ada,
  vLLM 0.30.0 CUDA) vs 83.73% (B70, vLLM-XPU 0.29.0) is a
  coverage + eval-stack difference, not pure hardware.
- vLLM-XPU cannot load int8 weights (XPU INC supports int bits {2,4} only),
  so the shipped checkpoint (17 int8 modules) cannot be evaluated on the
  B70; BF16 is the promotion target for variants and vLLM on-the-fly FP8
  (`--quantization fp8`, 27.99 GiB) is the 8-bit ceiling.
- AutoRound's `--scheme FP8_BLOCK` export (packing `auto_round:fp8`) fails to
  load in vLLM ("start (1) + length (1) exceeds dimension size (1)") — do not
  use it; the on-the-fly FP8 arm replaces it.
- AutoRound `--layer_config` values: scheme preset strings ("BF16") or dicts
  (`{"bits":16,"data_type":"fp"}`); keys are exact names, substrings, or
  regex, expanded against all supported layers.

## Analysis
    scripts/b70_sweep_report.py       # bits vs BoolQ table + McNemar vs FP8
    scripts/mcnemar.py --quant A --base B   # paired test, any two BoolQ runs

## Measured B70 performance (BoolQ, uniform config, chunk 768 / 8 seqs)
- W4 checkpoint   : ~1630 tok/s input (18.5 GiB weights)
- FP8 on-the-fly  : ~1420 tok/s input (28.0 GiB weights)
- engine init     : ~70 s (model load ~14-55 s + KV/warmup)
- full 3270-doc BoolQ (6540 requests): ~11.5 min

