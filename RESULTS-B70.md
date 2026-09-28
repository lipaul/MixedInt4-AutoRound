# B70 mixed-precision sweep — results

Companion to `RESULTS.md` (the Recovery-Rate verification on the RTX 6000 Ada).
This document covers the follow-up question: **which minimal mixed-precision
allocation fixes the BoolQ regression**, answered with a bit-width /
module-level sweep run entirely on the Intel Arc Pro B70 (30.3 GiB) — per the
project constraint, no sweep number comes from the Ada.

## The question

`RESULTS.md` established (all n=3270 BoolQ, Ada stack):

| model | BoolQ | stack |
|---|---|---|
| card's bf16 base | 86.64% | (card's number) |
| FP8 base (W8A8 dyn) | 87.89% ± 0.57 | vLLM 0.30.0 CUDA |
| shipped mixed quant | 80.28% ± 0.70 | vLLM 0.30.0 CUDA |
| RTN W4 int (opt_rtn coverage) | 79.39% | vLLM 0.30.0 CUDA |
| RTN W8 int | 86.45% | vLLM 0.30.0 CUDA |

Bit width — not the AutoRound recipe — causes the BoolQ regression, and
uniform 8-bit closes it. The sweep asks: **how few weights need to leave int4,
and from which module types, to recover BoolQ?**

## Stack (see RUNBOOK.md "B70" section for full detail)

- vLLM-XPU 0.29.0 (`.venv-vllm-xpu`) + **two local patches that are
  load-bearing**: (1) `VLLM_DISABLE_CUDA_PLATFORM=1` platform guard —
  vLLM-XPU's vendored pynvml CUDA probe ignores `CUDA_VISIBLE_DEVICES` and
  aborts with two GPUs visible; (2) XPU free-memory fallback in
  `MemorySnapshot.measure()` — after `import vllm`,
  `torch.accelerator.get_memory_info(xpu)` reports 0 free and engine init
  dies.
- AutoRound 0.15.1 in `.venv-xpu`, model-free RTN route
  (`--disable_opt_rtn` is mandatory on XPU: the opt_rtn path crashes with a
  torch-xpu-ops SYCL assertion).
- vLLM-XPU INC loads only int bits {2,4}; **int8 cannot be evaluated on the
  B70 at all** (so the shipped checkpoint, which has 17 int8 modules, is not
  B70-evaluable). Consequences for the design:
  - BF16 is the promotion target for mixed variants (not int8);
  - the 8-bit ceiling is vLLM **on-the-fly FP8** of the bf16 base
    (`--quantization fp8`, 27.99 GiB loaded);
  - AutoRound's `--scheme FP8_BLOCK` export is unusable here (its checkpoint
    fails to load in vLLM: "start (1) + length (1) exceeds dimension size (1)").

### Uniform eval config (every sweep point, `scripts/eval_b70.sh`)

`max_model_len 4096` · `max_num_seqs 8` · `max_num_batched_tokens 768` ·
`--kv-cache-memory 800000000` · seed 1234, full 3270-doc BoolQ.

Rationale: the FP8 arm's 28 GiB of weights leave ~1.3 GiB of headroom, so
vLLM's memory profiling and large chunks are unaffordable. 768-token chunks
bound the per-chunk logits matrix (768 x 151936 x 4B ≈ 0.45 GiB); lm-eval sorts
loglikelihood requests **longest-first**, so a full run hits peak memory
pressure immediately — a 20-doc smoke passes while the full set OOMs
(`UR_RESULT_ERROR_OUT_OF_RESOURCES` in `mamba_hybrid.py add_request`).
4096 max_model_len changes no actual processing (BoolQ's longest prompt is
~730 tokens); it only relaxes vLLM's init-time KV capacity check.

**Config sensitivity is a non-issue**: the W4 anchor measured 83.73% ± 0.65
under the earlier config (len 8192 / seqs 16 / chunk 4096) and 83.64% ± 0.65
under the uniform config — 0.09 pp apart, well within stderr.

## Variant design

Base recipe for every W4-family variant: uniform W4g32 symmetric RTN over all
504 supported linear modules (identical to the `xpu-w4-rtn` anchor), plus
exactly one module group promoted to BF16 via `--layer_config` (values
`"BF16"`, keys are substrings/regex). Each variant is verified after
quantization (promoted module count in `extra_config`) before its eval runs.

| tag | promoted group (→ BF16) | modules | params | eff. bits* |
|---|---|---|---|---|
| w2_rtn | — (uniform W2g32) | 504 | 24.8 B | 2.50 |
| w4_rtn_u | — (uniform W4g32) | 0 | 0 | 4.50 |
| v1_inprojab | `in_proj_a` + `in_proj_b` + `mtp.fc` | 97 | 76 M | 4.54 |
| v2_outproj | `linear_attn.out_proj` + `self_attn.o_proj` | 65 | 2.05 B | 5.44 |
| v4_qkvz | `in_proj_qkv` + `in_proj_z` | 96 | 4.03 B | 6.37 |
| v5_linattn | all `linear_attn` | 240 | 5.56 B | 6.68 |
| v3_downproj | `mlp.down_proj` | 65 | 5.79 B | 6.76 |
| v3h_downhalf | `mlp.down_proj`, every other layer | 32 | 2.90 B | 5.85 |
| v7_selfqkv | `self_attn` q/k/v | 51 | 1.25 B | 5.08 |
| v8_outonly | `linear_attn.out_proj` only | 48 | 1.51 B | 5.20 |
| v9_oonly | `self_attn.o_proj` only | 17 | 0.53 B | 4.75 |
| v10_gateupq | `mlp` gate+up, 16 layers (¼ dose) | 32 | 2.90 B | 5.85 |
| fp8_vllm_u | — (on-the-fly FP8) | 504 | 24.8 B | 8.01 |

*effective bits = param-weighted mean over quantizable weights
(int4g32 = 4 + 16/32 = 4.5 bits/w; int2g32 = 2.5; fp16 = 16;
fp8 + fp32 per-channel scale ≈ 8.008).

**The 30.3 GiB card caps promotions at ~4 GiB of fp16**: vLLM weight
*loading* OOMs above ~23.9 GiB of weights (v4 at 23.91 GiB loads; v3/v5 at
26.3/26.0 GiB die during load with `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`).
The two large groups — `mlp.down_proj` (5.79 B params) and all `linear_attn`
(5.56 B) — therefore cannot be promoted in full and are probed with
dose-response (`v3h` = 50% of down layers) or composition (v5 ≈ v1+v2+v4)
instead; `mlp` gate+up (11.6 B) gets a quarter dose (`v10`). `v8`/`v9` split
v2's out/o attribution.

## Anchors (B70 stack)

| model | BoolQ (n=3270) | weights | throughput |
|---|---|---|---|
| FP8 (on-the-fly) | **88.23% ± 0.56** | 27.99 GiB | ~1420 tok/s |
| W4 uniform (earlier cfg) | 83.73% ± 0.65 | 18.52 GiB | ~1630 tok/s |
| W4 uniform (uniform cfg) | 83.64% ± 0.65 | 18.52 GiB | ~700 tok/s |
| W2 uniform (floor point) | **75.78% ± 0.75** | 12.84 GiB | ~730 tok/s |

The floor point shows the curve is steep at the low end: 2.5 → 4.5 effective
bits buys +7.9 pp; 4.5 → 8.0 buys +4.6 pp. W2 vs FP8 is 548/141 discordant
(McNemar p ≈ 1.7e-57).

Cross-stack consistency: B70 FP8 88.23% vs Ada FP8 87.89% — the two stacks
agree at 8 bits (Δ 0.34 pp, within one stderr). The W4 comparison across
stacks is *not* apples-to-apples: the Ada "W4 RTN" ran the opt_rtn route,
which auto-keeps `in_proj_a/b` + `mtp.fc` in fp16 (shape-divisibility rule),
while the B70 model-free route quantizes all 504 modules. `v1_inprojab`
replicates the Ada coverage on the B70 stack and settles it: **the coverage
difference is irrelevant** (v1 = 83.70% ± 0.65 vs uniform W4 83.64% ± 0.65,
+0.06 pp; the Ada-vs-B70 W4 delta of ~4.3 pp is an eval-stack effect, not a
checkpoint-coverage effect). It also shows the shipped model's own fp16
keep-set (`in_proj_a/b` + `mtp.fc`, 76 M params, +0.1 GiB) buys **no BoolQ
recovery** — the remaining −4.6 pp lives elsewhere (v2/v4/v3/v5 locate it).

## Where the loss lives (W4 uniform vs FP8, B70, paired)

`scripts/mcnemar.py --quant results/b70/w4_rtn_u/boolq.json --base
results/b70/fp8_vllm_u/boolq.json`:

- 182 vs 32 discordant, exact McNemar **p = 1.1e-26** (regression is real).
- Stratified by passage length (chars): **short passages degrade most** —
  Q1 −8.57 pp, Q2 −5.15, Q3 −2.81, Q4 −1.83. Same monotone signature as on
  the Ada (Q1 −9.6 pp → Q4 −4.0 pp): the BoolQ loss is a *short-context
  precision* effect, not a long-context/KV effect.
- The recovery mirrors the loss: v2 (out/o_proj → fp16, +1.04 pp, McNemar
  p = 2.7e-4) recovers mostly in the same short-passage regime
  (Q1 +1.59, Q2 +1.96, Q3 +0.37, Q4 +0.24) — precision-sensitive docs are
  where both the loss and the fix concentrate.
- The int2 floor point (W2 vs W4-uniform, −7.9 pp, p = 4.2e-23) shows the
  **opposite** profile: long passages degrade most (Q3 −11.0, Q4 −10.6 vs
  Q1 −4.3). Consistent with two regimes: int4's marginal noise only tips
  borderline short-passage cases, while int2's larger noise also breaks
  long-range processing — plausibly accumulating through the GDN recurrent
  state over sequence length.

## Bits-vs-BoolQ curve

<!-- filled by scripts/b70_sweep_report.py when the sweep completes -->

## Findings

<!-- pending sweep completion -->

## Reproduce

See RUNBOOK.md "B70 mixed-precision sweep — runbook":
`scripts/b70_sweep.sh` (anchor + v1..v5), `scripts/b70_w2.sh` (W2 floor),
`MODEL_KIND=base_fp8 scripts/eval_b70.sh` (FP8 arm),
`scripts/b70_sweep_report.py` (table), `scripts/mcnemar.py` (paired test).
