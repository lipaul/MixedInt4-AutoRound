# Recovery Rate verification — environment log

Task: independently verify the Recovery Rate claimed by
`Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound` on a single RTX 6000 Ada.

## Hardware
- 1x NVIDIA RTX 6000 Ada Generation, 49140 MiB, compute capability (8,9)
- Driver 580.178.04, CUDA 13.1 toolkit (nvcc V13.1.115)
- 183 GB RAM, 24 cores

## Software (`.venv`, CPython 3.12.13)
- vllm 0.30.0
- torch 2.13.0+cu130
- transformers 5.17.0
- lm-eval 0.4.13
- auto-round 0.15.1

## Local checkpoints (pre-downloaded, not re-fetched)
- Base:  `/home/acm/work/models/Qwen3.8-27B`                         (~52 GB, 27.78 B params BF16)
- Quant: `/home/acm/work/models/Qwen3.8-27B-MixedInt4-AutoRound`    (~20 GB, 407 quantized linears)

## Quantization scheme of the checkpoint (from config.json)
- `quant_method: auto-round`, `packing_format: auto_round:auto_gptq`
- group_size 32, symmetric, base bits 4
- `extra_config` overrides: `linear_attn.in_proj_a/b` = fp16 (bits 16);
  selected `linear_attn.in_proj_qkv/z/out_proj`, `self_attn.o_proj`,
  `mlp.down_proj` = int8; `mtp.fc` = fp16
- `block_name_to_quantize`: `model.language_model.layers`, `mtp.layers`
- vision tower preserved at bf16

## vLLM support findings (Phase 1)
- `Qwen3_5ForConditionalGeneration` is registered natively in
  `vllm/model_executor/models/registry.py`.
- `quant_method: auto-round` is auto-overridden to the `inc` backend
  (`inc/inc.py::INCConfig.override_quantization_method`).
- `inc/config_parser.py` reads per-layer `extra_config` bits (exact, suffix and
  fused-module matches), so the mixed 4/8/16-bit allocation is honoured.
- On CUDA, int4/int8 WNA16 layers route to the Marlin/GPTQ path
  (`inc_wna16_scheme.py`); only 2/3/5/6/7-bit would fall back to humming.

## Phase 1 result: the checkpoint loads and generates correctly

- Serves with `--quantization inc`, text-only (`--limit-mm-per-prompt 0`).
- Weights ~19.4 GiB on GPU; both int4 and int8 Marlin layers accepted.
- Smoke test correct (`17*23` -> `391`); model runs in thinking mode by default.
- NOTE: `CUDA_HOME` must be set or flashinfer's JIT fails to find
  `cuda_runtime.h` and engine init dies.

## Throughput measurements (quant model, RTX 6000 Ada)

Microbenchmark (`scripts/microbench.py`), 64 prompts of ~1200 tokens,
`max_tokens=1` (loglikelihood-style, prefill-bound):

| Case | prefill tok/s | note |
|---|---|---|
| distinct prompts | 2295 | no shared prefix |
| shared long prefix (MMLU-like A/B/C/D) | 6111 | 2.7x - prefix caching works |
| identical prompts | 6237 | cache best case |
| decode, 8 seqs x 64 tokens | 151 tok/s aggregate | generation is the scarce resource |

Implication: MMLU-style multiple-choice is prefill/prefix-cache bound and is
affordable. **GSM8K (long thinking generations) is the bottleneck**, and would
be far worse on the CPU-offloaded bf16 base.

## Hard constraint: full-size evaluation is infeasible
The GPU sits at 100% util and **294 W of a 300 W power cap** during eval, at
~1750 effective tok/s. MMLU 5-shot prompts are ~2800 tokens and are scored 4x
(one request per choice), so uncapped cost is ~45 h per model:

| task | uncapped cost |
|---|---|
| MMLU | ~25 h |
| HellaSwag | ~9.5 h |
| ARC-Challenge | ~3 h |
| BoolQ | ~1.6 h |
| GSM8K (thinking, ~2k tok gen) | ~5 h |

Prefix caching gives little benefit here: 48 of 64 layers are linear-attention
(GDN), whose recurrent state must be recomputed, so cache reuse is limited.

**Therefore the full set is run with per-task caps** (identical for both
models), keeping all 7 benchmarks while bounding runtime to ~4 h/model.
Caps are cost control, not methodology; per-task stderr is reported.


