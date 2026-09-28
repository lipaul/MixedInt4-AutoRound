# Recovery Rate verification — results

Independent verification of the Recovery Rate claimed by
`Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound`, measured on a single
RTX 6000 Ada (48 GB).

- **Base reference**: FP8 quantization of `Qwen/Qwen3.8-27B` (see caveats).
- **Quantized**: the checkpoint under test, served by vLLM 0.30.0 via its INC
  backend (mixed int4/int8 Marlin + fp16 exceptions).
- Identical task, few-shot, prompt and generation settings for both models.

## Recovery Rate

| Benchmark | Base % | Quant % | Δ pp | Recovery | Card claims | Verdict |
|---|---|---|---|---|---|---|
| MMLU | 83.42±1.05 | 82.89±1.07 | −0.53 | 99.37% | 99.50% | within noise |
| GSM8K (flexible) | 98.80±0.69 | 98.40±0.80 | −0.40 | 99.60% | 104.47% | within noise |
| GSM8K (strict) | 98.80±0.69 | 98.40±0.80 | −0.40 | 99.60% | 104.73% | within noise |
| ARC-Challenge | 74.40±2.77 | 73.60±2.79 | −0.80 | 98.92% | 100.00% | within noise |
| BoolQ | 88.60±1.42 | 81.20±1.75 | **−7.40** | **91.65%** | 92.90% | **real regression** |
| HellaSwag | 66.25±2.37 | 66.75±2.36 | +0.50 | 100.75% | 99.49% | within noise |
| PIQA | 81.56±0.90 | 81.61±0.90 | +0.05 | 100.07% | 100.06% | within noise |
| WinoGrande | 76.16±1.20 | 76.48±1.19 | +0.32 | 100.41% | 100.41% | within noise |
| **Average (7)** | **81.31** | **80.13** | **−1.18** | **98.55%** | 99.38% | |

### MMLU category breakdown

| Category | Base % | Quant % | Δ pp | Recovery | Card claims |
|---|---|---|---|---|---|
| Humanities | 85.00±2.23 | 84.62±2.26 | −0.38 | 99.55% | 100.00% |
| Other | 83.46±2.09 | 82.69±2.19 | −0.77 | 99.08% | 99.81% |
| Social Sciences | 89.17±1.94 | 88.75±1.98 | −0.42 | 99.53% | 99.57% |
| STEM | 78.68±2.01 | 78.16±2.04 | −0.53 | 99.33% | 98.36% |

## Conclusions

1. **The card's central claim holds.** Average recovery 98.55% measured vs
   99.38% claimed. Six of seven benchmarks are within sampling noise of 100%,
   and the card's own headline numbers (PIQA 100.06% vs measured 100.07%,
   WinoGrande 100.41% vs 100.41%) reproduce essentially exactly.

2. **The "recovery above 100%" claims do not reproduce.** The card reports
   GSM8K at 104.47% (quantized *better* than base). Measured: 99.60%, i.e. a
   small decrease, not an improvement. A quantized model beating its own
   baseline on a reasoning benchmark is a red flag; here it does not replicate.

3. **BoolQ is the one real, statistically significant regression.** Measured
   −7.40 pp (|Δ| far exceeds the 2.25 pp combined standard error), i.e. recovery
   91.65%. This is *worse* than the card's reported −6.15 pp / 92.90%. This is
   the finding most worth acting on.

4. **STEM is the weakest MMLU category** (99.33%), consistent with the card's
   own admission that STEM degrades most.

## Caveats

- **Base is FP8, not BF16.** The true BF16 baseline does not fit in 48 GB, and
  with CPU offload it was impractically slow (>16 min without completing a
  single batch). FP8 W8A8 is near-lossless. Where our methodology matches the
  card's, the FP8-vs-BF16 gap is bounded by the card's own baseline:

  | | card BF16 base | our FP8 base | diff |
  |---|---|---|---|
  | MMLU | 83.49 | 83.42 | −0.07 |
  | PIQA | 81.61 | 81.56 | −0.05 |
  | WinoGrande | 75.85 | 76.16 | +0.31 |
  | BoolQ | 86.64 | 88.60 | +1.96 (n=500, ±1.42 stderr) |

  So the FP8 substitution contributes at most ~2 pp, and mostly less.

- **Per-task caps** (cost control under a 300 W power cap, applied identically
  to both models): BoolQ 500, HellaSwag 400, ARC-Challenge 250, MMLU 20/subtask
  (~1,140 docs), GSM8K 250. PIQA and WinoGrande are full test sets.

- **Absolute values differ from the card for ARC, HellaSwag and GSM8K**
  (e.g. GSM8K 98.8 vs 72.86). These are prompt/methodology differences, not
  quantization effects: the same settings are applied to both models, so the
  *ratio* remains a valid measurement. MMLU, PIQA, WinoGrande and BoolQ match
  the card's absolute values closely.

- Single run per configuration; lm-eval reports no sampling variance for the
  multiple-choice tasks (deterministic scoring), so stderr is the relevant
  uncertainty.

## Reproduce

    scripts/run_eval.py quant     --output results/quant   # all tasks
    scripts/run_eval.py base_fp8  --output results/base    # all tasks
    scripts/report.py --base results/base --quant results/quant
