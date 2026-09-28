#!/usr/bin/env bash
# Final pipeline at full (capped) scale, without the bf16 offload drift check.
#  1. wait for the in-flight FP8 base cheap-task run to finish
#  2. quant  : arc_challenge, mmlu, gsm8k
#  3. base   : arc_challenge, mmlu, gsm8k
#  4. final report
set -u
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env.sh

echo "[final] waiting for the base-cheap run to finish..."
while pgrep -f "[r]un_eval.py" >/dev/null; do sleep 20; done
echo "[final] base-cheap finished at $(date)"

"$PY" scripts/report.py --base results/base --quant results/quant \
  > results/report_partial.txt 2>&1
echo "[final] partial report (4 tasks) written"

./scripts/gpu_free.sh
echo "[final] quant arc/mmlu/gsm8k starting $(date)"
"$PY" scripts/run_eval.py quant --output results/quant \
  --only arc_challenge,mmlu,gsm8k \
  --max-num-seqs 32 --max-num-batched-tokens 16384 \
  --max-model-len 32768 --gpu-mem 0.85 --max-gen-toks 4096 \
  > logs/quant_rest.log 2>&1
echo "[final] quant rest finished $(date)"

./scripts/gpu_free.sh
echo "[final] base fp8 arc/mmlu/gsm8k starting $(date)"
"$PY" scripts/run_eval.py base_fp8 --output results/base \
  --only arc_challenge,mmlu,gsm8k \
  --max-num-seqs 32 --max-num-batched-tokens 16384 \
  --max-model-len 32768 --gpu-mem 0.88 --max-gen-toks 4096 \
  > logs/base_rest.log 2>&1
echo "[final] base rest finished $(date)"

./scripts/gpu_free.sh
"$PY" scripts/report.py --base results/base --quant results/quant \
  > results/report.txt 2>&1
cat results/report.txt
echo "[final] ALL DONE $(date)"
