#!/usr/bin/env bash
# Unattended continuation: wait for the running quant eval, then run the FP8
# baseline, the BF16 drift check, and produce the final report.
set -u
cd /home/acm/paul_nv/MixedInt4-AutoRound
source scripts/env.sh

echo "[pipeline] waiting for the quant run to finish..."
while pgrep -f "[r]un_eval.py" >/dev/null; do sleep 20; done
echo "[pipeline] quant run finished at $(date)"

echo "[pipeline] starting FP8 base run"
./scripts/run_base_fp8_full.sh > logs/base_fp8_full.log 2>&1
echo "[pipeline] FP8 base finished at $(date)"

echo "[pipeline] starting BF16 drift check"
DRIFT_LIMIT=5 ./scripts/run_drift.sh > logs/drift.log 2>&1
echo "[pipeline] drift check finished at $(date)"

echo "[pipeline] report"
"$PY" scripts/report.py --base results/base --quant results/quant \
  > results/report.txt 2>&1
cat results/report.txt
echo "[pipeline] ALL DONE at $(date)"
