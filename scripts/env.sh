# Shared environment for all scripts. Source this first.
# CUDA_HOME must be set or flashinfer's JIT cannot find cuda_runtime.h.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PROJECT_ROOT="/home/acm/paul_nv/MixedInt4-AutoRound"
export VENV="${PROJECT_ROOT}/.venv"
export PY="${VENV}/bin/python"

export BASE_MODEL="/home/acm/work/models/Qwen3.8-27B"
export QUANT_MODEL="/home/acm/work/models/Qwen3.8-27B-MixedInt4-AutoRound"
