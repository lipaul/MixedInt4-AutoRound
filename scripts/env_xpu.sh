# XPU (Intel Arc Pro B70) environment. Source before any XPU work.
# Deliberately keeps the RTX 6000 Ada out of the picture.

export PROJECT_ROOT="/home/acm/paul_nv/MixedInt4-AutoRound"
export VENV_XPU="${PROJECT_ROOT}/.venv-xpu"
export PY_XPU="${VENV_XPU}/bin/python"

export BASE_MODEL="/home/acm/work/models/Qwen3.8-27B"

# Pin Level-Zero device 0 = Intel Battlemage G31 [0xe223] (the B70).
export ONEAPI_DEVICE_SELECTOR="level_zero:0"
# Belt and braces: make the NVIDIA GPU invisible to anything CUDA-aware.
export CUDA_VISIBLE_DEVICES=""

export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"

# vLLM-XPU venv (separate from the AutoRound venv).
export VENV_VLLM_XPU="${PROJECT_ROOT}/.venv-vllm-xpu"
export PY_VLLM_XPU="${VENV_VLLM_XPU}/bin/python"

# This host has both an Intel B70 and an NVIDIA Ada. vLLM's XPU wheel detects
# CUDA via vendored pynvml (which ignores CUDA_VISIBLE_DEVICES), so both
# platform plugins activate and vLLM aborts. This flag disables CUDA detection.
# Patch applied in .venv-vllm-xpu/.../vllm/platforms/__init__.py.
export VLLM_DISABLE_CUDA_PLATFORM=1
