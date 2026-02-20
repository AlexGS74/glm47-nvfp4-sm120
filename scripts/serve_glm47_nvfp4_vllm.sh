#!/usr/bin/env bash
set -euo pipefail

# Serve GLM-4.7 NVFP4 with vLLM.
# See docs/sglang/sm120-blackwell-fp4-fixes.md for context on SM120 FP4 status.

# ── Python interpreter ────────────────────────────────────────────────────────
# Default: uv tool venv for vllm 0.15.1
# Install: uv tool install vllm==0.15.1
VLLM_PYTHON=${VLLM_PYTHON:-${HOME}/.local/share/uv/tools/vllm/bin/python}
# Ensure vllm venv bin (ninja, etc.) is on PATH for flashinfer JIT compilation
export PATH="$(dirname "${VLLM_PYTHON}"):${PATH}"

# ── Model / serving ──────────────────────────────────────────────────────────
# Use local HF cache by default to avoid re-downloading.
# Set MODEL_PATH to a HuggingFace repo ID to download instead.
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/huggingface/hub/models--Salyut1--GLM-4.7-NVFP4/snapshots/531df318dbe2877b24881f6e15024c7f945e7048}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
TP=${TP:-4}
DTYPE=${DTYPE:-bfloat16}
QUANTIZATION=${QUANTIZATION:-modelopt_fp4}
# flashinfer attention backend rejects SM120; TRITON_ATTN or FLASH_ATTN work
ATTENTION_BACKEND=${ATTENTION_BACKEND:-TRITON_ATTN}
# 0.80 leaves ~19 GiB free per GPU — enough for CUDA graph capture + sampler warmup
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.80}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm4.7}
# Force VLLM_CUTLASS MoE backend — FLASHINFER_CUTLASS returns zeros on SM120 (#2577)
export VLLM_USE_FLASHINFER_MOE_FP4=${VLLM_USE_FLASHINFER_MOE_FP4:-0}

# ─────────────────────────────────────────────────────────────────────────────

if [[ ! -x "${VLLM_PYTHON}" ]]; then
  echo "Python not found: ${VLLM_PYTHON}" >&2
  echo "Install with: uv tool install vllm==0.15.1" >&2
  exit 1
fi

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  cat <<'EOF'
Usage:
  serve_glm47_nvfp4_vllm.sh [extra vllm args...]

Environment variables:
  VLLM_PYTHON    (default: ~/.local/share/uv/tools/vllm/bin/python)
  MODEL_PATH     (default: Salyut1/GLM-4.7-NVFP4)
  HOST           (default: 0.0.0.0)
  PORT           (default: 30000)
  TP             (default: 4)
  DTYPE             (default: bfloat16)
  QUANTIZATION      (default: modelopt_fp4)
  ATTENTION_BACKEND (default: TRITON_ATTN)  # flashinfer rejects SM120
  GPU_MEM_UTIL      (default: 0.80)         # leave room for warmup + graphs
EOF
  exit 0
fi

echo "Python:      ${VLLM_PYTHON}"
echo "vLLM:        $("${VLLM_PYTHON}" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
echo "Model:       ${MODEL_PATH}"
echo "TP:          ${TP}"
echo "Attention:   ${ATTENTION_BACKEND}"
echo "GPU mem:     ${GPU_MEM_UTIL}"
echo ""

exec "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --quantization "${QUANTIZATION}" \
  --dtype "${DTYPE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  "$@"
