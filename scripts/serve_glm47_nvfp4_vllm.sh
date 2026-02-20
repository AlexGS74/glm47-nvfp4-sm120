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
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}   # model max: 202752; reduce if KV cache OOMs
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
SWAP_SPACE=${SWAP_SPACE:-16}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm4.7}

# ── SM120 / Blackwell fixes ───────────────────────────────────────────────────
# Force VLLM_CUTLASS MoE backend — FLASHINFER_CUTLASS returns zeros on SM120 (#2577)
export VLLM_USE_FLASHINFER_MOE_FP4=${VLLM_USE_FLASHINFER_MOE_FP4:-0}

# ── Official GLM-4.7 recipe env vars ─────────────────────────────────────────
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
export VLLM_USE_FLASHINFER_MOE_FP16=${VLLM_USE_FLASHINFER_MOE_FP16:-1}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

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
  VLLM_PYTHON         (default: ~/.local/share/uv/tools/vllm/bin/python)
  MODEL_PATH          (default: Salyut1/GLM-4.7-NVFP4 local cache)
  HOST                (default: 0.0.0.0)
  PORT                (default: 30000)
  TP                  (default: 4)  use 8 to add --enable-expert-parallel automatically
  DTYPE               (default: bfloat16)
  QUANTIZATION        (default: modelopt_fp4)
  ATTENTION_BACKEND   (default: TRITON_ATTN)   # flashinfer rejects SM120
  GPU_MEM_UTIL        (default: 0.80)          # leave room for warmup + graphs
  MAX_MODEL_LEN       (default: 32768)
  MAX_NUM_SEQS        (default: 32)
  SWAP_SPACE          (default: 16)
  SERVED_MODEL_NAME   (default: glm4.7)
EOF
  exit 0
fi

echo "Python:      ${VLLM_PYTHON}"
echo "vLLM:        $("${VLLM_PYTHON}" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
echo "Model:       ${MODEL_PATH}"
echo "TP:          ${TP}"
echo "Attention:   ${ATTENTION_BACKEND}"
echo "GPU mem:     ${GPU_MEM_UTIL}"
echo "Max len:     ${MAX_MODEL_LEN}"
echo ""

# --enable-expert-parallel is required when TP>=8 to shard MoE expert tensors
EXPERT_PARALLEL_FLAG=""
if [[ "${TP}" -ge 8 ]]; then
  EXPERT_PARALLEL_FLAG="--enable-expert-parallel"
fi

exec "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --quantization "${QUANTIZATION}" \
  --dtype "${DTYPE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  ${EXPERT_PARALLEL_FLAG} \
  --attention-backend "${ATTENTION_BACKEND}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --swap-space "${SWAP_SPACE}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --chat-template "${MODEL_PATH}/chat_template.jinja" \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --trust-remote-code \
  "$@"
