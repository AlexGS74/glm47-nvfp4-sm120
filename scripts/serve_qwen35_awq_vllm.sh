#!/usr/bin/env bash
set -euo pipefail

# Serve Qwen3.5-397B-A17B-AWQ-4bit with vLLM 0.17+ on SM120 (RTX PRO 6000 Blackwell).
# Model: cyankiwi/Qwen3.5-397B-A17B-AWQ-4bit
# Install: uv tool install --force vllm==0.17.0
# Download: hf download cyankiwi/Qwen3.5-397B-A17B-AWQ-4bit
#
# Usage:
#   ./scripts/serve_qwen35_awq_vllm.sh
#   ./scripts/serve_qwen35_awq_vllm.sh --stop
#   SPEC_TOKENS=2 ./scripts/serve_qwen35_awq_vllm.sh

HF_MODEL_ID="cyankiwi/Qwen3.5-397B-A17B-AWQ-4bit"
LOCAL_CACHE="${HOME}/.cache/huggingface/hub/models--cyankiwi--Qwen3.5-397B-A17B-AWQ-4bit/snapshots"

DOWNLOAD_FIRST=${DOWNLOAD_FIRST:-1}

VLLM_BIN=${VLLM_BIN:-${HOME}/.local/share/uv/tools/vllm/bin/vllm}
VLLM_PYTHON=${VLLM_PYTHON:-${HOME}/.local/share/uv/tools/vllm/bin/python}
export PATH="$(dirname "${VLLM_PYTHON}"):${PATH}"

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
TP=${TP:-4}
# Model uses compressed-tensors format — vLLM auto-detects, no --quantization needed
ATTENTION_BACKEND=${ATTENTION_BACKEND:-TRITON_ATTN}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-claude-opus-4-5-20251001}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4092}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.80}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}
SPEC_TOKENS=${SPEC_TOKENS:-0}
GPU_POWER_LIMIT=${GPU_POWER_LIMIT:-270}

# ── Env vars ──────────────────────────────────────────────────────────────────
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_MARLIN_USE_ATOMIC_ADD=${VLLM_MARLIN_USE_ATOMIC_ADD:-1}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-4}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export SAFETENSORS_FAST_GPU=${SAFETENSORS_FAST_GPU:-1}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}

# ── Stop mode ────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
  echo "Killing vllm on port ${PORT}..."
  pkill -f "vllm serve.*${PORT}" 2>/dev/null && echo "Stopped." || echo "Not running."
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────

if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "vllm not found: ${VLLM_BIN}" >&2
  echo "Install: uv tool install --force vllm==0.17.0" >&2
  exit 1
fi

# ── Find or download model ──────────────────────────────────────────────────
MODEL_PATH=""
for DIR in "${LOCAL_CACHE}"/*/; do
  [[ -d "${DIR}" ]] && MODEL_PATH="${DIR%/}" && break
done

if [[ -z "${MODEL_PATH}" ]]; then
  if [[ "${DOWNLOAD_FIRST}" -eq 1 ]]; then
    echo "Model not cached, downloading ${HF_MODEL_ID}..."
    HF_HUB_OFFLINE=0 hf download "${HF_MODEL_ID}"
    for DIR in "${LOCAL_CACHE}"/*/; do
      [[ -d "${DIR}" ]] && MODEL_PATH="${DIR%/}" && break
    done
    [[ -z "${MODEL_PATH}" ]] && { echo "ERROR: download failed" >&2; exit 1; }
  else
    echo "ERROR: model not found at ${LOCAL_CACHE}" >&2
    echo "  Run: hf download ${HF_MODEL_ID}" >&2
    exit 1
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────

sudo -n nvidia-smi -pl "${GPU_POWER_LIMIT}" -i 0,1,2,3 2>/dev/null \
  && echo "GPU power limit set to ${GPU_POWER_LIMIT}W" \
  || echo "WARNING: could not set GPU power limit" >&2

echo "vLLM:    $("${VLLM_BIN}" --version 2>/dev/null || echo unknown)"
echo "Model:   ${MODEL_PATH}"
echo "Quant:   compressed-tensors (auto-detected)"
echo "TP:      ${TP}  Attention: ${ATTENTION_BACKEND}  Max len: ${MAX_MODEL_LEN}"
echo "MTP:     SPEC_TOKENS=${SPEC_TOKENS} (0=off, 2=tool calls ok)"
echo ""

SPEC_ARGS=()
if [[ "${SPEC_TOKENS}" -gt 0 ]]; then
  SPEC_ARGS+=(--speculative-config.method qwen3_next_mtp)
  SPEC_ARGS+=(--speculative-config.num_speculative_tokens "${SPEC_TOKENS}")
fi

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  "${SPEC_ARGS[@]}" \
  --chat-template "${MODEL_PATH}/chat_template.jinja" \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  "$@"
