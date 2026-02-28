#!/usr/bin/env bash
set -euo pipefail

# Serve GLM-4.7 AWQ with vLLM on SM120 (RTX PRO 6000 Blackwell).
# Uses Triton kernels which compile natively to PTX, avoiding CUTLASS issues.
#
# Install:
#   uv tool install vllm==0.16.0
#   uv pip install huggingface_hub

HF_MODEL_ID="QuantTrio/GLM-4.7-AWQ"
LOCAL_CACHE="${HOME}/.cache/huggingface/models--QuantTrio--GLM-4.7-AWQ/snapshots"

# First download if not present, then serve from local cache
DOWNLOAD_FIRST=${DOWNLOAD_FIRST:-1}  # 1=download if missing, 0=skip download check

VLLM_BIN=${VLLM_BIN:-${HOME}/.local/share/uv/tools/vllm/bin/vllm}
VLLM_PYTHON=${VLLM_PYTHON:-${HOME}/.local/share/uv/tools/vllm/bin/python}
export PATH="$(dirname "${VLLM_PYTHON}"):${PATH}"

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
TP=${TP:-4}
DTYPE=${DTYPE:-half}
QUANTIZATION=${QUANTIZATION:-awq_marlin}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-claude-opus-4-5-20251001}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-200000}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
SWAP_SPACE=${SWAP_SPACE:-8}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
STREAM_INTERVAL=${STREAM_INTERVAL:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.80}
SPEC_TOKENS=${SPEC_TOKENS:-0}  # MTP slower at low concurrency; 76% acceptance rate but draft overhead dominates

ATTENTION_BACKEND=${ATTENTION_BACKEND:-FLASHINFER}
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
export VLLM_USE_FLASHINFER_MOE_FP16=${VLLM_USE_FLASHINFER_MOE_FP16:-1}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export VLLM_LOGGING_LEVEL=${LOG_LEVEL:-INFO}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
GPU_POWER_LIMIT=${GPU_POWER_LIMIT:-270}

# ── Adopted from FP8 reference recipe ────────────────────────────────────────
# spawn: safer than default fork with CUDA contexts (avoids worker deadlocks)
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
# AWQ uses Marlin kernels — atomic add fixes correctness issues on Blackwell
export VLLM_MARLIN_USE_ATOMIC_ADD=${VLLM_MARLIN_USE_ATOMIC_ADD:-1}
# Drop GPU to P8 between requests — free power win on idle
export VLLM_SLEEP_WHEN_IDLE=${VLLM_SLEEP_WHEN_IDLE:-1}
# Ensure GPU order matches nvidia-smi (important with CUDA_VISIBLE_DEVICES)
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
# Compilation level 3 + full cuda graph — higher throughput; override with
# COMPILATION_CONFIG='{"level":0}' to disable if startup time is an issue
COMPILATION_CONFIG=${COMPILATION_CONFIG:-'{"level": 3, "cudagraph_mode": "full"}'}

if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "vllm not found: ${VLLM_BIN}" >&2
  echo "Install with: uv tool install vllm==0.15.1" >&2
  exit 1
fi

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  cat <<'EOF'
Usage:
  serve_glm47_awq.sh [extra vllm args...]

First run (downloads model if missing):
  ./serve_glm47_awq.sh

Skip download check:
  DOWNLOAD_FIRST=0 ./serve_glm47_awq.sh
EOF
  exit 0
fi

# Find local model path or download
MODEL_PATH=""
SNAPSHOT_DIRS=("${LOCAL_CACHE}"/*)
if [[ -d "${SNAPSHOT_DIRS[0]}" ]]; then
  MODEL_PATH="${SNAPSHOT_DIRS[0]}"
  echo "Model already cached at: ${MODEL_PATH}"
elif [[ "${DOWNLOAD_FIRST}" -eq 1 ]]; then
  echo "Model not found in cache, downloading..."
  echo "  Model: ${HF_MODEL_ID}"
  hf download --repo-type model --cache-dir "${HOME}/.cache/huggingface" "${HF_MODEL_ID}"

  # Check again after download - glob may match multiple dirs
  for DIR in "${LOCAL_CACHE}"/*; do
    if [[ -d "${DIR}" ]]; then
      MODEL_PATH="${DIR}"
      break
    fi
  done

  if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: Download failed, model still not found" >&2
    echo "  Expected: ${LOCAL_CACHE}/*" >&2
    exit 1
  fi
else
  echo "ERROR: DOWNLOAD_FIRST=0 but model not found at ${LOCAL_CACHE}" >&2
  exit 1
fi

if sudo -n nvidia-smi -pl "${GPU_POWER_LIMIT}" -i 0,1,2,3 2>/dev/null; then
  echo "GPU power limit set to ${GPU_POWER_LIMIT}W"
else
  echo "WARNING: could not set GPU power limit" >&2
fi

echo "vLLM:        $("${VLLM_BIN}" --version 2>/dev/null || echo unknown)"
echo "Model:       ${MODEL_PATH}"
echo "TP:          ${TP}  Attention: ${ATTENTION_BACKEND}  GPU mem: ${GPU_MEM_UTIL}"
echo ""

SPEC_FLAGS=""
if [[ "${SPEC_TOKENS}" -gt 0 ]]; then
  SPEC_FLAGS="--speculative-config.method mtp --speculative-config.num_speculative_tokens ${SPEC_TOKENS}"
fi

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --dtype "${DTYPE}" \
  --quantization "${QUANTIZATION}" \
  --tensor-parallel-size "${TP}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --compilation-config "${COMPILATION_CONFIG}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --stream-interval "${STREAM_INTERVAL}" \
  --swap-space "${SWAP_SPACE}" \
  ${SPEC_FLAGS} \
  --chat-template "${MODEL_PATH}/chat_template.jinja" \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  "$@"