#!/usr/bin/env bash
set -euo pipefail

# Serve Qwen3.5-397B-A17B-NVFP4 (Sehyo/Qwen3.5-397B-A17B-NVFP4) with vLLM nightly on SM120.
# Based on community recipes from RTX6kPRO Discord (Feb 2026).
#
# Key recipe notes (from community thread):
#   - --tool-call-parser qwen35_coder (NOT qwen3_coder; fixed in vLLM PR #35347)
#   - MTP method qwen3_next_mtp, tokens=2: tool calls work + ~125-140 tok/s
#   - MTP tokens=3-5: breaks tool calls regardless of parser
#   - SPEC_TOKENS=0: max reliability for agentic/concurrent workloads (~64 tok/s)
#   - vLLM PR #35156 (mlp.gate fix) merged in nightly — no config.json edits needed
#   - SM120: use TRITON_ATTN (flashinfer may work on newer nightlies, see ATTENTION_BACKEND)
#
# Install: uv tool install --force --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
# Download: huggingface-cli download Sehyo/Qwen3.5-397B-A17B-NVFP4

HF_MODEL_ID="Sehyo/Qwen3.5-397B-A17B-NVFP4"
LOCAL_CACHE="${HOME}/.cache/huggingface/hub/models--Sehyo--Qwen3.5-397B-A17B-NVFP4/snapshots"

DOWNLOAD_FIRST=${DOWNLOAD_FIRST:-1}

VLLM_BIN=${VLLM_BIN:-${HOME}/.local/share/uv/tools/vllm/bin/vllm}
VLLM_PYTHON=${VLLM_PYTHON:-${HOME}/.local/share/uv/tools/vllm/bin/python}
export PATH="$(dirname "${VLLM_PYTHON}"):${PATH}"

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
TP=${TP:-4}
# SM120: TRITON_ATTN (flashinfer may work on newer nightlies, see ATTENTION_BACKEND)
ATTENTION_BACKEND=${ATTENTION_BACKEND:-TRITON_ATTN}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-claude-opus-4-5-20251001}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-128000}
GPU_POWER_LIMIT=${GPU_POWER_LIMIT:-270}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.80}   # community recipe default; GNOME on GPU 0 (~500MB) is well within this
# MTP: 2 = fast + tool calls work (qwen35_coder + qwen3_next_mtp, ~125-140 tok/s)
#       0 = max reliability / high concurrency (~64 tok/s)
#     3-5 = breaks tool calls (use only for single-request code gen demos)
SPEC_TOKENS=${SPEC_TOKENS:-2}

# ─────────────────────────────────────────────────────────────────────────────

if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "vllm not found: ${VLLM_BIN}" >&2
  echo "Install: uv tool install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly" >&2
  exit 1
fi

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  cat <<'EOF'
Usage:
  serve_qwen35_nvfp4_vllm.sh [extra vllm args...]

Environment variables:
  DOWNLOAD_FIRST      (default: 1)   — download model if not cached
  HOST / PORT         (default: 0.0.0.0:30000)
  TP                  (default: 4)
  ATTENTION_BACKEND   (default: TRITON_ATTN)  — SM120 workaround; try FLASHINFER on other hardware
  MAX_MODEL_LEN       (default: 128000)
  SERVED_MODEL_NAME   (default: claude-opus-4-5-20251001)
  GPU_POWER_LIMIT     (default: 270)
  GPU_MEM_UTIL        (default: 0.80)
  SPEC_TOKENS         (default: 2)   — MTP tokens; 0=off, 1-2=tool calls work, 3-5=tool calls broken
EOF
  exit 0
fi

# ── Find or download model ────────────────────────────────────────────────────
MODEL_PATH=""
for DIR in "${LOCAL_CACHE}"/*/; do
  [[ -d "${DIR}" ]] && MODEL_PATH="${DIR%/}" && break
done

if [[ -z "${MODEL_PATH}" ]]; then
  if [[ "${DOWNLOAD_FIRST}" -eq 1 ]]; then
    echo "Model not cached, downloading ${HF_MODEL_ID}..."
    HF_HUB_OFFLINE=0 huggingface-cli download "${HF_MODEL_ID}"
    for DIR in "${LOCAL_CACHE}"/*/; do
      [[ -d "${DIR}" ]] && MODEL_PATH="${DIR%/}" && break
    done
    [[ -z "${MODEL_PATH}" ]] && { echo "ERROR: download failed" >&2; exit 1; }
  else
    echo "ERROR: model not found at ${LOCAL_CACHE}" >&2
    echo "  Run: huggingface-cli download ${HF_MODEL_ID}" >&2
    exit 1
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────

sudo -n nvidia-smi -pl "${GPU_POWER_LIMIT}" -i 0,1,2,3 2>/dev/null \
  && echo "GPU power limit set to ${GPU_POWER_LIMIT}W" \
  || echo "WARNING: could not set GPU power limit" >&2

echo "vLLM:    $("${VLLM_BIN}" --version 2>/dev/null || echo unknown)"
echo "Model:   ${MODEL_PATH}"
echo "TP:      ${TP}  Attention: ${ATTENTION_BACKEND}  Max len: ${MAX_MODEL_LEN}"
echo "MTP:     SPEC_TOKENS=${SPEC_TOKENS} (0=off, 2=tool calls ok, 3-5=tool calls broken)"
echo ""

export VLLM_SLEEP_WHEN_IDLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# VLLM_NVFP4_GEMM_BACKEND=cutlass — causes illegal memory access with MTP on SM120
export NCCL_P2P_LEVEL=4
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=8
export SAFETENSORS_FAST_GPU=1

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
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --enable-prefix-caching \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  "${SPEC_ARGS[@]}" \
  --tool-call-parser qwen35_coder \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  "$@"
