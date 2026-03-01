#!/usr/bin/env bash
set -euo pipefail

# Serve GLM-4.7-FP8 (zai-org/GLM-4.7-FP8) with vLLM nightly on SM120.
# Based on the official Z.ai recipe; SM120-specific additions noted inline.
#
# Install: uv tool install --force --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
# Download: huggingface-cli download zai-org/GLM-4.7-FP8

HF_MODEL_ID="zai-org/GLM-4.7-FP8"
LOCAL_CACHE="${HOME}/.cache/huggingface/hub/models--zai-org--GLM-4.7-FP8/snapshots"

DOWNLOAD_FIRST=${DOWNLOAD_FIRST:-1}

VLLM_BIN=${VLLM_BIN:-${HOME}/.local/share/uv/tools/vllm/bin/vllm}
VLLM_PYTHON=${VLLM_PYTHON:-${HOME}/.local/share/uv/tools/vllm/bin/python}
export PATH="$(dirname "${VLLM_PYTHON}"):${PATH}"

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
TP=${TP:-4}
# SM120: flashinfer attention rejects SM120 at this nightly version; TRITON_ATTN works.
# Remove if a future nightly fixes SM120 flashinfer support.
ATTENTION_BACKEND=${ATTENTION_BACKEND:-TRITON_ATTN}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-claude-opus-4-5-20251001}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-88000}
GPU_POWER_LIMIT=${GPU_POWER_LIMIT:-270}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.95}  # GNOME display compositor holds ~500MB on GPU 0 even when not logged in
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}
SWAP_SPACE=${SWAP_SPACE:-16}
SPEC_TOKENS=${SPEC_TOKENS:-0}  # MTP hurts HumanEval quality on FP8 (45% → 32%); off by default

# ─────────────────────────────────────────────────────────────────────────────

if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "vllm not found: ${VLLM_BIN}" >&2
  echo "Install: uv tool install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly" >&2
  exit 1
fi

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  cat <<'EOF'
Usage:
  serve_glm47_fp8_vllm.sh [extra vllm args...]

Environment variables:
  DOWNLOAD_FIRST      (default: 1)   — download model if not cached
  HOST / PORT         (default: 0.0.0.0:30000)
  TP                  (default: 4)
  ATTENTION_BACKEND   (default: TRITON_ATTN)  — SM120 workaround; try FLASHINFER on other hardware
  MAX_MODEL_LEN       (default: 88000)   — vLLM estimates 88560 max with kv_cache_dtype=fp8 + gpu_mem_util=0.95
  SERVED_MODEL_NAME   (default: claude-opus-4-5-20251001)
  GPU_POWER_LIMIT     (default: 270)
  GPU_MEM_UTIL        (default: 0.98)
  MAX_NUM_SEQS        (default: 4)    — keep small to leave room for KV cache
  KV_CACHE_DTYPE      (default: fp8)  — halves KV cache VRAM vs bf16
  SWAP_SPACE          (default: 16)   — GB of CPU swap for KV cache overflow
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
echo "KV:      dtype=${KV_CACHE_DTYPE}  mem_util=${GPU_MEM_UTIL}  max_seqs=${MAX_NUM_SEQS}  swap=${SWAP_SPACE}GB"
echo ""

export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_SLEEP_WHEN_IDLE=1

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP}" \
  --enable-expert-parallel \
  --attention-backend "${ATTENTION_BACKEND}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --swap-space "${SWAP_SPACE}" \
  $([[ "${SPEC_TOKENS}" -gt 0 ]] && echo "--speculative-config.method mtp --speculative-config.num_speculative_tokens ${SPEC_TOKENS}") \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  "$@"
