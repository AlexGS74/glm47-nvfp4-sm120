#!/usr/bin/env bash
set -euo pipefail

# Serve GLM-4.7 NVFP4 with vLLM on SM120 (RTX PRO 6000 Blackwell).
# See docs/sglang/vllm-sm120-nvfp4-working-state.md for full context.
# Install: uv tool install vllm==0.15.1

VLLM_BIN=${VLLM_BIN:-${HOME}/.local/share/uv/tools/vllm/bin/vllm}
VLLM_PYTHON=${VLLM_PYTHON:-${HOME}/.local/share/uv/tools/vllm/bin/python}
# Ensure vllm venv bin (ninja, etc.) is on PATH for flashinfer JIT compilation
export PATH="$(dirname "${VLLM_PYTHON}"):${PATH}"

# ── Model / serving ──────────────────────────────────────────────────────────
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/huggingface/hub/models--Salyut1--GLM-4.7-NVFP4/snapshots/531df318dbe2877b24881f6e15024c7f945e7048}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}  # LiteLLM proxy removed; vLLM serves Anthropic /v1/messages natively
TP=${TP:-4}
DTYPE=${DTYPE:-bfloat16}
QUANTIZATION=${QUANTIZATION:-modelopt_fp4}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm4.7}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-200000}   # model max: 202752; Claude Code needs ~99k for system prompt
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
SWAP_SPACE=${SWAP_SPACE:-16}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}  # larger = fewer prefill chunks = less AllReduce overhead; was 16384
STREAM_INTERVAL=${STREAM_INTERVAL:-1}   # keep at 1 for smooth interactive streaming; interval=5 causes stalls with include_usage
# 0.80 leaves ~19 GiB free per GPU — enough for CUDA graph capture + sampler warmup
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.80}
# MTP speculative decoding — set to 0 to disable
SPEC_TOKENS=${SPEC_TOKENS:-0}  # MTP acceptance rate is 0% on NVFP4; disabled until fixed

# ── SM120 / Blackwell fixes ───────────────────────────────────────────────────
# flashinfer attention rejects SM120; TRITON_ATTN works
ATTENTION_BACKEND=${ATTENTION_BACKEND:-TRITON_ATTN}
# Force VLLM_CUTLASS MoE backend — FLASHINFER_CUTLASS returns zeros on SM120 (#2577)
export VLLM_USE_FLASHINFER_MOE_FP4=${VLLM_USE_FLASHINFER_MOE_FP4:-0}

# ── Official GLM-4.7 recipe env vars ─────────────────────────────────────────
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
export VLLM_USE_FLASHINFER_MOE_FP16=${VLLM_USE_FLASHINFER_MOE_FP16:-1}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export VLLM_LOGGING_LEVEL=${LOG_LEVEL:-INFO}
# PCIe-only multi-GPU comm tuning.
# CUDA_DEVICE_MAX_CONNECTIONS=1: serializes P2P connections, reduces contention.
# NCCL_P2P_DISABLE=1: forces NCCL through host SHM instead of direct P2P.
#   For decode (small ~14KB AllReduces), SHM latency can beat P2P negotiation overhead.
#   Try both 0 and 1 to benchmark — P2P wins for large prefill, SHM may win for decode.
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}

# ── GPU power limit (thermal management) ─────────────────────────────────────
# RTX PRO 6000 Max-Q hits 89–91°C at full 300W during inference.
# 250W keeps temps ~80–85°C with negligible decode performance impact
# (decode is memory-bandwidth bound, not compute bound).
# Requires: sudo visudo -f /etc/sudoers.d/nvidia-power
#   alex ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi -pl *
GPU_POWER_LIMIT=${GPU_POWER_LIMIT:-270}

# ─────────────────────────────────────────────────────────────────────────────

if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "vllm not found: ${VLLM_BIN}" >&2
  echo "Install with: uv tool install vllm==0.15.1" >&2
  exit 1
fi

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  cat <<'EOF'
Usage:
  serve_glm47_nvfp4_vllm.sh [extra vllm args...]

Environment variables:
  MODEL_PATH          (default: Salyut1/GLM-4.7-NVFP4 local cache)
  HOST / PORT         (default: 0.0.0.0:30000)
  TP                  (default: 4)  — set to 8 to enable --enable-expert-parallel
  DTYPE               (default: bfloat16)
  QUANTIZATION        (default: modelopt_fp4)
  ATTENTION_BACKEND   (default: TRITON_ATTN)   # flashinfer rejects SM120
  GPU_MEM_UTIL        (default: 0.80)          # leave room for warmup + cuda graphs
  MAX_MODEL_LEN       (default: 200000)        # model max: 202752
  MAX_NUM_SEQS        (default: 32)
  SWAP_SPACE          (default: 16)
  SERVED_MODEL_NAME   (default: glm4.7)
  SPEC_TOKENS         (default: 1)             # MTP speculative tokens; set to 0 to disable
  LOG_LEVEL           (default: INFO)          # set to DEBUG for verbose output
EOF
  exit 0
fi

if sudo -n nvidia-smi -pl "${GPU_POWER_LIMIT}" -i 0,1,2,3 2>/dev/null; then
  echo "GPU power limit set to ${GPU_POWER_LIMIT}W"
else
  echo "WARNING: could not set GPU power limit (sudo not configured?)" >&2
  echo "  Run: sudo visudo -f /etc/sudoers.d/nvidia-power" >&2
  echo "  Add: ${USER} ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi -pl *" >&2
fi

echo "vLLM:        $("${VLLM_BIN}" --version 2>/dev/null || echo unknown)"
echo "Model:       ${MODEL_PATH}"
echo "TP:          ${TP}  Attention: ${ATTENTION_BACKEND}  GPU mem: ${GPU_MEM_UTIL}"
echo "Max len:     ${MAX_MODEL_LEN}  Spec tokens: ${SPEC_TOKENS}"
echo ""

# --enable-expert-parallel is required at TP>=8; breaks VLLM_CUTLASS FP4 MoE path at TP=4
EXPERT_PARALLEL_FLAG=""
if [[ "${TP}" -ge 8 ]]; then
  EXPERT_PARALLEL_FLAG="--enable-expert-parallel"
fi

# MTP speculative decoding (optional, set SPEC_TOKENS=0 to disable)
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
  ${EXPERT_PARALLEL_FLAG} \
  --attention-backend "${ATTENTION_BACKEND}" \
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
  # --enable-log-requests \   # uncomment to log full request/response text (adds overhead)
  # --enable-log-outputs \
  --trust-remote-code \
  "$@"
