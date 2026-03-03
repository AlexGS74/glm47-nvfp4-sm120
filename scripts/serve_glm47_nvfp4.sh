#!/usr/bin/env bash
set -euo pipefail

# Serve GLM-4.7 NVFP4 (ModelOpt FP4) with SGLang.
#
# Tested working on SM120 (RTX PRO 6000 Blackwell) with SGLang v0.5.6 +
# flashinfer 0.5.3 using flashinfer_cutlass MoE backend.
# The flashinfer_trtllm backend (default in SGLang main) uses SM100-only
# cubins and fails on SM120. See docs/sglang/sm120-blackwell-fp4-fixes.md.

# ── Python interpreter ────────────────────────────────────────────────────────
# SGLANG_PYTHON: path to the Python binary to use.
#   - Default: the uv-managed tool venv for sglang 0.5.6.post2
#     (installed via: uv tool install sglang==0.5.6.post2
#                     --extra-index-url https://flashinfer.ai/whl/cu128/torch2.6/)
#   - Override to any other python, e.g. from a dev venv:
#     SGLANG_PYTHON=/home/alex/mllm/sglang/python/.venv/bin/python
SGLANG_PYTHON=${SGLANG_PYTHON:-${HOME}/.local/share/uv/tools/sglang/bin/python}

# ── Model / serving ──────────────────────────────────────────────────────────
# Use local HF cache by default to avoid re-downloading.
# Set MODEL_PATH to a HuggingFace repo ID to download instead.
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/huggingface/hub/models--Salyut1--GLM-4.7-NVFP4/snapshots/531df318dbe2877b24881f6e15024c7f945e7048}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
TP=${TP:-4}
DTYPE=${DTYPE:-half}
QUANTIZATION=${QUANTIZATION:-modelopt_fp4}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-claude-opus-4-5-20251001}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-200000}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-48}
MEM_FRACTION=${MEM_FRACTION:-0.95}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-8}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
SCHEDULE_CONSERV=${SCHEDULE_CONSERV:-0.3}

# ── Parsers ──────────────────────────────────────────────────────────────────
# glm45 is deprecated in v0.5.6 (use glm); glm47 was added after v0.5.6.
# glm works in v0.5.6 and main for tool-call-parser.
# Override to glm47 when using SGLang main: TOOL_CALL_PARSER=glm47
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-glm}
REASONING_PARSER=${REASONING_PARSER:-glm45}

# ── Backend selection ─────────────────────────────────────────────────────────
# flashinfer_cutlass: CUTLASS JIT path — works on SM120 with flashinfer 0.5.3.
# flashinfer_trtllm: downloads SM100-only cubins — crashes on SM120.
MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-flashinfer_cutlass}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flashinfer}   # trtllm attention rejects SM120
CUDA_GRAPH=${CUDA_GRAPH:-auto}                       # auto|0|1

# ── FP4 GEMM backend env var (required for flashinfer 0.5.x CUTLASS path) ───
export SGLANG_USE_CUTLASS_BACKEND_FOR_FP4_GEMM=${SGLANG_USE_CUTLASS_BACKEND_FOR_FP4_GEMM:-1}

# ── Env vars from reference recipe ─────────────────────────────────────────
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=${SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK:-True}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}

# ─────────────────────────────────────────────────────────────────────────────

if [[ ! -x "${SGLANG_PYTHON}" ]]; then
  echo "Python not found: ${SGLANG_PYTHON}" >&2
  echo "Install the default tool venv with:" >&2
  echo "  uv tool install sglang==0.5.6.post2 --extra-index-url https://flashinfer.ai/whl/cu128/torch2.6/" >&2
  echo "Or point SGLANG_PYTHON at any other Python that has sglang installed." >&2
  exit 1
fi

have_nvcc=0
if command -v nvcc >/dev/null 2>&1; then
  have_nvcc=1
elif [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  have_nvcc=1
elif [[ -n "${CUDA_PATH:-}" && -x "${CUDA_PATH}/bin/nvcc" ]]; then
  have_nvcc=1
fi

resolved_disable_cuda_graph=0
if [[ "${CUDA_GRAPH}" == "0" ]]; then
  resolved_disable_cuda_graph=1
elif [[ "${CUDA_GRAPH}" == "auto" && "${ATTENTION_BACKEND}" == "flashinfer" && "${have_nvcc}" == "0" ]]; then
  resolved_disable_cuda_graph=1
fi

# Basic help
if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  cat <<'EOF'
Usage:
  serve_glm47_nvfp4.sh [extra sglang args...]

Environment variables:
  SGLANG_PYTHON         (default: ~/.local/share/uv/tools/sglang/bin/python)
  MODEL_PATH            (default: Salyut1/GLM-4.7-NVFP4)
  HOST                  (default: 0.0.0.0)
  PORT                  (default: 30000)
  TP                    (default: 4)
  DTYPE                 (default: half)
  QUANTIZATION          (default: modelopt_fp4)
  MOE_RUNNER_BACKEND    (default: flashinfer_cutlass)  # use flashinfer_trtllm on SM100
  ATTENTION_BACKEND     (default: flashinfer)
  CUDA_GRAPH            (default: auto)
  SGLANG_USE_CUTLASS_BACKEND_FOR_FP4_GEMM  (default: 1)

Examples:
  # SM120 (RTX PRO 6000) — use v0.5.6 worktree with flashinfer 0.5.3
  bash serve_glm47_nvfp4.sh

  # Point at a dev venv (e.g. SGLang main)
  SGLANG_PYTHON=/home/alex/mllm/sglang/python/.venv/bin/python bash serve_glm47_nvfp4.sh

  # SM100 (B200/datacenter Blackwell) — main branch with trtllm backend
  SGLANG_PYTHON=/home/alex/mllm/sglang/python/.venv/bin/python \
  MOE_RUNNER_BACKEND=flashinfer_trtllm bash serve_glm47_nvfp4.sh

  # Different port + tensor parallel
  PORT=31000 TP=2 bash serve_glm47_nvfp4.sh

  # Tune memory
  bash serve_glm47_nvfp4.sh --mem-fraction-static 0.85
EOF
  exit 0
fi

extra_flags=()
extra_flags+=("--attention-backend" "${ATTENTION_BACKEND}")
if [[ "${resolved_disable_cuda_graph}" == "1" ]]; then
  extra_flags+=("--disable-cuda-graph")
fi

echo "Python:      ${SGLANG_PYTHON}"
echo "SGLang:      $("${SGLANG_PYTHON}" -c 'import sglang; print(sglang.__version__)' 2>/dev/null || echo unknown)"
echo "FlashInfer:  $("${SGLANG_PYTHON}" -c 'import flashinfer; print(flashinfer.__version__)' 2>/dev/null || echo unknown)"
echo "MoE backend: ${MOE_RUNNER_BACKEND}"
echo "Attention:   ${ATTENTION_BACKEND}"
echo "CUTLASS env: SGLANG_USE_CUTLASS_BACKEND_FOR_FP4_GEMM=${SGLANG_USE_CUTLASS_BACKEND_FOR_FP4_GEMM}"
echo ""

exec "${SGLANG_PYTHON}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --model-impl sglang \
  --quantization "${QUANTIZATION}" \
  --dtype "${DTYPE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp "${TP}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --enable-mixed-chunk \
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --schedule-conservativeness "${SCHEDULE_CONSERV}" \
  --sleep-on-idle \
  --enable-metrics \
  --enable-cache-report \
  --disable-shared-experts-fusion \
  --trust-remote-code \
  --tool-call-parser "${TOOL_CALL_PARSER}" \
  --reasoning-parser "${REASONING_PARSER}" \
  --enable-custom-logit-processor \
  --moe-runner-backend "${MOE_RUNNER_BACKEND}" \
  "${extra_flags[@]}" \
  "$@"

# ── Changes applied from zai-org FP8 reference recipe (2026-03-03) ──────────
# These were added based on the working zai-org/GLM-4.7-FP8 SGLang recipe.
# Remove this section once validated/tuned for our NVFP4 setup.
#
# --served-model-name        — spoof as claude-opus-4-5-20251001 for Claude Code
# --context-length 200000    — explicit cap (model max 202752); was unset (unlimited)
# --mem-fraction-static 0.95 — more VRAM for KV cache; was unset (~0.88 default)
# --max-running-requests 48  — prevents OOM under concurrent load; was unset (unlimited)
# --chunked-prefill-size 8192— controls prefill chunk granularity; was unset
# --enable-mixed-chunk       — allows decode to run during prefill chunks
# --cuda-graph-max-bs 8      — explicit cudagraph batch cap; was auto
# --kv-cache-dtype fp8_e4m3  — halves KV cache VRAM vs bf16; was unset (bf16 default)
# --schedule-conservativeness 0.3 — more aggressive scheduling (default 1.0)
# --sleep-on-idle            — drop GPU to P8 between requests
# --enable-metrics           — Prometheus /metrics endpoint
# --enable-cache-report      — prefix cache hit stats in logs
# --disable-shared-experts-fusion — stability on MoE models; was unset
# SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True — skip false-positive TP memory check
# PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 — reduce CUDA fragmentation
#
# NOT applied (FP8-only or needs more GPUs):
#   --dp 2                   — needs more VRAM/GPUs than we have for NVFP4
#   --fp8-gemm-backend triton— FP8 model only
#   USE_TRITON_W8A8_FP8_KERNEL=1 — FP8 model only
#   --enable-hierarchical-cache --hicache-ratio 5 — needs testing, may conflict with NVFP4
#   --enable-flashinfer-allreduce-fusion — needs testing on SM120
#   --speculative-algorithm EAGLE — different spec method, test separately
