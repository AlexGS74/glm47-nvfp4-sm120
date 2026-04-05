#!/usr/bin/env bash
set -euo pipefail

# Start GLM-4.7 FP8 block quant (AlexGS74) via SGLang on voipmonitor/sglang:cu130.
# Based on voipmonitor/rtx6kpro GLM-4.7 recipe:
#   https://github.com/voipmonitor/rtx6kpro/blob/master/models/glm47.md
# Uses our own FP8 block quant instead of zai-org/GLM-4.7-FP8.
#
# Features vs NVFP4 script:
#   - EAGLE speculative decoding (3-step, not NEXTN)
#   - Hierarchical KV cache with CPU offload (hicache-ratio 5)
#   - FP8 e4m3 KV cache
#   - Triton FP8 GEMM kernels
#   - Higher mem-fraction (0.95) — FP8 is smaller than NVFP4
#   - Mixed chunked prefill
#   - sleep-on-idle for VRAM recovery
#
# Usage:
#   ./scripts/docker_glm47_sglang_fp8_start.sh
#   ./scripts/docker_glm47_sglang_fp8_start.sh --stop
#   PORT=4999 ./scripts/docker_glm47_sglang_fp8_start.sh

CONTAINER_NAME=${CONTAINER_NAME:-glm47-sglang-fp8}

if [[ "${1:-}" == "--stop" ]]; then
  echo "Stopping container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" 2>/dev/null && echo "Stopped." || echo "Not running."
  exit 0
fi

IMAGE=${IMAGE:-voipmonitor/sglang:cu130}
PORT=${PORT:-30000}
TP=${TP:-4}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-claude-opus-4-5-20251001}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-48}
MEM_FRACTION=${MEM_FRACTION:-0.95}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-8}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
GPU_POWER_LIMIT=${GPU_POWER_LIMIT:-270}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flashinfer}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-200000}
HICACHE_RATIO=${HICACHE_RATIO:-5}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
SCHEDULE_CONSERVATIVENESS=${SCHEDULE_CONSERVATIVENESS:-0.3}

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_CACHE_DIR=""
SNAPSHOT_REL=""
for model_dir in \
  "/data/huggingface/hub/models--AlexGS74--GLM-4.7-FP8-block" \
  "${HOME}/.cache/huggingface/hub/models--AlexGS74--GLM-4.7-FP8-block" \
  "${HOME}/.cache/huggingface/hub/models--zai-org--GLM-4.7-FP8/snapshots" ; do
  # Handle both HF cache layout (snapshots/hash/) and flat layout (snapshots/main/)
  parent="${model_dir}"
  [[ "${parent}" == */snapshots ]] && parent="${parent%/snapshots}"
  for snap in "${parent}/snapshots"/*/; do
    cfg=$(readlink -f "${snap}config.json" 2>/dev/null || true)
    first_shard=$(readlink -f "${snap}model-00001-of-"*.safetensors 2>/dev/null || true)
    if [[ -f "${cfg}" && -n "${first_shard}" && -f "${first_shard}" ]] \
       && sz=$(stat --format=%s "${first_shard}" 2>/dev/null) && [[ "${sz}" -gt 1000000000 ]]; then
      MODEL_CACHE_DIR="${parent}"
      SNAPSHOT_REL="${snap#${parent}/}"
      SNAPSHOT_REL="${SNAPSHOT_REL%/}"
      break 2
    fi
  done
done

if [[ -z "${MODEL_CACHE_DIR}" ]]; then
  echo "ERROR: GLM-4.7 FP8 block quant not found in HF cache" >&2
  echo "  Expected: AlexGS74/GLM-4.7-FP8-block or zai-org/GLM-4.7-FP8" >&2
  exit 1
fi

MODEL_CONTAINER_PATH="/model/${SNAPSHOT_REL}"

# ─────────────────────────────────────────────────────────────────────────────

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Stopping existing container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1
fi

sudo -n nvidia-smi -pl "${GPU_POWER_LIMIT}" -i 0,1,2,3 2>/dev/null \
  && echo "GPU power limit set to ${GPU_POWER_LIMIT}W" \
  || echo "WARNING: could not set GPU power limit" >&2

_W=78
_HR=$(printf '%0.s─' $(seq 1 $((_W+1))))
_line() { printf '│ %-'"${_W}"'s│\n' "$1"; }
_sep()  { printf '├%s┤\n' "${_HR}"; }
_top()  { printf '┌%s┐\n' "${_HR}"; }
_bot()  { printf '└%s┘\n' "${_HR}"; }
_top
_line "GLM-4.7 FP8 block (AlexGS74) -- SGLang on ${IMAGE}"
_sep
_line "Model:       ${MODEL_CACHE_DIR}"
_line "Snapshot:    ${SNAPSHOT_REL}"
_line "Container:   ${CONTAINER_NAME}    Port: ${PORT}"
_line "TP: ${TP}   Mem: ${MEM_FRACTION}   Max running: ${MAX_RUNNING_REQUESTS}   Ctx: ${CONTEXT_LENGTH}"
_line "KV cache:    ${KV_CACHE_DTYPE}   Quant: compressed-tensors (FP8 block)"
_line "Backends:    FP8 GEMM=triton  Attn=${ATTENTION_BACKEND}"
_line "Chunked prefill: ${CHUNKED_PREFILL_SIZE}   Mixed chunk: on"
_line "Spec:        EAGLE 3-step, topk=1, draft=4"
_line "HiCache:     ratio=${HICACHE_RATIO} (CPU offload)"
_line "NCCL:        P2P_LEVEL=4  IB=off"
_line "Env:         TF32=1  OMP_THREADS=8  SAFETENSORS_FAST=1"
_line "Flags:       --disable-shared-experts-fusion --enable-metrics"
_line "             --reasoning-parser glm45 --tool-call-parser glm47"
_line "             --schedule-conservativeness ${SCHEDULE_CONSERVATIVENESS}"
_line "             --sleep-on-idle --enable-cache-report"
_bot
echo ""

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  --shm-size=8g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  -e NCCL_P2P_LEVEL=4 \
  -e NCCL_P2P_DISABLE=0 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 \
  -e NCCL_MIN_NCHANNELS=8 \
  -e NCCL_CUMEM_HOST_ENABLE=0 \
  -e SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=True \
  -e SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  -e SGLANG_ENABLE_DEEP_GEMM=0 \
  -e SGLANG_ENABLE_SPEC_V2=0 \
  -e USE_TRITON_W8A8_FP8_KERNEL=1 \
  -e NVIDIA_TF32_OVERRIDE=1 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e TORCH_COMPILE_DISABLE=1 \
  -e TORCHDYNAMO_DISABLE=1 \
  -e OMP_NUM_THREADS=8 \
  -e SAFETENSORS_FAST_GPU=1 \
  -v "${MODEL_CACHE_DIR}:/model:ro" \
  -v "/data/cache/sglang:/cache/jit" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model "${MODEL_CONTAINER_PATH}" \
  --chat-template "${MODEL_CONTAINER_PATH}/chat_template.jinja" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --tensor-parallel-size "${TP}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --fp8-gemm-backend triton \
  --trust-remote-code \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --enable-mixed-chunk \
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --disable-shared-experts-fusion \
  --disable-custom-all-reduce \
  --schedule-conservativeness "${SCHEDULE_CONSERVATIVENESS}" \
  --enable-metrics \
  --enable-cache-report \
  --sleep-on-idle \
  --host 0.0.0.0 --port "${PORT}" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}' \
  --enable-hierarchical-cache --hicache-ratio "${HICACHE_RATIO}" \
  --enable-flashinfer-allreduce-fusion

echo ""
echo "Container started: ${CONTAINER_NAME}"
echo "Stop: ./scripts/docker_glm47_sglang_fp8_start.sh --stop"
echo ""
sleep 1
exec docker logs -f "${CONTAINER_NAME}"
