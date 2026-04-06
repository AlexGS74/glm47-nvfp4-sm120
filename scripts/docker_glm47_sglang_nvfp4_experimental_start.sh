#!/usr/bin/env bash
set -euo pipefail

# Start GLM-4.7 NVFP4 (nvidia ModelOpt FP4) via SGLang on voipmonitor/sglang:cu130.
# Based on the FP8 docker script, adapted for NVFP4 quantization.
#
# Differences from FP8:
#   - quantization: modelopt_fp4 (not compressed-tensors FP8)
#   - dtype: bfloat16 (QK-norm float32 dispatch fails with half on SM120)
#   - KV cache: bf16 (not fp8_e4m3)
#   - MoE backend: flashinfer_cutlass (flashinfer_trtllm SM100-only cubins crash on SM120)
#   - CUTLASS FP4 GEMM env var instead of Triton FP8
#   - DeepGEMM disabled (non-ue8m0 scale format)
#   - Lower mem-fraction (0.90 — NVFP4 model is larger than FP8)
#
# Usage:
#   ./scripts/docker_glm47_sglang_nvfp4_start.sh
#   ./scripts/docker_glm47_sglang_nvfp4_start.sh --stop
#   PORT=4999 ./scripts/docker_glm47_sglang_nvfp4_start.sh

CONTAINER_NAME=${CONTAINER_NAME:-glm47-sglang-nvfp4}

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
MEM_FRACTION=${MEM_FRACTION:-0.90}
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
  "/data/huggingface/hub/models--nvidia--GLM-4.7-NVFP4" \
  "${HOME}/.cache/huggingface/hub/models--nvidia--GLM-4.7-NVFP4" ; do
  for snap in "${model_dir}/snapshots"/*/; do
    cfg=$(readlink -f "${snap}config.json" 2>/dev/null || true)
    first_shard=$(readlink -f "${snap}model-00001-of-"*.safetensors 2>/dev/null || true)
    if [[ -f "${cfg}" && -n "${first_shard}" && -f "${first_shard}" ]] \
       && sz=$(stat --format=%s "${first_shard}" 2>/dev/null) && [[ "${sz}" -gt 1000000000 ]]; then
      MODEL_CACHE_DIR="${model_dir}"
      SNAPSHOT_REL="${snap#${model_dir}/}"
      SNAPSHOT_REL="${SNAPSHOT_REL%/}"
      break 2
    fi
  done
done

if [[ -z "${MODEL_CACHE_DIR}" ]]; then
  echo "ERROR: GLM-4.7 NVFP4 not found in HF cache" >&2
  echo "  Expected: nvidia/GLM-4.7-NVFP4" >&2
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
_line "GLM-4.7 NVFP4 (nvidia) -- SGLang on ${IMAGE}"
_sep
_line "Model:       ${MODEL_CACHE_DIR}"
_line "Snapshot:    ${SNAPSHOT_REL}"
_line "Container:   ${CONTAINER_NAME}    Port: ${PORT}"
_line "TP: ${TP}   Mem: ${MEM_FRACTION}   Max running: ${MAX_RUNNING_REQUESTS}   Ctx: ${CONTEXT_LENGTH}"
_line "KV cache:    ${KV_CACHE_DTYPE}   Quant: modelopt_fp4 (NVFP4)"
_line "Backends:    FP4 GEMM=cutlass  MoE=flashinfer_cutlass  Attn=${ATTENTION_BACKEND}"
_line "Chunked prefill: ${CHUNKED_PREFILL_SIZE}   Mixed chunk: on"
_line "Spec:        EAGLE 3-step, topk=1, draft=4"
_line "HiCache:     ratio=${HICACHE_RATIO} (CPU offload)"
_line "NCCL:        P2P_LEVEL=4"
_line "Env:         CUTLASS_FP4_GEMM=1  DeepGEMM=off"
_line "Flags:       --disable-shared-experts-fusion --enable-metrics"
_line "             --reasoning-parser glm45 --tool-call-parser glm47"
_line "             --schedule-conservativeness ${SCHEDULE_CONSERVATIVENESS}"
_line "             --sleep-on-idle --enable-cache-report"
_bot
echo ""

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus '"device=0,1,2,3"' \
  --ipc=host \
  --shm-size=8g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  -e NCCL_P2P_LEVEL=4 \
  -e SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=True \
  -e SGLANG_USE_CUTLASS_BACKEND_FOR_FP4_GEMM=1 \
  -e SGLANG_DISABLE_DEEP_GEMM=1 \
  -e SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  -e SGLANG_ENABLE_DEEP_GEMM=0 \
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
  --quantization modelopt_fp4 \
  --dtype bfloat16 \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --moe-runner-backend flashinfer_cutlass \
  --trust-remote-code \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --enable-mixed-chunk \
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --disable-shared-experts-fusion \
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
echo "Stop: ./scripts/docker_glm47_sglang_nvfp4_start.sh --stop"
echo ""
sleep 1
exec docker logs -f "${CONTAINER_NAME}"
