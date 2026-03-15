#!/usr/bin/env bash
set -euo pipefail

# Start GLM-4.7 NVFP4 via vLLM inside voipmonitor/llm-pytorch-blackwell:nightly
# This image has both vLLM and SGLang pre-patched for SM120/Blackwell.
#
# Usage:
#   ./scripts/docker_glm47_vllm_start.sh
#   PORT=8000 ./scripts/docker_glm47_vllm_start.sh

CONTAINER_NAME=${CONTAINER_NAME:-glm47-vllm}

if [[ "${1:-}" == "--stop" ]]; then
  echo "Stopping container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" 2>/dev/null && echo "Stopped." || echo "Not running."
  exit 0
fi

IMAGE=${IMAGE:-voipmonitor/llm-pytorch-blackwell:nightly}
PORT=${PORT:-30000}
TP=${TP:-4}
DTYPE=${DTYPE:-bfloat16}
QUANTIZATION=${QUANTIZATION:-modelopt_fp4}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-claude-opus-4-5-20251001}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-200000}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.80}
SWAP_SPACE=${SWAP_SPACE:-16}
GPU_POWER_LIMIT=${GPU_POWER_LIMIT:-270}
SPEC_TOKENS=${SPEC_TOKENS:-0}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-TRITON_ATTN}

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_CACHE_DIR="${HOME}/.cache/huggingface/hub/models--Salyut1--GLM-4.7-NVFP4"
SNAPSHOT_REL=""
for snap in "${MODEL_CACHE_DIR}/snapshots"/*/; do
  cfg=$(readlink -f "${snap}config.json" 2>/dev/null || true)
  first_shard=$(readlink -f "${snap}model-00001-of-"*.safetensors 2>/dev/null || true)
  if [[ -f "${cfg}" && -n "${first_shard}" && -f "${first_shard}" ]]; then
    SNAPSHOT_REL="${snap#${MODEL_CACHE_DIR}/}"
    SNAPSHOT_REL="${SNAPSHOT_REL%/}"
    break
  fi
done

if [[ -z "${SNAPSHOT_REL}" ]]; then
  echo "ERROR: GLM-4.7-NVFP4 not found in HF cache" >&2
  echo "  Run: hf download Salyut1/GLM-4.7-NVFP4" >&2
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

SPEC_ARG=""
if [[ "${SPEC_TOKENS}" -gt 0 ]]; then
  SPEC_ARG="--speculative-config.method mtp --speculative-config.num_speculative_tokens ${SPEC_TOKENS}"
fi

echo "Image:     ${IMAGE}"
echo "Model:     ${MODEL_CACHE_DIR}"
echo "Snapshot:  ${SNAPSHOT_REL}"
echo "Container: ${CONTAINER_NAME}"
echo "Port:      ${PORT}"
echo "Backend:   vLLM  Attention: ${ATTENTION_BACKEND}"
echo "TP:        ${TP}  Max len: ${MAX_MODEL_LEN}  GPU util: ${GPU_MEM_UTIL}"
echo ""

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus '"device=0,1,2,3"' \
  --ipc=host \
  --shm-size=16g \
  -p "${PORT}:8000" \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_USE_FLASHINFER_MOE_FP16=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_SLEEP_WHEN_IDLE=1 \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e NCCL_P2P_LEVEL=4 \
  -e NCCL_IB_DISABLE=1 \
  -e OMP_NUM_THREADS=4 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e "PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512" \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor \
  -v "${MODEL_CACHE_DIR}:/model" \
  -v "${HOME}/.cache/torch:/root/.cache/torch" \
  -v "${HOME}/.cache/vllm:/root/.cache/vllm" \
  "${IMAGE}" \
  python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_CONTAINER_PATH}" \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --dtype "${DTYPE}" \
  --quantization "${QUANTIZATION}" \
  --tensor-parallel-size "${TP}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --swap-space "${SWAP_SPACE}" \
  --kv-cache-dtype auto \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --chat-template "${MODEL_CONTAINER_PATH}/chat_template.jinja" \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  ${SPEC_ARG}

echo ""
echo "Container started: ${CONTAINER_NAME}"
echo "Stop: docker rm -f ${CONTAINER_NAME}"
echo ""
sleep 1
exec docker logs -f "${CONTAINER_NAME}"
