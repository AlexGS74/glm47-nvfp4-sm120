#!/usr/bin/env bash
set -euo pipefail

# Start Qwen3.5-397B-A17B-AWQ-4bit via orthozany/vllm-qwen35-mtp Docker image.
# Model: cyankiwi/Qwen3.5-397B-A17B-AWQ-4bit (compressed-tensors, auto-detected)
# Same patched vLLM image as the NVFP4 script for fair comparison.
#
# Usage:
#   ./scripts/docker_qwen35_awq_start.sh
#   ./scripts/docker_qwen35_awq_start.sh --stop
#   SPEC_TOKENS=2 ./scripts/docker_qwen35_awq_start.sh

CONTAINER_NAME=${CONTAINER_NAME:-qwen35-awq}
IMAGE=${IMAGE:-orthozany/vllm-qwen35-mtp:latest}
PORT=${PORT:-30000}
TP=${TP:-4}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-claude-opus-4-5-20251001}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.80}
GPU_POWER_LIMIT=${GPU_POWER_LIMIT:-270}
SPEC_TOKENS=${SPEC_TOKENS:-0}

# ── Stop mode ────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
  echo "Stopping container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" 2>/dev/null && echo "Stopped." || echo "Not running."
  exit 0
fi

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_CACHE_DIR=""
SNAPSHOT_REL=""
for model_dir in \
  "${HOME}/.cache/huggingface/hub/models--cyankiwi--Qwen3.5-397B-A17B-AWQ-4bit"; do
  snap=$(ls -d "${model_dir}/snapshots"/*/ 2>/dev/null | head -1 || true)
  [[ -z "${snap}" ]] && continue
  cfg=$(readlink -f "${snap}config.json" 2>/dev/null || true)
  first_shard=$(readlink -f "${snap}model-00001-of-"*.safetensors 2>/dev/null || true)
  if [[ -f "${cfg}" && -n "${first_shard}" && -f "${first_shard}" ]] \
     && sz=$(stat --format=%s "${first_shard}" 2>/dev/null) && [[ "${sz}" -gt 1000000000 ]]; then
    MODEL_CACHE_DIR="${model_dir}"
    SNAPSHOT_REL="${snap#${model_dir}/}"
    SNAPSHOT_REL="${SNAPSHOT_REL%/}"
    break
  fi
done

if [[ -z "${MODEL_CACHE_DIR}" ]]; then
  echo "ERROR: Qwen3.5-397B-A17B-AWQ-4bit not found in HF cache" >&2
  echo "  Run: hf download cyankiwi/Qwen3.5-397B-A17B-AWQ-4bit" >&2
  exit 1
fi

MODEL_HOST_PATH="${MODEL_CACHE_DIR}"
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
  SPEC_ARG="--speculative-config {\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}"
fi

echo "Image:     ${IMAGE}"
echo "Model:     ${MODEL_CACHE_DIR}"
echo "Snapshot:  ${SNAPSHOT_REL}"
echo "Container: ${CONTAINER_NAME}"
echo "Port:      ${PORT}"
echo "Quant:     compressed-tensors (auto-detected)"
echo "TP:        ${TP}  Max len: ${MAX_MODEL_LEN}  GPU util: ${GPU_MEM_UTIL}"
echo "MTP:       SPEC_TOKENS=${SPEC_TOKENS} (method=qwen3_next_mtp)"
echo ""

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus '"device=0,1,2,3"' \
  --ipc=host \
  --shm-size=16g \
  -p "${PORT}:8000" \
  -e NCCL_P2P_LEVEL=4 \
  -e NCCL_IB_DISABLE=1 \
  -e OMP_NUM_THREADS=8 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_SLEEP_WHEN_IDLE=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_LOG_STATS_INTERVAL=1 \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor \
  -v "${MODEL_HOST_PATH}:/model" \
  -v "/data/cache/torch:/root/.cache/torch" \
  -v "/data/cache/vllm:/root/.cache/vllm" \
  -v "${HOME}/mllm/glm47-nvfp4-sm120/patches/moe-configs/E=512,N=256,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition.json:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/E=512,N=256,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition.json:ro" \
  "${IMAGE}" \
  --model "${MODEL_CONTAINER_PATH}" \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens 4092 \
  --max-num-seqs 128 \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --enable-auto-tool-choice \
  --chat-template "${MODEL_CONTAINER_PATH}/chat_template.jinja" \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  ${SPEC_ARG}

echo ""
echo "Container started: ${CONTAINER_NAME}"
echo "Stop: ./scripts/docker_qwen35_awq_start.sh --stop"
echo ""
sleep 1
exec docker logs -f "${CONTAINER_NAME}"
