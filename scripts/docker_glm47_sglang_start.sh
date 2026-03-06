#!/usr/bin/env bash
set -euo pipefail

# Start GLM-4.7 NVFP4 via SGLang inside voipmonitor/llm-pytorch-blackwell:nightly
# This image has both vLLM and SGLang pre-patched for SM120/Blackwell.
#
# Usage:
#   ./scripts/docker_glm47_sglang_start.sh
#   PORT=8000 ./scripts/docker_glm47_sglang_start.sh

CONTAINER_NAME=${CONTAINER_NAME:-glm47-sglang}
IMAGE=${IMAGE:-voipmonitor/llm-pytorch-blackwell:nightly}
PORT=${PORT:-30000}
TP=${TP:-4}
DTYPE=${DTYPE:-bfloat16}
QUANTIZATION=${QUANTIZATION:-modelopt_fp4}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-claude-opus-4-5-20251001}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-16}
MEM_FRACTION=${MEM_FRACTION:-0.90}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-16}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bf16}
GPU_POWER_LIMIT=${GPU_POWER_LIMIT:-270}
MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-flashinfer_cutlass}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flashinfer}

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

echo "Image:     ${IMAGE}"
echo "Model:     ${MODEL_CACHE_DIR}"
echo "Snapshot:  ${SNAPSHOT_REL}"
echo "Container: ${CONTAINER_NAME}"
echo "Port:      ${PORT}"
echo "Backend:   SGLang  MoE: ${MOE_RUNNER_BACKEND}  Attention: ${ATTENTION_BACKEND}"
echo "TP:        ${TP}  Mem: ${MEM_FRACTION}  Max running: ${MAX_RUNNING_REQUESTS}"
echo ""

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus '"device=0,1,2,3"' \
  --ipc=host \
  --shm-size=16g \
  -p "${PORT}:8000" \
  -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
  -e SGLANG_USE_CUTLASS_BACKEND_FOR_FP4_GEMM=1 \
  -e SGLANG_DISABLE_DEEP_GEMM=1 \
  -e SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=True \
  -e "PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512" \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 \
  -e NCCL_MIN_NCHANNELS=8 \
  -e OMP_NUM_THREADS=8 \
  -e SAFETENSORS_FAST_GPU=1 \
  -v "${MODEL_CACHE_DIR}:/model" \
  -v "${HOME}/.cache/flashinfer:/root/.cache/flashinfer" \
  "${IMAGE}" \
  python -m sglang.launch_server \
  --model-path "${MODEL_CONTAINER_PATH}" \
  --model-impl sglang \
  --tp "${TP}" \
  --trust-remote-code \
  --attention-backend "${ATTENTION_BACKEND}" \
  --moe-runner-backend "${MOE_RUNNER_BACKEND}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --chat-template "${MODEL_CONTAINER_PATH}/chat_template.jinja" \
  --tool-call-parser glm47 \
  --reasoning-parser deepseek-r1 \
  --quantization "${QUANTIZATION}" \
  --dtype "${DTYPE}" \
  --disable-custom-all-reduce \
  --mem-fraction-static "${MEM_FRACTION}" \
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --chunked-prefill-size 512 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 4}' \
  --enable-torch-compile \
  --enable-hierarchical-cache --hicache-ratio 5 \
  --sleep-on-idle \
  --enable-metrics

echo ""
echo "Container started: ${CONTAINER_NAME}"
echo "Stop: docker rm -f ${CONTAINER_NAME}"
echo ""
sleep 1
exec docker logs -f "${CONTAINER_NAME}"
