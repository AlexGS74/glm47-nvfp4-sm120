#!/usr/bin/env bash
set -euo pipefail

# Start GLM-4.7 NVFP4 (nvidia quant) via SGLang on voipmonitor/sglang:cu130.
# b12x backends for single-user decode; override MOE_RUNNER_BACKEND=cutlass for 4+ concurrent.
#
# Usage:
#   ./scripts/docker_glm47_sglang_nvidia_start.sh
#   ./scripts/docker_glm47_sglang_nvidia_start.sh --stop
#   PORT=8000 ./scripts/docker_glm47_sglang_nvidia_start.sh

CONTAINER_NAME=${CONTAINER_NAME:-glm47-sglang-nvidia}

if [[ "${1:-}" == "--stop" ]]; then
  echo "Stopping container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" 2>/dev/null && echo "Stopped." || echo "Not running."
  exit 0
fi

IMAGE=${IMAGE:-voipmonitor/sglang:cu130}
PORT=${PORT:-30000}
TP=${TP:-4}
DTYPE=${DTYPE:-bfloat16}
QUANTIZATION=${QUANTIZATION:-modelopt_fp4}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-claude-opus-4-5-20251001}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-16}
MEM_FRACTION=${MEM_FRACTION:-0.80}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-8}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bf16}
GPU_POWER_LIMIT=${GPU_POWER_LIMIT:-270}
MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-b12x}
FP4_GEMM_BACKEND=${FP4_GEMM_BACKEND:-b12x}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flashinfer}

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_CACHE_DIR=""
SNAPSHOT_REL=""
for model_dir in \
  "/data/huggingface/hub/models--nvidia--GLM-4.7-NVFP4" \
  "${HOME}/.cache/huggingface/hub/models--nvidia--GLM-4.7-NVFP4"; do
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
  echo "ERROR: nvidia/GLM-4.7-NVFP4 not found in HF cache" >&2
  echo "  Run: HF_HOME=/data/huggingface hf download nvidia/GLM-4.7-NVFP4" >&2
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

_W=78  # inner width between │ chars
_HR=$(printf '%0.s─' $(seq 1 $((_W+2))))  # _W+1 dashes: space + content
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
_line "TP: ${TP}   Mem: ${MEM_FRACTION}   Max running: ${MAX_RUNNING_REQUESTS}   CUDA graphs: max_bs=${CUDA_GRAPH_MAX_BS}"
_line "KV cache:    ${KV_CACHE_DTYPE}   Quant: ${QUANTIZATION}"
_line "Backends:    MoE=${MOE_RUNNER_BACKEND}  FP4=${FP4_GEMM_BACKEND}  Attn=${ATTENTION_BACKEND}"
_line "Chunked prefill: 4096"
_line "NCCL:        P2P=SYS  IB=off  LL_BUFFERS=1  MIN_CH=8"
_line "Env:         TF32=1  OMP_THREADS=8  SAFETENSORS_FAST=1"
_line "Flags:       --disable-custom-all-reduce --enable-metrics"
_line "             --reasoning-parser glm45 --tool-call-parser glm47"
_line "             --schedule-conservativeness 0.1"
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
  -e CUDA_HOME=/usr/local/cuda \
  -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
  -e TORCH_COMPILE_DISABLE=1 \
  -e TORCHDYNAMO_DISABLE=1 \
  -e SGLANG_ENABLE_SPEC_V2=True \
  -e SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=True \
  -e SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  -e SGLANG_ENABLE_DEEP_GEMM=0 \
  -e NVIDIA_TF32_OVERRIDE=1 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_P2P_DISABLE=0 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 \
  -e NCCL_MIN_NCHANNELS=8 \
  -e NCCL_CUMEM_HOST_ENABLE=0 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e OMP_NUM_THREADS=8 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "${MODEL_CACHE_DIR}:/model:ro" \
  -v "/data/cache/sglang:/cache/jit" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model "${MODEL_CONTAINER_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --tensor-parallel-size "${TP}" \
  --quantization "${QUANTIZATION}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --trust-remote-code \
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --chunked-prefill-size 4096 \
  --mem-fraction-static "${MEM_FRACTION}" \
  --host 0.0.0.0 --port "${PORT}" \
  --disable-custom-all-reduce \
  --enable-metrics \
  --schedule-conservativeness 0.1 \
  --attention-backend "${ATTENTION_BACKEND}" \
  --fp4-gemm-backend "${FP4_GEMM_BACKEND}" \
  --moe-runner-backend "${MOE_RUNNER_BACKEND}"

echo ""
echo "Container started: ${CONTAINER_NAME}"
echo "Stop: ./scripts/docker_glm47_sglang_nvidia_start.sh --stop"
echo ""
sleep 1
exec docker logs -f "${CONTAINER_NAME}"
