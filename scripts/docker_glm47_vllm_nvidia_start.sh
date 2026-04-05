#!/usr/bin/env bash
set -euo pipefail

# Start GLM-4.7 NVFP4 (nvidia quant) via vLLM on voipmonitor/vllm:cu130.
# vLLM supports concurrent request batching (unlike SGLang MTP).
#
# Usage:
#   ./scripts/docker_glm47_vllm_nvidia_start.sh
#   ./scripts/docker_glm47_vllm_nvidia_start.sh --stop
#   SPEC=2 ./scripts/docker_glm47_vllm_nvidia_start.sh          # MTP with 2 spec tokens
#   SPEC=0 ./scripts/docker_glm47_vllm_nvidia_start.sh          # no MTP
#   PORT=8000 ./scripts/docker_glm47_vllm_nvidia_start.sh

CONTAINER_NAME=${CONTAINER_NAME:-glm47-vllm-nvidia}

if [[ "${1:-}" == "--stop" ]]; then
  echo "Stopping container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" 2>/dev/null && echo "Stopped." || echo "Not running."
  exit 0
fi

IMAGE=${IMAGE:-voipmonitor/vllm:cu130}
PORT=${PORT:-30000}
TP=${TP:-4}
DTYPE=${DTYPE:-bfloat16}
QUANTIZATION=${QUANTIZATION:-modelopt_fp4}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-claude-opus-4-5-20251001}
MEM_FRACTION=${MEM_FRACTION:-0.80}
GPU_POWER_LIMIT=${GPU_POWER_LIMIT:-270}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-200000}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-TRITON_ATTN}
# MTP speculative decoding: 0=off, 2=recommended sweet spot
SPEC=${SPEC:-0}

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_CACHE_DIR=""
SNAPSHOT_REL=""
for model_dir in \
  "/data/huggingface/hub/models--nvidia--GLM-4.7-NVFP4" \
  "${HOME}/.cache/huggingface/hub/models--nvidia--GLM-4.7-NVFP4" \
  "/data/huggingface/hub/models--Tengyunw--GLM-4.7-NVFP4" \
  "${HOME}/.cache/huggingface/hub/models--Tengyunw--GLM-4.7-NVFP4"; do
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

# ── Pre-flight checks ────────────────────────────────────────────────────────
if ! grep -q 'iommu=pt' /proc/cmdline 2>/dev/null; then
  echo "WARNING: iommu=pt not in kernel params. Recommended for Threadripper P2P."
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─────────────────────────────────────────────────────────────────────────────

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Stopping existing container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1
fi

sudo -n nvidia-smi -pl "${GPU_POWER_LIMIT}" -i 0,1,2,3 2>/dev/null \
  && echo "GPU power limit set to ${GPU_POWER_LIMIT}W" \
  || echo "WARNING: could not set GPU power limit" >&2

SPEC_FLAGS=""
if [[ "${SPEC}" -gt 0 ]]; then
  SPEC_FLAGS="--speculative-config.method mtp --speculative-config.num_speculative_tokens ${SPEC}"
fi

_W=78
_HR=$(printf '%0.s─' $(seq 1 $((_W+1))))
_line() { printf '│ %-'"${_W}"'s│\n' "$1"; }
_sep()  { printf '├%s┤\n' "${_HR}"; }
_top()  { printf '┌%s┐\n' "${_HR}"; }
_bot()  { printf '└%s┘\n' "${_HR}"; }
_top
_line "GLM-4.7 NVFP4 (nvidia) -- vLLM on ${IMAGE}"
_sep
_line "Model:       ${MODEL_CACHE_DIR}"
_line "Snapshot:    ${SNAPSHOT_REL}"
_line "Container:   ${CONTAINER_NAME}    Port: ${PORT}"
_line "TP: ${TP}   Mem: ${MEM_FRACTION}   Max seqs: ${MAX_NUM_SEQS}   Max len: ${MAX_MODEL_LEN}"
_line "KV cache:    ${KV_CACHE_DTYPE}   Quant: ${QUANTIZATION}   Dtype: ${DTYPE}"
_line "Attention:   ${ATTENTION_BACKEND}"
_line "Batched tokens: ${MAX_NUM_BATCHED_TOKENS}   Spec: SPEC=${SPEC} (0=off, 2=MTP)"
_line "NCCL:        P2P=on  IB=off  P2P_LEVEL=SYS"
_line "Env:         DEEP_GEMM=off  FLASHINFER_MOE_FP4=off  FLASHINFER_MOE_FP16=on"
_line "Flags:       --trust-remote-code --enable-auto-tool-choice"
_line "             --reasoning-parser glm45 --tool-call-parser glm47"
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
  -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_USE_FLASHINFER_MOE_FP16=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
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
  -v "${SCRIPT_DIR}/patches:/patches:ro" \
  --entrypoint bash \
  "${IMAGE}" \
  -c "python3 /patches/glm4_moe_skip_scales.py /opt/vllm/vllm/model_executor/models/glm4_moe.py \
  && exec python -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_CONTAINER_PATH} \
  --host 0.0.0.0 \
  --port ${PORT} \
  --served-model-name ${SERVED_MODEL_NAME} \
  --dtype ${DTYPE} \
  --quantization ${QUANTIZATION} \
  --tensor-parallel-size ${TP} \
  --attention-backend ${ATTENTION_BACKEND} \
  --gpu-memory-utilization ${MEM_FRACTION} \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-seqs ${MAX_NUM_SEQS} \
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
  --kv-cache-dtype ${KV_CACHE_DTYPE} \
  --chat-template ${MODEL_CONTAINER_PATH}/chat_template.jinja \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  ${SPEC_FLAGS}"

echo ""
echo "Container started: ${CONTAINER_NAME}"
echo "Stop: ./scripts/docker_glm47_vllm_nvidia_start.sh --stop"
echo ""
sleep 1
exec docker logs -f "${CONTAINER_NAME}"
