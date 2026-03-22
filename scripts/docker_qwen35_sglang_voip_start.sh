#!/usr/bin/env bash
set -euo pipefail

# Start Qwen3.5-397B-A17B-NVFP4 via voipmonitor/sglang:test-cu132.
# Custom SM120 MoE and FP4 kernels (b12x backend) written from scratch.
# 168 tok/s bench, 200-250 tok/s single batch code gen.
# See: https://github.com/voipmonitor/llm-inference-bench
#
# Usage:
#   ./scripts/docker_qwen35_sglang_voip_start.sh
#   ./scripts/docker_qwen35_sglang_voip_start.sh --stop
#   SPEC=0 ./scripts/docker_qwen35_sglang_voip_start.sh          # no speculative decoding
#   PORT=8000 ./scripts/docker_qwen35_sglang_voip_start.sh

CONTAINER_NAME=${CONTAINER_NAME:-qwen35-sglang-voip}

if [[ "${1:-}" == "--stop" ]]; then
  echo "Stopping container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" 2>/dev/null && echo "Stopped." || echo "Not running."
  exit 0
fi

IMAGE=${IMAGE:-voipmonitor/sglang:test-cu132}
PORT=${PORT:-30000}
TP=${TP:-4}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-claude-opus-4-5-20251001}
MEM_FRACTION=${MEM_FRACTION:-0.90}
GPU_POWER_LIMIT=${GPU_POWER_LIMIT:-270}
# Speculative decoding: 1=on (NEXTN 5-step), 0=off
SPEC=${SPEC:-0}

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_CACHE_DIR=""
SNAPSHOT_REL=""
for model_dir in \
  "/data/huggingface/hub/models--lukealonso--Qwen3.5-397B-A17B-NVFP4" \
  "${HOME}/.cache/huggingface/hub/models--lukealonso--Qwen3.5-397B-A17B-NVFP4" \
  "/data/huggingface/hub/models--Sehyo--Qwen3.5-397B-A17B-NVFP4" \
  "${HOME}/.cache/huggingface/hub/models--Sehyo--Qwen3.5-397B-A17B-NVFP4"; do
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
  echo "ERROR: Qwen3.5-397B-A17B-NVFP4 not found in HF cache" >&2
  echo "  Run: hf download lukealonso/Qwen3.5-397B-A17B-NVFP4" >&2
  exit 1
fi

MODEL_HOST_PATH="${MODEL_CACHE_DIR}"
MODEL_CONTAINER_PATH="/model/${SNAPSHOT_REL}"

# ── Pre-flight checks ────────────────────────────────────────────────────────
if ! grep -q 'iommu=pt' /proc/cmdline 2>/dev/null; then
  echo "WARNING: iommu=pt not in kernel params. Recommended for Threadripper P2P."
fi

# ─────────────────────────────────────────────────────────────────────────────

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Stopping existing container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1
fi

sudo -n nvidia-smi -pl "${GPU_POWER_LIMIT}" -i 0,1,2,3 2>/dev/null \
  && echo "GPU power limit set to ${GPU_POWER_LIMIT}W" \
  || echo "WARNING: could not set GPU power limit" >&2

SPEC_ARGS=""
if [[ "${SPEC}" -gt 0 ]]; then
  SPEC_ARGS="--speculative-algo NEXTN --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6"
fi

echo "Image:     ${IMAGE}"
echo "Model:     ${MODEL_CACHE_DIR}"
echo "Snapshot:  ${SNAPSHOT_REL}"
echo "Container: ${CONTAINER_NAME}"
echo "Port:      ${PORT}"
echo "TP:        ${TP}  Mem fraction: ${MEM_FRACTION}"
echo "Spec:      SPEC=${SPEC} (1=NEXTN 5-step, 0=off)"
echo ""

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  --shm-size=8g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  -e SGLANG_ENABLE_SPEC_V2=True \
  -e NCCL_P2P_DISABLE=0 \
  -e NCCL_IB_DISABLE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "${MODEL_HOST_PATH}:/model:ro" \
  -v "/data/cache/sglang:/cache/jit" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model "${MODEL_CONTAINER_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --tensor-parallel-size "${TP}" \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --cuda-graph-max-bs 16 \
  --max-running-requests 16 \
  --chunked-prefill-size 4096 \
  --mamba-scheduler-strategy extra_buffer \
  --mem-fraction-static "${MEM_FRACTION}" \
  --host 0.0.0.0 --port "${PORT}" \
  --disable-custom-all-reduce \
  --enable-metrics \
  --schedule-conservativeness 0.1 \
  --attention-backend flashinfer \
  --fp4-gemm-backend b12x \
  --moe-runner-backend b12x \
  ${SPEC_ARGS}

echo ""
echo "Container started: ${CONTAINER_NAME}"
echo "Stop: ./scripts/docker_qwen35_sglang_voip_start.sh --stop"
echo ""
sleep 1
exec docker logs -f "${CONTAINER_NAME}"
