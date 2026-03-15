#!/usr/bin/env bash
set -euo pipefail

# Start Qwen3.5-397B-A17B-NVFP4 via verdictai/vllm-blackwell-k64 Docker image.
# This image has custom CUTLASS K=64 kernel fix for SM120 (99KB SMEM).
# See: https://github.com/flashinfer-ai/flashinfer/pull/2786
#
# Requirements:
#   - NVIDIA Driver 595+ (apt install nvidia-open from CUDA repo)
#   - CUDA 13.2+ (bundled in the Docker image)
#   - iommu=pt in kernel boot params (for Threadripper)
#   - Model: lukealonso/Qwen3.5-397B-A17B-NVFP4 (or Sehyo variant)
#
# Usage:
#   ./scripts/docker_qwen35_k64_start.sh
#   ./scripts/docker_qwen35_k64_start.sh --stop
#   SPEC_TOKENS=3 ./scripts/docker_qwen35_k64_start.sh   # MTP=3 for multi-user
#   PORT=8000 ./scripts/docker_qwen35_k64_start.sh

CONTAINER_NAME=${CONTAINER_NAME:-qwen35-k64}

if [[ "${1:-}" == "--stop" ]]; then
  echo "Stopping container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" 2>/dev/null && echo "Stopped." || echo "Not running."
  exit 0
fi

IMAGE=${IMAGE:-verdictai/vllm-blackwell-k64:latest}
PORT=${PORT:-30000}
TP=${TP:-4}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-claude-opus-4-5-20251001}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
GPU_POWER_LIMIT=${GPU_POWER_LIMIT:-270}
# MTP=5 for single-user (max throughput), MTP=3 for multi-user (stability)
SPEC_TOKENS=${SPEC_TOKENS:-5}

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_CACHE_DIR=""
SNAPSHOT_REL=""
for model_dir in \
  "/data/huggingface/hub/models--lukealonso--Qwen3.5-397B-A17B-NVFP4" \
  "${HOME}/.cache/huggingface/hub/models--Sehyo--Qwen3.5-397B-A17B-NVFP4" \
  "${HOME}/.cache/huggingface/hub/models--lukealonso--Qwen3.5-397B-A17B-NVFP4"; do
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

# Mount full cache dir (symlinks in snapshots/ point to ../../blobs)
MODEL_HOST_PATH="${MODEL_CACHE_DIR}"
MODEL_CONTAINER_PATH="/model/${SNAPSHOT_REL}"

# ── Pre-flight checks ────────────────────────────────────────────────────────
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
DRIVER_MAJOR=${DRIVER_VER%%.*}
if [[ -n "${DRIVER_MAJOR}" && "${DRIVER_MAJOR}" -lt 595 ]]; then
  echo "WARNING: Driver ${DRIVER_VER} detected. K=64 kernel works best with driver 595+"
  echo "  Install: sudo apt install nvidia-open (from CUDA 13.2 repo)"
fi

if ! grep -q 'iommu=pt' /proc/cmdline 2>/dev/null; then
  echo "WARNING: iommu=pt not in kernel params. Recommended for Threadripper P2P."
  echo "  Add 'iommu=pt' to GRUB_CMDLINE_LINUX_DEFAULT in /etc/default/grub"
fi

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
  SPEC_ARG="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}'"
fi

echo "Image:     ${IMAGE}"
echo "Model:     ${MODEL_CACHE_DIR}"
echo "Snapshot:  ${SNAPSHOT_REL}"
echo "Container: ${CONTAINER_NAME}"
echo "Port:      ${PORT}"
echo "Driver:    ${DRIVER_VER:-unknown} (recommended: 595+)"
echo "TP:        ${TP}  Max len: ${MAX_MODEL_LEN}  GPU util: ${GPU_MEM_UTIL}"
echo "MTP:       SPEC_TOKENS=${SPEC_TOKENS} (5=single-user, 3=multi-user)"
echo ""

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus '"device=0,1,2,3"' \
  --ipc=host \
  --shm-size=32g \
  -p "${PORT}:8000" \
  -e NCCL_P2P_DISABLE=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e OMP_NUM_THREADS=6 \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  -e SAFETENSORS_FAST_GPU=1 \
  -v "${MODEL_HOST_PATH}:/model:ro" \
  -v "${HOME}/.cache/torch:/root/.cache/torch" \
  -v "${HOME}/.cache/vllm:/root/.cache/vllm" \
  --entrypoint bash \
  "${IMAGE}" \
  -c "
# Run image's K=64 patch and AOT install, then launch with our args
FI=/opt/venv/lib/python3.12/site-packages/flashinfer
python3 -c \"
p = '\$FI/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch_tma_ws.h'
with open(p) as f: c = f.read()
old = '(TileM == 256 && TileN == 128 && TileK == 128);'
new = '''(TileM == 256 && TileN == 128 && TileK == 128) ||
         (TileM == 128 && TileN == 128 && TileK == 64) ||
         (TileM == 128 && TileN == 256 && TileK == 64) ||
         (TileM == 256 && TileN == 128 && TileK == 64);'''
if 'TileK == 64' not in c:
    c = c.replace(old, new, 1)
    with open(p, 'w') as f: f.write(c)
    print('K=64 validation: APPLIED')
else:
    print('K=64 validation: already present')
\"
AOT_DIR=\"/cache/jit/flashinfer/.cache/flashinfer/0.6.6/120a/cached_ops/fused_moe_120\"
mkdir -p \"\$AOT_DIR\"
cp /tmp/k64_aot.so \"\$AOT_DIR/fused_moe_120.so\"
echo 'K=64 AOT .so installed'

exec python3 -m vllm.entrypoints.openai.api_server \\
  --model '${MODEL_CONTAINER_PATH}' \\
  --served-model-name '${SERVED_MODEL_NAME}' \\
  --host 0.0.0.0 --port 8000 \\
  --trust-remote-code \\
  --tensor-parallel-size ${TP} \\
  --gpu-memory-utilization ${GPU_MEM_UTIL} \\
  --max-model-len ${MAX_MODEL_LEN} \\
  --max-num-batched-tokens 4096 \\
  --max-num-seqs 128 \\
  --kv-cache-dtype auto \\
  --enable-prefix-caching \\
  --reasoning-parser qwen3 \\
  --enable-auto-tool-choice \\
  --tool-call-parser qwen3_coder \\
  --chat-template '${MODEL_CONTAINER_PATH}/chat_template.jinja' \\
  --mm-encoder-tp-mode data --mm-processor-cache-type shm ${SPEC_ARG}
"

echo ""
echo "Container started: ${CONTAINER_NAME}"
echo "Stop: ./scripts/docker_qwen35_k64_start.sh --stop"
echo ""
sleep 1
exec docker logs -f "${CONTAINER_NAME}"
