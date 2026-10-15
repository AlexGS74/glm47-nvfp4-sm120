#!/usr/bin/env bash
set -euo pipefail

# Launch Qwen3.8-Flash-Next (NVFP4 QAD, local-inference-lab) on 2x RTX PRO 6000
# via the lab's karmic-kraken image, qwen38-flash-next profile (MTP3).
#
# Recipe: rtx6kpro/models/qwen38-flash-next.md (TP2/DCP1/MTP3 qualified,
#         262144 context, fp8 KV; Qwen is the stable arm of the KK line).
# Image:  karmic-kraken-beta-20260922 (digest pinned below).
#
# Aliases: the KK launcher types served-model-name as a single string, so only
# SERVED_MODEL_NAME (default: sloppus) is served. Canonical-name requests 404.
#
# GPU-exclusive with the other launchers — guard below. ComfyUI on GPU 3 is
# unaffected (this stack pins GPUs 0-1 via GPUS).
#
# Usage:
#   ./run_qwen38.sh                  # start detached, TP2/DCP1 MTP3 on :30000
#   ./run_qwen38.sh up qad-step5500-ple1000   # serve a different HF revision
#   MODE=off ./run_qwen38.sh         # no speculation
#   GPUS=0 TP=1 ./run_qwen38.sh      # single-card trial
#   ./run_qwen38.sh --logs | --status | --stop
#
# Checkpoints (HF revision of local-inference-lab/Qwen3.8-Flash-Next-NVFP4;
# pre-fetch with: HF_HOME=/home/alex/models-main/huggingface hf download
#   local-inference-lab/Qwen3.8-Flash-Next-NVFP4 --revision <rev>):
#   main                    - default cache copy (99G)
#   qad-step5500-ple1000    - 2026-10-03: 91.1 Aider PR2, best checkpoint
#   qad-step-4000           - older QAD step

IMAGE="${IMAGE:-ghcr.io/local-inference-lab/vllm@sha256:bf2c8f5da6f1de82b345367102b528692231a23eadc1b49c180c82c397c77522}"
NAME="qwen38flash"
PORT="${PORT:-30000}"
GPUS="${GPUS:-0,1}"
TP="${TP:-2}"
DCP="${DCP:-1}"
MODE="${MODE:-mtp}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-sloppus}"
REVISION="${2:-}"   # optional HF revision passed after 'up'
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"   # profile default is 262144; pool 4.6M tokens = 4.4x @1M
HUB="${HUB:-/home/alex/models-main/huggingface}"   # boot-NVMe copy 2026-09-24; source left on /models
CACHE_DIR="${CACHE_DIR:-/models/qwen38-flash-next/runtime}"

case "${1:-up}" in
  stop|--stop)
    docker rm -f "$NAME"
    ;;
  logs|--logs)
    docker logs -f "$NAME"
    ;;
  status|--status)
    docker ps -a --filter "name=$NAME" --format "{{.Names}}\t{{.Status}}"
    ;;
  up|"")
    for rival in "glm53flash" "glm53-r34-tp3" "glm53-kraken-tp3" "qwen38-tp2" "ds41-jovian" "deepseek-v41" "mimo26flash"; do
      if docker ps --format '{{.Names}}' | grep -q "$rival"; then
        echo "'$rival' is running and owns GPUs — stop it first." >&2
        exit 1
      fi
    done
    # --gpus wants the device list quoted as one CSV entry; a bare device=0,1
    # splits into two requests (Count + DeviceIDs) and docker rejects it.
    # Clear a leftover container from a failed start; refuse if actually running.
    if [ "$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || true)" = "running" ]; then
      echo "$NAME is already running — $0 --stop first." >&2
      exit 1
    fi
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    mkdir -p "$HUB" "$CACHE_DIR"
    native_args=(--served-model-name "$SERVED_MODEL_NAME")
    [ "$MODE" != "mtp" ] && native_args+=(--mode "$MODE")
    # --revision pins the trunk weights; empty = profile default (main).
    [ -n "$REVISION" ] && native_args+=(--revision "$REVISION")
    # QAD branches carry block32-MXFP8 MTP experts, which the profile's
    # moe_backend=b12x draft cannot run ("b12x is not supported for MXFP8
    # MoE" -> worker crash at speculator load). The community-validated fix
    # (and what the image's own glm53-flash profile does) is an explicit
    # speculative-config with a marlin DRAFT backend -- trunk stays b12x.
    # Explicit config bypasses the runtime's draft-revision derivation, so
    # the revision must be repeated inside the JSON. Stock main keeps the
    # profile default (NVFP4 draft, b12x).
    if [ -n "$REVISION" ] && [ "$MODE" = "mtp" ]; then
      spec_json=$(printf '{"method":"mtp","moe_backend":"marlin","num_speculative_tokens":3,"revision":"%s","draft_sample_method":"probabilistic","rejection_sample_method":"standard"}' "$REVISION")
      native_args+=(--speculative-config "$spec_json")
    fi
    echo "== Qwen3.8-Flash-Next NVFP4: served=$SERVED_MODEL_NAME port=$PORT TP=$TP/DCP$DCP mode=$MODE rev=${REVISION:-main} gpus=$GPUS"
    docker run -d --name "$NAME" \
      --gpus "\"device=$GPUS\"" --ipc=host --network host \
      --ulimit memlock=-1:-1 --shm-size=32g --init \
      --restart no \
      -e PROFILE="qwen38-flash-next" \
      -e HARDWARE_PROFILE="rtx-pro-6000-pcie" \
      -e TP="$TP" -e DCP="$DCP" -e PORT="$PORT" -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
      -v "$HUB:/root/.cache/huggingface:rw" \
      -v "$CACHE_DIR:/cache:rw" \
      "$IMAGE" \
      -- "${native_args[@]}"
    echo
    echo "== API:    http://<this-host>:${PORT}/v1"
    echo "== Follow: $0 --logs   (first boot: download + autotune, 10-30 min)"
    ;;
  *)
    echo "usage: $0 [up [revision] | --logs | --status | --stop]" >&2
    exit 2
    ;;
esac
