#!/usr/bin/env bash
# Benchmark GLM-4.7 serving throughput at various concurrency levels.
# Hits vLLM directly on port 30000 (bypass buster-ripper).
# Works with any quant (AWQ, NVFP4, etc.) — tokenizer auto-detected from local cache.
#
# Usage:
#   ./scripts/bench_serving.sh
#   SPEC_TOKENS_LABEL=nvfp4-mtp ./scripts/bench_serving.sh
#   TOKENIZER_PATH=/path/to/model ./scripts/bench_serving.sh

set -euo pipefail

VLLM_BIN=${VLLM_BIN:-${HOME}/.local/share/uv/tools/vllm/bin/vllm}
MODEL_FAMILY=${MODEL_FAMILY:-${1:-}}

# Auto-detect tokenizer from MODEL_FAMILY
if [[ -z "${TOKENIZER_PATH:-}" ]]; then
  case "${MODEL_FAMILY}" in
    qwen|qwen35)
      TOKENIZER_PATH=$(ls -d "${HOME}/.cache/huggingface/hub/models--nvidia--Qwen3.5-397B-A17B-NVFP4/snapshots/"*/ 2>/dev/null | head -1 || true)
      TOKENIZER_PATH=${TOKENIZER_PATH:-$(ls -d "${HOME}/.cache/huggingface/hub/models--Sehyo--Qwen3.5-397B-A17B-NVFP4/snapshots/"*/ 2>/dev/null | head -1 || true)}
      ;;
    glm|glm47)
      TOKENIZER_PATH=$(ls -d "${HOME}/.cache/huggingface/hub/models--Salyut1--GLM-4.7-NVFP4/snapshots/"*/ 2>/dev/null | head -1 || true)
      TOKENIZER_PATH=${TOKENIZER_PATH:-$(ls -d "${HOME}/.cache/huggingface/models--QuantTrio--GLM-4.7-AWQ/snapshots/"*/ 2>/dev/null | head -1 || true)}
      TOKENIZER_PATH=${TOKENIZER_PATH:-$(ls -d "${HOME}/.cache/huggingface/hub/models--zai-org--GLM-4.7-FP8/snapshots/"*/ 2>/dev/null | head -1 || true)}
      ;;
    *)
      echo "Usage: $0 <qwen|glm>" >&2
      echo "  Or set TOKENIZER_PATH directly." >&2
      exit 1
      ;;
  esac
  if [[ -z "${TOKENIZER_PATH:-}" ]]; then
    echo "ERROR: no cached tokenizer found for '${MODEL_FAMILY}'" >&2
    exit 1
  fi
fi
BASE_URL=${BASE_URL:-http://localhost:30000}
MODEL=${MODEL:-claude-opus-4-5-20251001}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-256}
NUM_PROMPTS=${NUM_PROMPTS:-32}
LABEL=${SPEC_TOKENS_LABEL:-no-mtp}

echo "GLM-4.7 benchmark — ${LABEL} — $(date '+%Y-%m-%d %H:%M')"
echo "Base URL: ${BASE_URL}  Model: ${MODEL}"
echo "Input: ${INPUT_LEN} tok  Output: ${OUTPUT_LEN} tok  Prompts: ${NUM_PROMPTS}"
echo ""
printf "%-14s %-16s %-16s %-14s %-10s\n" "Concurrency" "Sys tok/s" "TPOT median ms" "TTFT median ms" "TTFT P99 ms"
echo "--------------------------------------------------------------------------"

for C in 1 2 4 8 16; do
    RESULT=$(HF_HUB_OFFLINE=1 "${VLLM_BIN}" bench serve \
        --base-url "${BASE_URL}" \
        --model "${MODEL}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --dataset-name random \
        --random-input-len "${INPUT_LEN}" \
        --random-output-len "${OUTPUT_LEN}" \
        --num-prompts "${NUM_PROMPTS}" \
        --max-concurrency "${C}" \
        --percentile-metrics ttft,tpot,e2el 2>&1)

    TPOT_MED=$(echo "$RESULT" | grep "Median TPOT" | awk '{print $NF}')
    TTFT_MED=$(echo "$RESULT" | grep "Median TTFT" | awk '{print $NF}')
    TTFT_P99=$(echo "$RESULT" | grep "P99 TTFT"    | awk '{print $NF}')

    # System throughput = concurrency / (TPOT_ms / 1000)
    SYS_TOKS=$(python3 -c "print(f'{$C * 1000 / $TPOT_MED:.0f}')" 2>/dev/null || echo "?")

    printf "%-14s %-16s %-16s %-14s %-10s\n" \
        "${C}" "${SYS_TOKS}" "${TPOT_MED}" "${TTFT_MED}" "${TTFT_P99}"
done
