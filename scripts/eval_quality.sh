#!/usr/bin/env bash
# Evaluate GLM-4.7 quality using lm-evaluation-harness.
# Hits a running vLLM server at BASE_URL (default: localhost:30000).
# Run once with AWQ, once with NVFP4 to compare quality.
#
# Usage:
#   LABEL=awq      ./scripts/eval_quality.sh
#   LABEL=nvfp4    ./scripts/eval_quality.sh
#   LABEL=nvfp4 TASKS=gsm8k_cot_zeroshot ./scripts/eval_quality.sh
#
# Results saved to: ./evals/<LABEL>/

set -euo pipefail

# ── Server ────────────────────────────────────────────────────────────────────
BASE_URL=${BASE_URL:-http://localhost:30000/v1}
MODEL=${MODEL:-claude-opus-4-5-20251001}

# ── Tokenizer (for lm-eval token counting) ────────────────────────────────────
# Matched to the model being evaluated so token budgets are accurate.
# Override with TOKENIZER_PATH=/path/to/model
NVFP4_TOKENIZER=$(ls -d "${HOME}/.cache/huggingface/hub/models--Salyut1--GLM-4.7-NVFP4/snapshots/"*/ 2>/dev/null | head -1)
AWQ_TOKENIZER=$(ls -d "${HOME}/.cache/huggingface/models--QuantTrio--GLM-4.7-AWQ/snapshots/"*/ 2>/dev/null | head -1)

if [[ -n "${TOKENIZER_PATH:-}" ]]; then
  : # already set by caller
elif [[ "${LABEL}" == "awq"* ]] && [[ -n "${AWQ_TOKENIZER}" ]]; then
  TOKENIZER_PATH="${AWQ_TOKENIZER}"
elif [[ "${LABEL}" == "nvfp4"* ]] && [[ -n "${NVFP4_TOKENIZER}" ]]; then
  TOKENIZER_PATH="${NVFP4_TOKENIZER}"
else
  # Fallback: prefer NVFP4 then AWQ
  TOKENIZER_PATH="${NVFP4_TOKENIZER:-${AWQ_TOKENIZER}}"
fi

if [[ -z "${TOKENIZER_PATH:-}" ]]; then
  echo "ERROR: no local tokenizer found; set TOKENIZER_PATH=/path/to/model" >&2
  exit 1
fi

# ── Eval config ───────────────────────────────────────────────────────────────
# humaneval_instruct: chat-formatted HumanEval (164 problems, pass@1)
# mbpp_instruct:      chat-formatted MBPP (500 problems, pass@1)
# gsm8k_cot_zeroshot: zero-shot chain-of-thought math (1319 problems)
TASKS=${TASKS:-humaneval_instruct,mbpp_instruct,gsm8k_cot_zeroshot}

# Limit samples per task for speed — set to 0 for full eval
NUM_SAMPLES=${NUM_SAMPLES:-100}

# Label identifies the run (awq / nvfp4 / etc.)
LABEL=${LABEL:-no-label}

OUTPUT_DIR=${OUTPUT_DIR:-./evals/${LABEL}}
mkdir -p "${OUTPUT_DIR}"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if ! command -v uvx &>/dev/null; then
  echo "uvx not found — install uv: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

# Check server is up
if ! curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: vLLM server not reachable at ${BASE_URL}" >&2
  echo "Start the server first (serve_glm47_awq.sh or serve_glm47_nvfp4_vllm.sh)" >&2
  exit 1
fi

# ── Run ───────────────────────────────────────────────────────────────────────
LIMIT_FLAG=""
if [[ "${NUM_SAMPLES}" -gt 0 ]]; then
  LIMIT_FLAG="--limit ${NUM_SAMPLES}"
fi

echo "GLM-4.7 quality eval — ${LABEL} — $(date '+%Y-%m-%d %H:%M')"
echo "Server:    ${BASE_URL}  Model: ${MODEL}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Tasks:     ${TASKS}"
echo "Samples per task: ${NUM_SAMPLES:-full}"
echo "Output:    ${OUTPUT_DIR}"
echo ""

uvx lm_eval run \
  --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${BASE_URL},tokenizer=${TOKENIZER_PATH},tokenizer_backend=huggingface,max_retries=3,timeout=120" \
  --tasks "${TASKS}" \
  --apply_chat_template \
  --output_path "${OUTPUT_DIR}" \
  --log_samples \
  ${LIMIT_FLAG} \
  --verbosity WARNING

echo ""
echo "Results saved to: ${OUTPUT_DIR}"
