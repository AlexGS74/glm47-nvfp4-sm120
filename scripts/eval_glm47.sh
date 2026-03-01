#!/usr/bin/env bash
# Evaluate GLM-4.7 quality using lm-evaluation-harness.
# Routes through buster-ripper (eval-mode) which:
#   - strips max_gen_toks (lm-eval internal field; breaks response truncation if sent)
#   - strips empty Bearer auth headers
#   - injects chat_template_kwargs.enable_thinking=true/false (EVAL_THINKING)
#   - strips <think>...</think> blocks from content (eval path: clean OAI format)
#   - strips code fences from content (build_predictions_instruct compat)
#   - copies reasoning_content → content when content is empty (thinking mode)
#
# Usage:
#   LABEL=awq      ./scripts/eval_glm47.sh
#   LABEL=nvfp4    ./scripts/eval_glm47.sh
#   LABEL=nvfp4 TASKS=gsm8k_cot_zeroshot ./scripts/eval_glm47.sh
#
# Results saved to: ./evals/<LABEL>/
# Note: lm-eval only writes results_*.json at the END of all tasks.
# Use ONE_AT_A_TIME=1 to run each task separately so results save incrementally.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Server ────────────────────────────────────────────────────────────────────
SERVER_BASE=${SERVER_BASE:-http://localhost:30000}
PROXY_PORT=${PROXY_PORT:-30002}
PROXY_URL="http://127.0.0.1:${PROXY_PORT}"
BASE_URL="${PROXY_URL}/v1/chat/completions"
MODEL=${MODEL:-claude-opus-4-5-20251001}
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-4096}
EVAL_THINKING=${EVAL_THINKING:-0}   # 0 = disable (faster, ~44%); 1 = enable thinking (no benefit seen)
REPETITION_PENALTY=${REPETITION_PENALTY:-1.05}

# ── Tokenizer (for lm-eval token counting) ────────────────────────────────────
NVFP4_TOKENIZER=$(ls -d "${HOME}/.cache/huggingface/hub/models--Salyut1--GLM-4.7-NVFP4/snapshots/"*/ 2>/dev/null | head -1 || true)
AWQ_TOKENIZER=$(ls -d "${HOME}/.cache/huggingface/models--QuantTrio--GLM-4.7-AWQ/snapshots/"*/ 2>/dev/null | head -1 || true)
FP8_TOKENIZER=$(ls -d "${HOME}/.cache/huggingface/hub/models--zai-org--GLM-4.7-FP8/snapshots/"*/ 2>/dev/null | head -1 || true)

if [[ -n "${TOKENIZER_PATH:-}" ]]; then
  : # already set by caller
elif [[ "${LABEL:-}" == "fp8"* ]] && [[ -n "${FP8_TOKENIZER:-}" ]]; then
  TOKENIZER_PATH="${FP8_TOKENIZER}"
elif [[ "${LABEL:-}" == "awq"* ]] && [[ -n "${AWQ_TOKENIZER:-}" ]]; then
  TOKENIZER_PATH="${AWQ_TOKENIZER}"
elif [[ "${LABEL:-}" == "nvfp4"* ]] && [[ -n "${NVFP4_TOKENIZER:-}" ]]; then
  TOKENIZER_PATH="${NVFP4_TOKENIZER}"
else
  TOKENIZER_PATH="${FP8_TOKENIZER:-${NVFP4_TOKENIZER:-${AWQ_TOKENIZER:-}}}"
fi

if [[ -z "${TOKENIZER_PATH:-}" ]]; then
  echo "ERROR: no local tokenizer found; set TOKENIZER_PATH=/path/to/model" >&2
  exit 1
fi

# ── Eval config ───────────────────────────────────────────────────────────────
# humaneval_instruct:          chat-formatted HumanEval (164 problems, pass@1)
# mbpp_instruct:               chat-formatted MBPP (500 problems, pass@1)
# gsm8k_cot_zeroshot:          zero-shot CoT math (1319 problems)
# minerva_math500:             competition math, LaTeX answer matching (500 problems)
# ifeval:                      verifiable instruction following (541 problems)
# gpqa_diamond_cot_zeroshot:   PhD-level science reasoning (198 problems) — gated, needs HF login + access request
# All tasks: thinking OFF (enable_thinking=false injected by glm47 profile)
TASKS=${TASKS:-humaneval_instruct,mbpp_instruct,gsm8k_cot_zeroshot,minerva_math500,gpqa_diamond_cot_zeroshot,ifeval}

# Number of samples per task — 0 = full eval
NUM_SAMPLES=${NUM_SAMPLES:-0}

# Run each task separately so results save after each one completes (safer for long runs)
ONE_AT_A_TIME=${ONE_AT_A_TIME:-1}

LABEL=${LABEL:-no-label}
OUTPUT_DIR=${OUTPUT_DIR:-./evals/${LABEL}}
mkdir -p "${OUTPUT_DIR}"

# Parallel requests to vLLM — 16 concurrent saturates the server nicely
NUM_CONCURRENT=${NUM_CONCURRENT:-16}

# ── Sanity checks ─────────────────────────────────────────────────────────────
if ! command -v uvx &>/dev/null; then
  echo "uvx not found — install uv: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if ! curl -sf "${SERVER_BASE}/v1/models" >/dev/null 2>&1; then
  echo "ERROR: vLLM server not reachable at ${SERVER_BASE}" >&2
  echo "Start the server first (serve_glm47_awq.sh or serve_glm47_nvfp4_vllm.sh)" >&2
  exit 1
fi

# ── Start buster-ripper (strips max_gen_toks + empty auth, injects enable_thinking=false) ──
fuser -k "${PROXY_PORT}/tcp" 2>/dev/null || true
sleep 0.3

PROXY_PID=""
cleanup() { [[ -n "${PROXY_PID}" ]] && kill "${PROXY_PID}" 2>/dev/null || true; }
trap cleanup EXIT

THINKING_FLAG=""
[[ "${EVAL_THINKING}" == "1" ]] && THINKING_FLAG="--eval-thinking"

buster-ripper \
  --upstream "${SERVER_BASE}" \
  --port "${PROXY_PORT}" \
  --host 127.0.0.1 \
  --eval-mode \
  --eval-profile glm47 \
  --eval-max-tokens "${EVAL_MAX_TOKENS}" \
  ${THINKING_FLAG} \
  >/tmp/buster-ripper-eval.log 2>&1 &
PROXY_PID=$!

for i in $(seq 1 15); do
  sleep 0.5
  if curl -sf "${PROXY_URL}/v1/models" >/dev/null 2>&1; then break; fi
done
echo "buster-ripper listening on :${PROXY_PORT}"

# ── Run ───────────────────────────────────────────────────────────────────────
LIMIT_FLAG=""
if [[ "${NUM_SAMPLES}" -gt 0 ]]; then
  LIMIT_FLAG="--limit ${NUM_SAMPLES}"
fi

# humaneval/mbpp execute model-generated code — required opt-in
export HF_ALLOW_CODE_EVAL=1
# All datasets cached locally — no network needed
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "GLM-4.7 quality eval — ${LABEL} — $(date '+%Y-%m-%d %H:%M')"
echo "Server:    ${SERVER_BASE} (via proxy :${PROXY_PORT})  Model: ${MODEL}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Tasks:     ${TASKS}"
echo "Samples per task: ${NUM_SAMPLES:-full}  Concurrent: ${NUM_CONCURRENT}  Thinking: $([[ "${EVAL_THINKING}" == "1" ]] && echo on || echo off)  RepPenalty: ${REPETITION_PENALTY}"
echo "Output:    ${OUTPUT_DIR}"
echo ""

LM_EVAL_COMMON=(
  uvx lm_eval run
  --model local-chat-completions
  --model_args "model=${MODEL},base_url=${BASE_URL},tokenizer=${TOKENIZER_PATH},tokenizer_backend=huggingface,max_retries=3,timeout=300,num_concurrent=${NUM_CONCURRENT}"
  --apply_chat_template
  --confirm_run_unsafe_code
  --gen_kwargs "max_tokens=${EVAL_MAX_TOKENS},max_gen_toks=${EVAL_MAX_TOKENS},repetition_penalty=${REPETITION_PENALTY}"
  --log_samples
  --verbosity WARNING
)
[[ -n "${LIMIT_FLAG}" ]] && LM_EVAL_COMMON+=(${LIMIT_FLAG})

if [[ "${ONE_AT_A_TIME}" == "1" ]]; then
  IFS=',' read -ra TASK_LIST <<< "${TASKS}"
  for TASK in "${TASK_LIST[@]}"; do
    TASK_OUT="${OUTPUT_DIR}/${TASK}"
    mkdir -p "${TASK_OUT}"
    echo "--- Task: ${TASK} — $(date '+%H:%M') ---"
    "${LM_EVAL_COMMON[@]}" --tasks "${TASK}" --output_path "${TASK_OUT}"
    echo "Results saved to: ${TASK_OUT}"
    echo ""
  done
else
  "${LM_EVAL_COMMON[@]}" --tasks ${TASKS//,/ } --output_path "${OUTPUT_DIR}"
  echo ""
  echo "Results saved to: ${OUTPUT_DIR}"
fi
