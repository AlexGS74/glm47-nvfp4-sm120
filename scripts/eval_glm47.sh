#!/usr/bin/env bash
# Evaluate GLM-4.7 quality using lm-evaluation-harness.
# Routes through buster-ripper (eval-mode) which:
#   - strips max_gen_toks (lm-eval internal field; breaks response truncation if sent)
#   - strips empty Bearer auth headers
#   - injects chat_template_kwargs.enable_thinking=false
#
# Usage:
#   LABEL=awq      ./scripts/eval_quality.sh
#   LABEL=nvfp4    ./scripts/eval_quality.sh
#   LABEL=nvfp4 TASKS=gsm8k_cot_zeroshot ./scripts/eval_quality.sh
#
# Results saved to: ./evals/<LABEL>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Server ────────────────────────────────────────────────────────────────────
SERVER_BASE=${SERVER_BASE:-http://localhost:30000}
PROXY_PORT=${PROXY_PORT:-30002}
PROXY_URL="http://127.0.0.1:${PROXY_PORT}"
BASE_URL="${PROXY_URL}/v1/chat/completions"
MODEL=${MODEL:-claude-opus-4-5-20251001}
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-4096}
EVAL_THINKING=${EVAL_THINKING:-1}   # 1 = enable chain-of-thought; 0 = disable (faster)
REPETITION_PENALTY=${REPETITION_PENALTY:-1.05}

# ── Tokenizer (for lm-eval token counting) ────────────────────────────────────
NVFP4_TOKENIZER=$(ls -d "${HOME}/.cache/huggingface/hub/models--Salyut1--GLM-4.7-NVFP4/snapshots/"*/ 2>/dev/null | head -1 || true)
AWQ_TOKENIZER=$(ls -d "${HOME}/.cache/huggingface/models--QuantTrio--GLM-4.7-AWQ/snapshots/"*/ 2>/dev/null | head -1 || true)

if [[ -n "${TOKENIZER_PATH:-}" ]]; then
  : # already set by caller
elif [[ "${LABEL:-}" == "awq"* ]] && [[ -n "${AWQ_TOKENIZER:-}" ]]; then
  TOKENIZER_PATH="${AWQ_TOKENIZER}"
elif [[ "${LABEL:-}" == "nvfp4"* ]] && [[ -n "${NVFP4_TOKENIZER:-}" ]]; then
  TOKENIZER_PATH="${NVFP4_TOKENIZER}"
else
  TOKENIZER_PATH="${NVFP4_TOKENIZER:-${AWQ_TOKENIZER:-}}"
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

# Number of samples per task — 0 = full eval
NUM_SAMPLES=${NUM_SAMPLES:-0}

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

uvx lm_eval run \
  --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${BASE_URL},tokenizer=${TOKENIZER_PATH},tokenizer_backend=huggingface,max_retries=3,timeout=300,num_concurrent=${NUM_CONCURRENT}" \
  --tasks "${TASKS}" \
  --apply_chat_template \
  --confirm_run_unsafe_code \
  --gen_kwargs "max_tokens=${EVAL_MAX_TOKENS},max_gen_toks=${EVAL_MAX_TOKENS},repetition_penalty=${REPETITION_PENALTY}" \
  --output_path "${OUTPUT_DIR}" \
  --log_samples \
  ${LIMIT_FLAG} \
  --verbosity WARNING

echo ""
echo "Results saved to: ${OUTPUT_DIR}"
