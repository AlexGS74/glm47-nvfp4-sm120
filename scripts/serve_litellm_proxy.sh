#!/usr/bin/env bash
set -euo pipefail

# LiteLLM proxy: translates Anthropic /v1/messages → OpenAI /v1/chat/completions → vLLM
# Listens on PORT (default 30000), forwards to vLLM on 30001.
# Install: uv tool install 'litellm[proxy]'

LITELLM=${LITELLM:-${HOME}/.local/share/uv/tools/litellm/bin/litellm}
CONFIG=${CONFIG:-$(dirname "$0")/litellm_config.yaml}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}

if [[ ! -x "${LITELLM}" ]]; then
  echo "litellm not found: ${LITELLM}" >&2
  echo "Install with: uv tool install 'litellm[proxy]'" >&2
  exit 1
fi

echo "LiteLLM proxy: ${HOST}:${PORT} → vLLM on :30001"
echo ""

exec "${LITELLM}" --config "${CONFIG}" --host "${HOST}" --port "${PORT}"
