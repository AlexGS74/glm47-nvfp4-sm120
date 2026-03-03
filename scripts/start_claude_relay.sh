#!/usr/bin/env bash
set -euo pipefail

# Start claude-relay proxy for Claude Code → vLLM/SGLang.
# Translates Anthropic API format to OpenAI-compatible backend.
#
# Usage:
#   ./scripts/start_claude_relay.sh
#   BACKEND_PORT=8000 ./scripts/start_claude_relay.sh

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30001}
BACKEND_PORT=${BACKEND_PORT:-30000}
BACKEND_URL=${BACKEND_URL:-http://localhost:${BACKEND_PORT}}
LOG_LEVEL=${LOG_LEVEL:-info}

echo "claude-relay"
echo "  Listen:  ${HOST}:${PORT}"
echo "  Backend: ${BACKEND_URL}"
echo ""

exec claude-relay \
  --host "${HOST}" \
  --port "${PORT}" \
  --backend "${BACKEND_URL}" \
  --log-level "${LOG_LEVEL}" \
  "$@"
