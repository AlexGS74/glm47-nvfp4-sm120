#!/usr/bin/env bash
set -euo pipefail

# Normalizing proxy for vLLM Anthropic endpoint.
# Sorts tool definitions by name before forwarding so the tool block is
# byte-identical across MCP reconnects, stabilizing the vLLM prefix cache.
#
# Usage:
#   bash proxy/serve_proxy.sh
#   STRIP_DATE=1 bash proxy/serve_proxy.sh   # also strip daily date injection
#
# Then point Claude Code at port 30001:
#   ANTHROPIC_BASE_URL=http://localhost:30001 claude ...
# or update the glm47 bash function in ~/.bashrc

UPSTREAM=${UPSTREAM:-http://localhost:30000}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30001}
STRIP_DATE=${STRIP_DATE:-0}
VERBOSE=${VERBOSE:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXTRA_FLAGS=""
[[ "${STRIP_DATE}" == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --strip-date"
[[ "${VERBOSE}"     == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --verbose"

echo "proxy → ${UPSTREAM}  listening on ${HOST}:${PORT}"

exec uv run "${SCRIPT_DIR}/proxy.py" \
  --upstream "${UPSTREAM}" \
  --host     "${HOST}" \
  --port     "${PORT}" \
  ${EXTRA_FLAGS}
