#!/usr/bin/env bash
set -euo pipefail

# Stop the Qwen3.5 Docker container.

CONTAINER_NAME=${CONTAINER_NAME:-qwen35-nvfp4}

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Stopping ${CONTAINER_NAME}..."
  docker rm -f "${CONTAINER_NAME}"
  echo "Stopped."
else
  echo "Container '${CONTAINER_NAME}' not found."
fi
