#!/usr/bin/env bash
set -euo pipefail

# Stop all Docker containers, then hibernate.
# Containers are NOT restarted on resume — start your serving script manually.

RUNNING=$(docker ps -q 2>/dev/null)
if [[ -n "${RUNNING}" ]]; then
  echo "Stopping Docker containers..."
  docker stop ${RUNNING}
  echo "Stopped."
fi

echo "Hibernating..."
sudo systemctl hibernate
