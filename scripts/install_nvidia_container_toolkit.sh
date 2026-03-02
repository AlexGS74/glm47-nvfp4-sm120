#!/usr/bin/env bash
set -euo pipefail

# Install NVIDIA Container Toolkit for Docker GPU support.
# Run with: sudo bash scripts/install_nvidia_container_toolkit.sh

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo bash $0" >&2
  exit 1
fi

echo "Adding NVIDIA Container Toolkit repo..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "Installing..."
apt-get update -qq
apt-get install -y nvidia-container-toolkit

echo "Configuring Docker runtime..."
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "Done. Test with: docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi"
