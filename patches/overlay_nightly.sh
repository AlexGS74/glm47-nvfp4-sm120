#!/usr/bin/env bash
set -euo pipefail

# Overlay patched files from the vllm-patches repo onto the installed vLLM nightly.
#
# Run this once after upgrading vLLM:
#   uv tool install --force --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
#   bash patches/overlay_nightly.sh
#
# The source files live in the sm120-nvfp4 branch of:
#   https://github.com/AlexGS74/vllm-patches
# Clone / update with:
#   git -C ~/mllm/vllm-patches pull   (or clone --depth=1 if not present)

VLLM_PYTHON=${VLLM_PYTHON:-${HOME}/.local/share/uv/tools/vllm/bin/python}
PATCHES_REPO=${PATCHES_REPO:-${HOME}/mllm/vllm-patches/vllm}

if [[ ! -x "${VLLM_PYTHON}" ]]; then
  echo "ERROR: vllm python not found: ${VLLM_PYTHON}" >&2
  exit 1
fi

if [[ ! -d "${PATCHES_REPO}" ]]; then
  echo "ERROR: patches repo not found: ${PATCHES_REPO}" >&2
  echo "  Clone with: git clone --depth=1 -b sm120-nvfp4 https://github.com/AlexGS74/vllm-patches ~/mllm/vllm-patches" >&2
  exit 1
fi

VLLM_SITE=$("${VLLM_PYTHON}" -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
VLLM_VER=$("${VLLM_PYTHON}" -c "import vllm; print(vllm.__version__)")

echo "vllm version: ${VLLM_VER}"
echo "vllm site:    ${VLLM_SITE}"
echo "patches repo: ${PATCHES_REPO}"
echo ""

FILES=(
  "entrypoints/anthropic/protocol.py"
  "entrypoints/anthropic/serving.py"
  "tool_parsers/glm47_moe_tool_parser.py"
  "model_executor/models/glm4_moe.py"
  # Qwen3.5 tool call fixes (PR #35615 + #35347)
  "tool_parsers/__init__.py"
  "tool_parsers/qwen3coder_tool_parser.py"
  "tool_parsers/qwen35coder_tool_parser.py"
  "entrypoints/openai/chat_completion/serving.py"
  # PR #35219 (KV cache block zeroing) — incompatible with our nightly (dev106 vs dev92)
  # Use Docker image orthozany/vllm-qwen35-mtp for this fix instead
)

for rel in "${FILES[@]}"; do
  src="${PATCHES_REPO}/${rel}"
  dst="${VLLM_SITE}/${rel}"
  if [[ ! -f "${src}" ]]; then
    echo "  MISSING source: ${src}" >&2
    exit 1
  fi
  cp -v "${src}" "${dst}"
done

echo ""
echo "All patches applied successfully."
