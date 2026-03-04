#!/usr/bin/env bash
set -euo pipefail

# Graft MTP weights from zai-org/GLM-4.7 (BF16) onto Salyut1/GLM-4.7-NVFP4.
#
# The NVFP4 checkpoint ships without MTP weights (layer 92), so MTP speculative
# decoding is impossible. This script downloads the BF16 mtp.safetensors (~10.3 GB)
# and merges its weight keys into the NVFP4 model.safetensors.index.json.
#
# After grafting, start vLLM with:
#   SPEC_TOKENS=1 ./serve_glm47_nvfp4_vllm.sh
#
# To revert: run with --revert

BF16_REPO="zai-org/GLM-4.7"
MTP_FILE="mtp.safetensors"

NVFP4_SNAPSHOT="${HOME}/.cache/huggingface/hub/models--Salyut1--GLM-4.7-NVFP4/snapshots/531df318dbe2877b24881f6e15024c7f945e7048"
INDEX_FILE="${NVFP4_SNAPSHOT}/model.safetensors.index.json"
INDEX_BACKUP="${INDEX_FILE}.pre-mtp-graft"
CONFIG_FILE="${NVFP4_SNAPSHOT}/config.json"
CONFIG_BACKUP="${CONFIG_FILE}.pre-mtp-graft"

# ── Revert mode ─────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--revert" ]]; then
  if [[ -f "${INDEX_BACKUP}" ]]; then
    cp "${INDEX_BACKUP}" "${INDEX_FILE}"
    rm -f "${NVFP4_SNAPSHOT}/${MTP_FILE}"
    echo "Reverted: restored original index, removed ${MTP_FILE}"
  else
    echo "ERROR: no backup found at ${INDEX_BACKUP}" >&2
    exit 1
  fi
  if [[ -f "${CONFIG_BACKUP}" ]]; then
    cp "${CONFIG_BACKUP}" "${CONFIG_FILE}"
    echo "Reverted: restored original config.json"
  fi
  exit 0
fi

# ── Preflight checks ───────────────────────────────────────────────────────
if [[ ! -f "${INDEX_FILE}" ]]; then
  echo "ERROR: NVFP4 index not found: ${INDEX_FILE}" >&2
  echo "Download model first: huggingface-cli download Salyut1/GLM-4.7-NVFP4" >&2
  exit 1
fi

# Check if already grafted
if python3 -c "
import json
with open('${INDEX_FILE}') as f:
    wm = json.load(f)['weight_map']
if any('layers.92' in k for k in wm):
    exit(0)
exit(1)
" 2>/dev/null; then
  echo "MTP weights already present in index. Nothing to do."
  echo "To re-graft, run: $0 --revert && $0"
  exit 0
fi

# ── Step 1: Download mtp.safetensors from BF16 repo ────────────────────────
MTP_DEST="${NVFP4_SNAPSHOT}/${MTP_FILE}"

if [[ -f "${MTP_DEST}" ]]; then
  echo "mtp.safetensors already downloaded ($(du -h "${MTP_DEST}" | cut -f1))"
else
  echo "Downloading ${MTP_FILE} from ${BF16_REPO} (~10.3 GB)..."
  huggingface-cli download "${BF16_REPO}" "${MTP_FILE}" \
    --local-dir "${NVFP4_SNAPSHOT}" \
    --local-dir-use-symlinks False
  echo "Downloaded: $(du -h "${MTP_DEST}" | cut -f1)"
fi

# Verify file exists and is reasonable size (>10GB)
FILE_SIZE=$(stat --format=%s "${MTP_DEST}" 2>/dev/null || stat -f%z "${MTP_DEST}" 2>/dev/null)
if [[ "${FILE_SIZE}" -lt 10000000000 ]]; then
  echo "ERROR: ${MTP_FILE} is too small (${FILE_SIZE} bytes). Expected ~10.3 GB." >&2
  echo "Delete and re-download: rm ${MTP_DEST}" >&2
  exit 1
fi

# ── Step 2: Fetch BF16 index to get MTP key names ──────────────────────────
echo "Fetching MTP weight keys from ${BF16_REPO}..."

BF16_INDEX_URL="https://huggingface.co/${BF16_REPO}/raw/main/model.safetensors.index.json"

# ── Step 3: Merge MTP keys into NVFP4 index ─────────────────────────────────
echo "Merging MTP keys into NVFP4 index..."

# Resolve symlink for backup (backup the actual content, not the symlink)
REAL_INDEX=$(readlink -f "${INDEX_FILE}")
cp "${REAL_INDEX}" "${INDEX_BACKUP}"
echo "Backup saved: ${INDEX_BACKUP}"

python3 -c "
import json, urllib.request, sys

# Fetch BF16 index to get MTP keys
print('  Fetching BF16 weight map...')
with urllib.request.urlopen('${BF16_INDEX_URL}') as resp:
    bf16_data = json.loads(resp.read())

bf16_wm = bf16_data['weight_map']
mtp_keys = {k: '${MTP_FILE}' for k, v in bf16_wm.items() if v == '${MTP_FILE}'}
print(f'  Found {len(mtp_keys)} MTP keys in BF16 checkpoint')

if not mtp_keys:
    print('ERROR: no MTP keys found in BF16 checkpoint', file=sys.stderr)
    sys.exit(1)

# Load NVFP4 index
with open('${INDEX_FILE}') as f:
    nvfp4_data = json.load(f)

existing_count = len(nvfp4_data['weight_map'])
nvfp4_data['weight_map'].update(mtp_keys)
new_count = len(nvfp4_data['weight_map'])

print(f'  NVFP4 keys: {existing_count} -> {new_count} (+{new_count - existing_count})')

# Write updated index (NOT the symlink target — write to the actual path)
# First resolve the symlink so we write a real file
import os
real_path = os.path.realpath('${INDEX_FILE}')

# Write to a new file in the snapshot dir, replacing the symlink
output_path = '${INDEX_FILE}'
if os.path.islink(output_path):
    os.unlink(output_path)

with open(output_path, 'w') as f:
    json.dump(nvfp4_data, f, indent=2)

print(f'  Written: {output_path}')
"

# ── Step 4: Patch config.json — exclude layer 92 from NVFP4 quantization ──
echo "Patching config.json quantization ignore list..."

python3 -c "
import json, os, shutil

cfg_path = '${CONFIG_FILE}'
backup_path = '${CONFIG_BACKUP}'

# Resolve symlink for backup
real_path = os.path.realpath(cfg_path)
if not os.path.exists(backup_path):
    shutil.copy2(real_path, backup_path)
    print(f'  Config backup saved: {backup_path}')

with open(cfg_path) as f:
    cfg = json.load(f)

ignore = cfg.get('quantization_config', {}).get('ignore', [])
mtp_pattern = 'model.layers.92'

if mtp_pattern in ignore:
    print(f'  Already in ignore list: {mtp_pattern}')
else:
    cfg['quantization_config']['ignore'].append(mtp_pattern)
    print(f'  Added to ignore list: {mtp_pattern}')
    print(f'  Ignore list now: {cfg[\"quantization_config\"][\"ignore\"]}')

    # Write (replace symlink if needed)
    if os.path.islink(cfg_path):
        os.unlink(cfg_path)
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'  Written: {cfg_path}')
"

# ── Step 5: Verify ──────────────────────────────────────────────────────────
echo ""
echo "Verifying graft..."
python3 -c "
import json
with open('${INDEX_FILE}') as f:
    wm = json.load(f)['weight_map']
mtp = [k for k in wm if 'layers.92' in k]
print(f'  Total keys: {len(wm)}')
print(f'  MTP keys (layer 92): {len(mtp)}')
print(f'  MTP file mapped: {wm.get(\"model.layers.92.shared_head.head.weight\", \"MISSING\")}')
"

echo ""
echo "Done! MTP weights grafted successfully."
echo ""
echo "To use MTP with NVFP4, start vLLM with:"
echo "  SPEC_TOKENS=1 ./serve_glm47_nvfp4_vllm.sh"
echo ""
echo "To revert:"
echo "  $0 --revert"
