# GLM-4.7-NVFP4 on SM120 — Patches & Serve Scripts

**→ [Quick Start Guide](docs/quickstart.md)** — just the steps to get running.

Patches and serve scripts to run [`Salyut1/GLM-4.7-NVFP4`](https://huggingface.co/Salyut1/GLM-4.7-NVFP4) and [`QuantTrio/GLM-4.7-AWQ`](https://huggingface.co/QuantTrio/GLM-4.7-AWQ) on **SM120 (RTX PRO 6000 Blackwell, RTX 5090)** hardware.

**Working as of 2026-02-21 with vLLM 0.15.1. Tool calling confirmed working with Claude Code via Anthropic `/v1/messages` endpoint.**

SGLang is blocked by a checkpoint format incompatibility in v0.5.6/v0.5.7 — see `docs/sm120-blackwell-fp4-fixes.md` for details.

---

## Hardware / Software

| | |
|---|---|
| GPU | 4x NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, CC 12.0, 96 GiB each) |
| Driver | 580.105.08 |
| vLLM | 0.15.1 |
| PyTorch | 2.9.1+cu128 |
| flashinfer | 0.6.1 |
| Python | 3.12 |

---

## Quickstart

### 1. Install vLLM

```bash
uv tool install vllm==0.15.1
```

### 2. Apply patches

```bash
~/.local/share/uv/tools/vllm/bin/python apply_patches.py
```

Check status anytime:
```bash
~/.local/share/uv/tools/vllm/bin/python apply_patches.py --check
```

Revert to originals (from `.bak` backups):
```bash
~/.local/share/uv/tools/vllm/bin/python apply_patches.py --revert
```

### 3. Serve

```bash
# Set MODEL_PATH to your local snapshot or HF repo ID
export MODEL_PATH=~/.cache/huggingface/hub/models--Salyut1--GLM-4.7-NVFP4/snapshots/531df318dbe2877b24881f6e15024c7f945e7048

TP=4 bash scripts/serve_glm47_nvfp4_vllm.sh > vllm_run.log 2>&1 &
tail -f vllm_run.log
```

Server is ready when you see:
```
Application startup complete.
```

### 4. Test

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_PATH"'",
    "messages": [{"role": "user", "content": "Hello, what can you do?"}],
    "max_tokens": 200
  }'
```

---

## What the patches fix

### `patches/vllm_glm4_moe.py` — skip missing k_scale/v_scale

`Salyut1/GLM-4.7-NVFP4` doesn't contain FP8 KV-cache scale tensors. vLLM 0.15.1 crashes with `KeyError: 'layers.N.self_attn.qkv_proj.k_scale'` when loading the checkpoint. The patch adds a guard in two places in the weight loader to skip these missing tensors.

---

## What the patches fix (continued)

### `patches/vllm_anthropic_serving.py` — Anthropic endpoint tool call bugs

Four bugs in `vllm/entrypoints/anthropic/serving.py` that cause tool calls to be silently dropped when using the `/v1/messages` Anthropic endpoint:

1. **Non-streaming NoneType** — `message.tool_calls` is `None` with no tools → `TypeError`. Fix: `or []` guard.
2. **Streaming NoneType** — `delta.tool_calls` is `None` for non-tool chunks → `TypeError`. Fix: `None` guard before `len()`.
3. **Single-chunk args not emitted** — vLLM emits id + args in one chunk; only `content_block_start` was sent, not `content_block_delta`. Fix: emit delta immediately when args are present.
4. **Empty `delta.content` bypasses tool calls** — vLLM sends `delta.content=""` alongside `tool_calls`. Empty string passes `is not None`, enters text branch, `elif tool_calls` never runs. Fix: check tool_calls **before** content.

### `patches/vllm_glm47_tool_parser.py` — GLM-4.7 no-newline tool call format

GLM-4.7 emits `<tool_call>Bash<arg_key>...` without a newline between function name and args. The parent class parser requires `\n`. This patch installs a subclass that handles both formats.

Required serve flags: `--tool-call-parser glm47 --enable-auto-tool-choice`

---

## Key serve flags for SM120

| Flag | Value | Reason |
|------|-------|--------|
| `--attention-backend` | `TRITON_ATTN` | flashinfer attention crashes on SM120 |
| `--gpu-memory-utilization` | `0.80` | default 0.90 OOMs at sampler warmup |
| `--quantization` | `modelopt_fp4` | NVFP4 checkpoint format |
| `--tensor-parallel-size` | `4` | ~94 GiB model needs all 4 GPUs |

The MoE backend chosen automatically at runtime: **`VLLM_CUTLASS`** — vLLM's own CUTLASS FP4 kernel, which compiles for SM120 at startup. This avoids flashinfer's `FLASHINFER_TRTLLM` (SM100-only cubins) and `FLASHINFER_CUTLASS` (returns zeros on SM120, flashinfer [#2577](https://github.com/flashinfer-ai/flashinfer/issues/2577)) paths.

---

## Re-applying patches after reinstall

The patch to `glm4_moe.py` lives inside the vLLM install and is lost on reinstall. After any `uv tool install --reinstall vllm` or version upgrade, run:

```bash
~/.local/share/uv/tools/vllm/bin/python apply_patches.py
```

---

## Docs

- [`docs/vllm-sm120-nvfp4-working-state.md`](docs/vllm-sm120-nvfp4-working-state.md) — full vLLM working state, errors encountered, startup sequence
- [`docs/sm120-blackwell-fp4-fixes.md`](docs/sm120-blackwell-fp4-fixes.md) — SGLang investigation report, flashinfer regression history, upstream issue tracking

---

## Status Notes (2026-02-21)

The primary path to running GLM-4.7-NVFP4 on SM120 today is via **vLLM 0.15.1** with the included patches — this is confirmed working with tool calling via the Anthropic API.

**SGLang** has an upstream fix in commit `33c33a7de` ([#18546](https://github.com/sgl-project/sglang/pull/18546)) that addresses KV cache scale loading that previously required patching. However, the fundamental SM120 FP4 MoE backend issue remains unresolved in SGLang (no working backend: triton gives garbage, cutlass returns zeros, trtllm uses SM100-only cubins). See `docs/sm120-blackwell-fp4-fixes.md` for complete details.
