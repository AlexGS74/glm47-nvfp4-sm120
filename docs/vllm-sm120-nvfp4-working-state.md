# vLLM SM120 NVFP4 — Working State Report

**Date:** 2026-02-20
**Hardware:** 4x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (SM120, CC 12.0, 96 GiB each)
**Driver:** 580.105.08
**Model:** `Salyut1/GLM-4.7-NVFP4` (177B ModelOpt/NVFP4, local HF cache snapshot)

---

## Working Dependency Versions

| Package | Version |
|---------|---------|
| vllm | 0.15.1 |
| torch | 2.9.1+cu128 |
| transformers | 4.57.6 |
| flashinfer | 0.6.1 |
| Python | 3.12 |
| CUDA | 12.8 (cu128) |
| Driver | 580.105.08 |
| GPU | RTX PRO 6000 Blackwell Max-Q (SM120 / CC 12.0) |

Install:
```bash
uv tool install vllm==0.15.1
```

---

## Serve Script

`/home/alex/mllm/serve_glm47_nvfp4_vllm.sh`

Run with:
```bash
TP=4 bash /home/alex/mllm/serve_glm47_nvfp4_vllm.sh
# output goes to stdout/stderr; redirect as needed:
TP=4 bash /home/alex/mllm/serve_glm47_nvfp4_vllm.sh > /home/alex/mllm/vllm_run.log 2>&1 &
```

Key variables and defaults:
```bash
VLLM_PYTHON=${VLLM_PYTHON:-${HOME}/.local/share/uv/tools/vllm/bin/python}
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/huggingface/hub/models--Salyut1--GLM-4.7-NVFP4/snapshots/531df318dbe2877b24881f6e15024c7f945e7048}
HF_HUB_OFFLINE=1                 # use local cache, no re-download
TP=4
DTYPE=bfloat16
QUANTIZATION=modelopt_fp4
ATTENTION_BACKEND=TRITON_ATTN    # flashinfer rejects SM120
GPU_MEM_UTIL=0.80                # 0.90 default OOMs at sampler warmup
```

---

## Flags Required for SM120 and Why

| Flag | Value | Why needed |
|------|-------|-----------|
| `--attention-backend` | `TRITON_ATTN` | flashinfer attention backend crashes at `prefill_wrapper.plan()` on SM120. Note: vLLM names are uppercase — `TRITON_ATTN` not `triton`. |
| `--gpu-memory-utilization` | `0.80` | At default 0.90, only ~1.69 GiB remains after model + KV cache + CUDA graphs. Sampler warmup with 1024 dummy requests needs ~1.74 GiB → OOM. 0.80 leaves ~19 GiB free per GPU. |
| `--quantization` | `modelopt_fp4` | NVFP4 checkpoint format. |
| `--dtype` | `bfloat16` | Activation dtype. |
| `--tensor-parallel-size` | `4` | ~94 GiB model; requires all 4 GPUs. |
| `--trust-remote-code` | — | Required by GLM-4.7 custom model code. |

---

## Backends Selected at Runtime (confirmed in logs)

```
Using 'VLLM_CUTLASS' NvFp4 MoE backend out of potential backends:
    ['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL', 'FLASHINFER_CUTLASS', 'VLLM_CUTLASS', 'MARLIN']
Using AttentionBackendEnum.TRITON_ATTN backend.
```

`VLLM_CUTLASS` is vLLM's own CUTLASS-based FP4 MoE kernel, JIT-compiled for SM120 at startup. This bypasses the flashinfer TRTLLM (SM100-only cubins) and CUTLASS (zeros bug on SM120, flashinfer #2577) paths entirely.

---

## Patches Applied to vLLM

### Patch 1 — Skip missing k_scale / v_scale in weight loader

**File:** `~/.local/share/uv/tools/vllm/lib/python3.12/site-packages/vllm/model_executor/models/glm4_moe.py`

**Symptom:**
```
KeyError: 'layers.41.self_attn.qkv_proj.k_scale'
```

**Cause:** `Salyut1/GLM-4.7-NVFP4` does not contain FP8 KV-cache scale tensors (`k_scale`, `v_scale`). vLLM 0.15.1 added an `AutoWeightLoader` path that iterates raw checkpoint keys and hits a `KeyError` when those tensors are absent from `params_dict`.

**Patch — Location 1 (~line 528, QKV shard loading path):**
```python
# inserted before:  param = params_dict[name]
if ('k_scale' in name or 'v_scale' in name) and name not in params_dict:
    continue
```

**Patch — Location 2 (~line 590, general weight loading path):**
```python
# inserted before:  param = params_dict[name]
if ('k_scale' in name or 'v_scale' in name) and name not in params_dict:
    continue
```

After the patch, vLLM logs a benign warning:
```
WARNING: Checkpoint does not provide a q scaling factor. Setting it to k_scale.
         This only matters for FP8 Attention backends (flash-attn or flashinfer).
```
Harmless when using `TRITON_ATTN`.

**Survivability:** Lost on `uv tool install --reinstall vllm`. Must re-apply after reinstall.

---

## Errors Encountered (in order)

### 1. flashinfer attention backend fails on SM120

```
File "vllm/v1/attention/backends/flashinfer.py", line 1045, in build
    prefill_wrapper.plan(...)
```

**Fix:** `--attention-backend TRITON_ATTN`

Attempted `--attention-backend triton` first, which raised:
```
ValueError: Unknown attention backend: 'TRITON'. Valid options are: FLASH_ATTN, TRITON_ATTN, ...
```
Backend names in vLLM v0.15.1 are uppercase.

---

### 2. OOM during sampler warmup

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.74 GiB.
GPU 3 has a total capacity of 94.97 GiB of which 1.69 GiB is free.
...
RuntimeError: CUDA out of memory occurred when warming up sampler with 1024 dummy requests.
Please try lowering `max_num_seqs` or `gpu_memory_utilization`.
```

Occurred after CUDA graphs captured successfully. The warmup allocates batches up to `max_num_seqs` (default 1024) simultaneously.

**Fix:** `--gpu-memory-utilization 0.80` (set as default in script).

---

## Startup Sequence (normal healthy run)

```
Python:      ~/.local/share/uv/tools/vllm/bin/python
vLLM:        0.15.1
Model:       ...GLM-4.7-NVFP4/snapshots/531df318...
TP:          4
Attention:   TRITON_ATTN
GPU mem:     0.80

[Worker] Using 'VLLM_CUTLASS' NvFp4 MoE backend
[Worker] Using AttentionBackendEnum.TRITON_ATTN backend.
[Worker] Loading safetensors checkpoint shards: 100% | 41/41
[Worker] Loading weights took 13.96 seconds
[Worker] WARNING: Checkpoint does not provide a q scaling factor ...  ← benign
[Worker] Model loading took 47.11 GiB memory and 14.96 seconds
[Worker] torch.compile: Dynamo bytecode transform: 10.40 s
[Worker] Compiling graph for compile range (1, 8192): 12.23 s
[Worker] Available KV cache memory: 31.99 GiB
[EngineCore] GPU KV cache size: 729,248 tokens
[Worker] Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 51/51
[Worker] Capturing CUDA graphs (decode, FULL): 51/51
[APIServer] Application startup complete.
```

Total cold-start time: ~3–4 minutes (dominated by torch.compile + CUDA graph capture).
