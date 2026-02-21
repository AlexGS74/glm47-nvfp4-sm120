# vLLM SM120 NVFP4 — Working State Report

**Date:** 2026-02-21
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

### Patch 2 — GLM-4.7 tool call parser (no-newline format)

**File:** `~/.local/share/uv/tools/vllm/lib/python3.12/site-packages/vllm/tool_parsers/glm47_moe_tool_parser.py`

**Symptom:** Tool calls silently not returned; `Failed to parse tool call` warnings in logs.

**Cause:** GLM-4.7 emits tool calls without a newline between the function name and the first arg tag:
```
GLM-4.5: <tool_call>func_name\n<arg_key>...</arg_key>...</tool_call>
GLM-4.7: <tool_call>func_name<arg_key>...</arg_key>...</tool_call>
```
The parent class (`Glm4MoeModelToolParser`) regex is `r"<tool_call>([^\n]*)\n(.*)</tool_call>"` — requires a newline that GLM-4.7 doesn't emit.

**Fix:** `Glm47MoeModelToolParser` subclass overrides both regexes:
```python
self.func_detail_regex = re.compile(
    r"<tool_call>([^\s<]+)\s*(.*?)</tool_call>", re.DOTALL
)
self.func_arg_regex = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
)
```
Works for both GLM-4.5 (with newline) and GLM-4.7 (no newline) formats.

**Required serve flags:**
```bash
--tool-call-parser glm47 --enable-auto-tool-choice
```

---

### Patch 3 — Anthropic endpoint tool call bugs (4 fixes)

**File:** `~/.local/share/uv/tools/vllm/lib/python3.12/site-packages/vllm/entrypoints/anthropic/serving.py`

**Symptom:** Tool calls not returned to Anthropic SDK clients (e.g. Claude Code via `/v1/messages`). No errors in logs.

**Bug 1 — `messages_full_converter` (non-streaming):**
```python
# Before:
for tool_call in generator.choices[0].message.tool_calls:
# After:
for tool_call in (generator.choices[0].message.tool_calls or []):
```
`message.tool_calls` is `None` when there are no tool calls → `TypeError: 'NoneType' is not iterable`.

**Bug 2 — `message_stream_converter` (streaming) NoneType:**
```python
# Before:
elif len(origin_chunk.choices[0].delta.tool_calls) > 0:
# After:
elif origin_chunk.choices[0].delta.tool_calls and len(...) > 0:
```
`delta.tool_calls` is `None` for non-tool-call chunks → `TypeError: object of type 'NoneType' has no len()`.

**Bug 3 — Single-chunk tool call args not emitted (streaming):**
vLLM's tool parser emits the complete tool call (id + function name + arguments) in a single streaming chunk. The original code only emitted `content_block_start` but not the follow-up `content_block_delta` for the arguments. Fixed by emitting `content_block_delta` immediately after `content_block_start` when args are already present.

**Bug 4 — Empty `delta.content` blocks tool calls (streaming, root cause):**
vLLM sends `delta.content=""` (empty string, not `None`) in the same chunk as `delta.tool_calls`. The content check `if ... delta.content is not None` passes for empty string, enters the text branch, and the `elif tool_calls` is never reached — all tool calls silently dropped.

**Fix:** Check `tool_calls` BEFORE `content` in the streaming dispatch:
```python
# tool calls — check BEFORE content; vLLM sends delta.content="" alongside tool_calls
if origin_chunk.choices[0].delta.tool_calls and len(...) > 0:
    ...
elif origin_chunk.choices[0].delta.content is not None:
    ...
```

---

## Performance

### Benchmarked throughput (thinking disabled, cold prefix cache)

Measured with concurrent async requests, `chat_template_kwargs: {enable_thinking: false}`,
`stream_interval=1`, `max_num_batched_tokens=16384`, `CUDA_DEVICE_MAX_CONNECTIONS=1`.
(Script default is now `32768`; benchmarks were run at `16384` to isolate the effect.)

| Concurrency | System tok/s | Per-req tok/s | TTFT |
|-------------|-------------|---------------|------|
| 1 | 48 | 48 | 49 ms |
| 2 | 62 | 45 avg | 46 ms |
| 4 | 118 | 36 avg | 68 ms |
| 8 | 203 | 31 avg | 126 ms |

System throughput scales well — MoE batching amortises expert dispatch across requests.
Hasn't saturated at C=8; higher concurrency will continue to improve system throughput at the
cost of per-request speed.

**Single-request decode:** ~52 tok/s measured via streaming ITL (with thinking enabled,
TTFT 2.6 s includes thinking phase). GPU memory bandwidth utilisation ~90% of theoretical
ceiling (22 GB weights/GPU at 1.28 TB/s → ~58 tok/s max). Not PCIe bottlenecked.

### Interconnect

All 4 GPUs are PCIe-only (`NODE` topology, no NVLink). PCIe Gen 5 x16 per GPU.
`NCCL_P2P_DISABLE` makes no measurable difference — `CUDA_DEVICE_MAX_CONNECTIONS=1`
already serialises the connections. P2P vs SHM path is a wash for the small (~14 KB)
AllReduce messages produced during decode.

### Thinking mode impact

GLM-4.7 enables chain-of-thought by default (`--reasoning-parser glm45`). For throughput
benchmarks always pass `chat_template_kwargs: {enable_thinking: false}` or results will
be dominated by silent thinking tokens (TTFT 2–3 s, visible tok/s ~4× lower).

### Flags that made a measurable difference

| Change | Effect |
|--------|--------|
| `--max-num-batched-tokens 16384` → `32768` | +72% at C=4, +126% at C=8 vs default; script default raised to 32768 |
| `CUDA_DEVICE_MAX_CONNECTIONS=1` | Marginal at C=1, helps at high concurrency |
| Removing `--enable-log-requests/outputs` | ~5–10% across all concurrency levels |
| `--stream-interval 5` | **Do not use** — causes stalls when `include_usage` is set; no throughput benefit |
| `NCCL_P2P_DISABLE=1` | No measurable effect |
| `--num-scheduler-steps` | Not available in vLLM 0.15.1 V1 engine (V1 async scheduling is equivalent and on by default) |
| MTP speculative decoding | 0% acceptance — neither NVFP4 checkpoint (Salyut1 or Tengyunw) includes MTP draft head weights |

### Logs during Claude Code usage (prefix cache warm)

| Metric | Observed |
|--------|----------|
| Avg prompt throughput | 3,900–7,500 tok/s |
| Avg generation throughput | 15–33 tok/s (rolling avg includes idle time; real decode ~52 tok/s) |
| Prefix cache hit rate | 83–85% |

Prompt throughput is inflated by prefix caching — 83–85% of tokens are served from cache.

### Prefix cache hit rate behaviour

vLLM V1 prefix caching is enabled by default (hash-based matching). Hit rate varies significantly
with session state:

| Scenario | Observed hit rate |
|----------|-------------------|
| Warm cache, proxy running, single session continuing | 80–85% |
| Without proxy (raw Claude Code → vLLM) | 22–40% — see below |
| After server restart, first requests | ~0% (cold) |

---

### Claude Code cache busters (2026-02-21 audit)

Diagnosed using the normalizing proxy in `proxy/` with `--dump-dir` session
diffing. Three distinct cache busters were identified, in order of impact:

#### Cache buster 1 — Per-request billing nonce `cch=` (critical)

Claude Code injects a billing tracking header as the **first block** of the
system prompt on every single request:

```
x-anthropic-billing-header: cc_version=2.1.50.f15; cc_entrypoint=cli; cch=27acd;
```

The `cch=` value is a per-request nonce that changes every time. Because it
sits at token position 0 — before the system instructions, before the tools —
it invalidates the **entire KV cache** on every request. Without a fix, prefix
cache hit rate is 22–40% regardless of how long the session has been running.

**Fix (in proxy):** Strip the `x-anthropic-billing-header:` block from the
system prompt before forwarding. This is Anthropic's internal billing
telemetry and has no effect on model behaviour for a local instance.

This is a Claude Code client issue worth reporting upstream:
https://github.com/anthropics/claude-code/issues — the nonce could be moved
to an HTTP header instead of the system prompt body, which would leave the
cacheable prefix intact.

#### Cache buster 2 — Moving `cache_control` breakpoints (critical)

Claude Code uses Anthropic's prompt caching API. It attaches
`cache_control: {type: ephemeral}` to the latest messages as cache breakpoints,
then **removes** those markers on the next turn (the breakpoint moves forward
to the new tail messages). For vLLM these fields are meaningless, but the
changed JSON content modifies the hash of every affected message, causing a
cache miss for all tokens from that point onward — invalidating the entire
conversation history on every turn.

Identified in `--dump-dir` diffs: turn_001 → turn_002 showed `cache_control`
removed from prior `tool_use` and `tool_result` blocks.

**Fix (in proxy):** Strip all `cache_control` fields from system blocks and
message content blocks before forwarding.

#### Confirmed fixed — post-proxy hit rate

After applying all three proxy fixes, session diffs show only `+N/-0` lines
(pure additions) for every turn after the first. No more modifications to
existing messages. Sessions that previously created a new unique hash on every
request now maintain a stable session ID across all turns.

Observed hit rate post-proxy: **76%+ server-wide average** (vs 22–40% without proxy),
climbing higher on long-running sessions:
- Long-running interactive sessions (50+ msgs): ~85–90%
- Sub-agents with distinct system prompts: starts cold, climbs per-turn
- Server-wide average across mixed concurrent sessions: ~76%

The remaining misses are genuine — different specialized sub-agents (e.g.,
CODIFY Decide/Invite/Forward phases) have genuinely different system prompts
and each maintains its own separate KV cache prefix. This is correct behaviour,
not a fixable bug.

#### Cache buster 3 — MCP tool reordering

With 80+ tool definitions (built-ins + MCP servers), the tool block is large
and injected at position 0 of the formatted GLM-4.7 prompt. MCP servers
reconnect and return tools in arbitrary order, so the tool block hash changes
between requests even when the content is identical.

**Fix (in proxy):** Sort tools alphabetically by name before forwarding.

#### Cache buster 4 — `currentDate` injection (daily)

Claude Code appends `Today's date is YYYY-MM-DD.` to MEMORY.md content before
injecting it into `<system-reminder>` blocks in user messages. Changes at
midnight.

**Fix (in proxy):** `--strip-date` flag. Optional — omit if date awareness
matters.

---

### Proxy

`proxy/` contains a FastAPI normalizing proxy (`uv run`, no install) that
applies all three fixes before forwarding to vLLM. With the proxy running,
observed hit rate returns to 80–85% on warm sessions.

```bash
bash proxy/serve_proxy.sh   # binds 0.0.0.0:30001, forwards to localhost:30000
```

Point Claude Code at the proxy port instead of vLLM directly:
```bash
ANTHROPIC_BASE_URL=http://localhost:30001 claude ...
```

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
