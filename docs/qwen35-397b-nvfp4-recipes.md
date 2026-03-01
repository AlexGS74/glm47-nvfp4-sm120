# Qwen3.5-397B-A17B-NVFP4 Working Recipes

> Captured from #qwen-35 Discord thread (RTX6kPRO server) — Updated February 26, 2026
> Model: `nvidia/Qwen3.5-397B-A17B-NVFP4` | Hardware: 4x Blackwell GPUs (RTX Pro 6000 / SM120)

---

## Recipe 1 — kcramp's Docker Run (Most Cited, Confirmed Working)

> **Status:** Confirmed working by multiple users. Pinpointed fix for "illegal memory access" CUDA error via `disable_flashinfer_q_quantization: true`.


```bash
docker run -d \
  --name vllm-ava-397b \
  --gpus '"device=0,1,2,3"' \
  --ipc=host \
  --shm-size=16g \
  -p 8000:8000 \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e NCCL_P2P_LEVEL=4 \
  -e NCCL_IB_DISABLE=1 \
  -e OMP_NUM_THREADS=8 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_LOG_MODEL_INSPECTION=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -v /mnt/raid0/models/Qwen3.5-397B-A17B-NVFP4:/model:ro \
  vllm/vllm-openai:nightly-f91808ae0ddf750acfdeb351fa072c91d4d678fc \
  --model /model \
  --served-model-name X \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.80 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --attention-backend FLASHINFER \
  --attention-config '{"use_trtllm_attention": false, "disable_flashinfer_q_quantization": true}'
```

**Image:** `vllm/vllm-openai:nightly-f91808ae0ddf750acfdeb351fa072c91d4d678fc`

---

## Recipe 2 — destroyed's Docker Compose (~98-128 tok/s, "Flawlessly Working")

> **Status:** Confirmed working by destroyed. MTP speculative decoding enabled. Image: `cu130-nightly`.

```yaml
services:
  vllm-qwen35-397b:
    image: vllm/vllm-openai:cu130-nightly
    runtime: nvidia
    container_name: vllm-qwen35-397b-nvfp4
    ports:
      - "5001:8000"
    ipc: host
    shm_size: 16gb
    environment:
      - NVIDIA_VISIBLE_DEVICES=0,1,2,3
      - NCCL_P2P_LEVEL=4
      - NCCL_IB_DISABLE=1
      - OMP_NUM_THREADS=8
      - SAFETENSORS_FAST_GPU=1
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
    command: >
      --model /mnt/Qwen3.5-397B-A17B-NVFP4
      --trust-remote-code
      --tensor-parallel-size 4
      --port 8000
      --host 0.0.0.0
      --served-model-name Qwen3.5-397B-A17B
      --gpu-memory-utilization 0.80
      --mm-encoder-tp-mode data
      --mm-processor-cache-type shm
      --enable-auto-tool-choice
      --tool-call-parser qwen3_coder
      --reasoning-parser qwen3
      --enable-prefix-caching
      --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
    volumes:
      - /path/to/Qwen3.5-397B-A17B-NVFP4:/mnt/Qwen3.5-397B-A17B-NVFP4:ro
    restart: unless-stopped
```

> **Tip:** Switch `"mtp"` to `"qwen3_next_mtp"` and tokens to `2` for extra speed (per Festr).
> Destroyed got **125-140 tok/s** on code gen with `num_speculative_tokens: 5`.

---

## Recipe 3 — Festr's Manual Python in Docker (qwen3_next_mtp, 100-200+ tok/s)

> **Status:** Working. 100-105 tok/s normal chat, 200+ tok/s single-request code gen with tokens=5.
> WARNING Feb 26 Update: Use `--tool-call-parser qwen35_coder` (note: `qwen35` not `qwen3`) for working tool calls with speculative decoding. Only `num_speculative_tokens=1` or `2` work; 3, 4, 5 break tool calls even with the new parser.

**Step 1 — Enter docker container interactively:**

```bash
docker run -it --rm --entrypoint /bin/bash \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt:/mnt/ \
  --ipc=host --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus all --network host \
  --mount type=tmpfs,destination=/usr/local/cuda-13.0/compat \
  vllm/vllm-openai:cu130-nightly
```

**Step 2 — Run inside the container (with updated parser):**

```bash
VLLM_LOG_STATS_INTERVAL=1 NCCL_P2P_LEVEL=4 SAFETENSORS_FAST_GPU=1 \
python3 -m vllm.entrypoints.openai.api_server \
  --model nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --host 0.0.0.0 \
  --port 5000 \
  --served-model-name glm-4.7-flash \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 4092 \
  --max-num-seqs 128 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen35_coder \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

> **Note:** `qwen3_next_mtp` is recommended by the official Qwen3 model page on vLLM.
> MTP gains are best for **single-user** setups - gains disappear under concurrent load.

---

## Recipe 4 — chisleu's Docker Run (qwen3_5-cu130 image)

> **Status:** Working. Uses the official `qwen3_5-cu130` image from vLLM recipes page.

```bash
docker run --rm --gpus 4 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "SAFETENSORS_FAST_GPU=1" \
  --env "VLLM_SLEEP_WHEN_IDLE=1" \
  -p 5000:5000 \
  --ipc=host \
  vllm/vllm-openai:qwen3_5-cu130 \
  --model nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --served-model-name model \
  --port 5000 \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --kv-cache-dtype auto \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --trust-remote-code
```

**Image source:** https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html#gb200-deployment-2-nodes-x-4-gpus

---

## Recipe 5 — SGLang / kcramp (NVFP4, fp8_e5m2 KV cache, ~42 tok/s)

> **Status:** Working ~42 tok/s. Lower speed than vLLM but more stable on some setups.

```bash
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
SGLANG_SET_CPU_AFFINITY=true \
SGLANG_DISABLE_CUDNN_CHECK=1 \
python -m sglang.launch_server \
  --model-path /mnt/raid0/models/Qwen3.5-397B-A17B-NVFP4 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e5m2 \
  --tensor-parallel-size 4 \
  --context-length 262144 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --attention-backend triton \
  --moe-runner-backend flashinfer_cutlass \
  --mem-fraction-static 0.80 \
  --chunked-prefill-size 4096 \
  --max-running-requests 4 \
  --cuda-graph-max-bs 16 \
  --preferred-sampling-params '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  --host 0.0.0.0 \
  --port 8000
```

---

## Recipe 6 — Ixtrix's SGLang with Speculative Decoding

> **Status:** Working. The faster alias omits `--speculative-draft-model-quantization unquant` which caused 20% slowdown.

**Faster variant (recommended):**

```bash
alias qwen-sglang='source /models/sglang/.venv/bin/activate && \
  export CUDA_HOME=/usr/local/cuda-12.9 && \
  export CUDA_VISIBLE_DEVICES=0,1,2,3 && \
  python -m sglang.launch_server \
  --model /models/Qwen3.5-397B-A17B-NVFP4 \
  --tensor-parallel-size 4 \
  --quantization modelopt_fp4 \
  --trust-remote-code \
  --attention-backend triton \
  --moe-runner-backend flashinfer_cutlass \
  --fp4-gemm-backend flashinfer_cudnn \
  --host 0.0.0.0 \
  --port 8000 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3'
```

**With speculative decoding (slightly slower - quantization flag bug):**

```bash
alias qwen-sglang-spec='source /models/sglang/.venv/bin/activate && \
  export CUDA_HOME=/usr/local/cuda-12.9 && \
  export CUDA_VISIBLE_DEVICES=0,1,2,3 && \
  python -m sglang.launch_server \
  --model /models/Qwen3.5-397B-A17B-NVFP4 \
  --tensor-parallel-size 4 \
  --quantization modelopt_fp4 \
  --trust-remote-code \
  --attention-backend triton \
  --moe-runner-backend flashinfer_cutlass \
  --fp4-gemm-backend flashinfer_cudnn \
  --host 0.0.0.0 \
  --port 8000 \
  --speculative-eagle-topk 1 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4'
```

---

## Recipe 7 — Festr's SGLang Source Build (First Working Recipe, Feb 19)

> **Status:** Working ~85 tok/s. Requires building SGLang from a specific branch.
> Credit: https://huggingface.co/vincentzed-hf/Qwen3.5-397B-A17B-NVFP4/discussions/1

```bash
git clone --branch feat/transformers-v5-qwen35-nvfp4 https://github.com/joninco/sglang.git
cd sglang
pip install -e "python[all]" --no-build-isolation --no-deps
NCCL_P2P_LEVEL=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m sglang.launch_server \
  --model-path vincentzed-hf/Qwen3.5-397B-A17B-NVFP4 \
  --tp 4 \
  --host 0.0.0.0 \
  --port 5000 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend triton \
  --moe-runner-backend flashinfer_cutlass \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-algo NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-draft-model-quantization unquant \
  --context-length 262144 \
  --mem-fraction-static 0.90 \
  --max-running-requests 4 \
  --cuda-graph-max-bs 16 \
  --chunked-prefill-size 4096 \
  --schedule-policy lpm \
  --trust-remote-code
```

---

## Recipe 8 — CyySky's SGLang FP8 (8-GPU, 75-125 tok/s, Channel's First)

> **Status:** Working 75-125 tok/s on 8x GPUs. Uses FP8 model variant.

```bash
# Download the FP8 model:
HF_XET_HIGH_PERFORMANCE=1 HF_ENDPOINT=https://hf-mirror.com \
  hf download Qwen/Qwen3.5-397B-A17B-FP8 --local-dir Qwen3.5-397B-A17B-FP8

# Pull and enter docker:
sudo docker pull lmsysorg/sglang:dev-cu13
sudo docker run -it --rm -v /home/gpusvr/:/home/gpusvr/ \
  --ipc=host --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus all --network host lmsysorg/sglang:dev-cu13 bash

# Inside container (SGLang, FP8, 8 GPU):
python -m sglang.launch_server \
  --model-path /home/gpusvr/Qwen3.5-397B-A17B-FP8 \
  --host 0.0.0.0 --port 9501 \
  --tp-size 8 \
  --mem-fraction-static 0.8 \
  --context-length 262144 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --served-model-name llm_model \
  --speculative-algo NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --attention-backend triton \
  --fp8-gemm-backend triton \
  --moe-runner-backend triton

# Alternative: vLLM (FP8, 8 GPU):
vllm serve Qwen3.5-397B-A17B-FP8 \
  --port 9501 \
  --tensor-parallel-size 8 \
  --max-model-len -1 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.9 \
  --served-model-name llm_model \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

---

## Critical Config Fixes (NVFP4 model)

> **Note:** destroyed confirmed the config.json edit is **NOT required** if using `cu130-nightly` correctly. Try without first.
> If you get weight-loading assertion errors, add these to the ignore list in **both** `config.json` and `hf_quant_config.json`:

```
"model.language_model.layers..mlp.gate"
"mtp.fc"
```

Also: **manually download `hf_quant_config.json`** from HuggingFace if it wasn't included in your download - it's required for the NVFP4 model to load.

---

## Tool Call Fix - New Parser: qwen35_coder (Feb 26, 2026)

> **Status:** Confirmed working with vLLM PR #35347. Only num_speculative_tokens 1 or 2 work with tool calls; 3-5 break tool calls.

**Key discovery (Festr, 6:16 AM Feb 26):** vLLM added a **new tool call parser** `qwen35_coder` specifically for Qwen 3.5. The old `qwen3_coder` parser causes `IndexError: list index out of range` when combined with speculative decoding. Switching to `qwen35_coder` restores tool call reliability with `num_speculative_tokens=2`.

**What works vs what doesn't with the new parser:**

| num_speculative_tokens | Tool calls work? |
|---|---|
| 1 | Working |
| 2 | Working |
| 3 | Broken |
| 4 | Broken |
| 5 | Broken |

**Related PR:** https://github.com/vllm-project/vllm/pull/35347

**To enable:** Replace `--tool-call-parser qwen3_coder` with `--tool-call-parser qwen35_coder` in your vLLM launch command.

> **Note:** The PR fix itself (sunqingn7's json malformation fix) did not fully resolve the issue on its own. The key was using the new parser name `qwen35_coder` that the PR introduced. Also tested: `VLLM_USE_V2_MODEL_RUNNER=1` - not implemented, does not work.

---

## Bonus - Video Input with vLLM (Ixtrix, Feb 26)

> Enable video input when launching vLLM with the `--media-io-kwargs` flag. Currently only supported in vLLM (not SGLang).

**Launch flag:**

```bash
--media-io-kwargs '{"video": {"num_frames": -1}}'
```

**API call example:**

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-397B-A17B",
    messages=messages,
    max_tokens=81920,
    temperature=0.6,
    top_p=0.95,
    extra_body={
        "top_k": 20,
        "mm_processor_kwargs": {"fps": 2, "do_sample_frames": True},
    },
)
```

> By default: `fps=2` and `do_sample_frames=True`. Video frame sampling can be configured via `extra_body` by setting `fps`.

---

## Bonus - FastAPI Proxy for OpenCode/SGLang Compatibility

> By Ixtrix. Normalizes multiple system messages into one, fixing "bad prompt" errors when using OpenCode with SGLang.

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
import httpx, json, logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = FastAPI()
SGLANG_URL = "http://localhost:8000"

def normalize_messages(messages: list) -> list:
    system_msgs = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    if not system_msgs:
        return rest
    if len(system_msgs) == 1:
        return system_msgs + rest
    merged_content = "\n\n".join(
        m["content"] if isinstance(m["content"], str)
        else " ".join(part.get("text","") for part in m["content"] if isinstance(part, dict))
        for m in system_msgs
    )
    merged = {"role": "system", "content": merged_content}
    return [merged] + rest

@app.post("/v1/chat/completions")
async def proxy(request: Request):
    body = await request.json()
    if "messages" in body:
        body["messages"] = normalize_messages(body["messages"])
    # Forward to SGLANG_URL ...
```

---

## Performance Summary (4x Blackwell / RTX Pro 6000)

| Setup | Tok/s (single req) | Notes |
|---|---|---|
| vLLM + qwen3_next_mtp, spec_tokens=5 | 200-275 | Code gen only, single req; tool calls broken |
| vLLM + qwen3_next_mtp, spec_tokens=2 | 125-140 | Code gen; tool calls work w/ qwen35_coder parser |
| vLLM + MTP method=mtp, spec_tokens=1 | 98-105 | Normal chat; tool calls work |
| vLLM, no MTP | ~64 | Baseline; tool calls fully reliable |
| vLLM, --decode-context-parallel-size 2 | 80-85 | Larger KV cache (1.8M), slower TPS |
| SGLang NVFP4 (kcramp) | ~42 | Stable, lower speed |
| SGLang FP8, 8 GPU | 75-125 | Requires 8 GPUs |

> WARNING: MTP - Speculative decoding gains disappear under concurrent load. Best for single-user setups.
> WARNING: Tool calls with spec decode - Use --tool-call-parser qwen35_coder (not qwen3_coder) and keep num_speculative_tokens at 2 or below.

---

## Useful Links

- **vLLM official recipe:** https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html#text-only
- **SGLang NVFP4 Gist (SM120):** https://gist.github.com/catid/87cca824963f17fe7479a0ed26221397
- **HF model page (NVFP4):** https://huggingface.co/vincentzed-hf/Qwen3.5-397B-A17B-NVFP4
- **vLLM mlp.gate bugfix PR:** https://github.com/vllm-project/vllm/pull/35156
- **vLLM Qwen3.5 tool call fix PR:** https://github.com/vllm-project/vllm/pull/35347
- **SGLang NVFP4 support PR:** https://github.com/sgl-project/sglang/pull/18937
- **HF discussion thread:** https://huggingface.co/vincentzed-hf/Qwen3.5-397B-A17B-NVFP4/discussions/1
- **Benchmark comparison (122b vs 397b):** https://qwen122-vs-397-20260224-1154.surge.sh
- **Qwen 3.5 FP8 quant (community):** https://huggingface.co/Shifusen/Qwen3.5-122B-A10B-FP8


---

## Recipe 9 — orangezed's Clean Python Launch (MTP=5, ~150 tok/s, Feb 27)

> **Status:** Confirmed working on 4x Blackwell. Cleanest minimal recipe for MTP=5.
> **Prerequisite:** Add "mtp.fc" to quantization_config.ignore in config.json (see Critical Config Fixes below).

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.5-397B-A17B-NVFP4 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 4092 \
  --max-num-seqs 128 \
  --trust-remote-code \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":5}'
```

**Result:** ~150 tok/s decode (batch 1) on 4x Blackwell, up from ~75 tok/s without MTP.

**vLLM build:** Any recent main (~ec8f943db, Feb 26 2026). Cherry-pick:
- PR #35219 — FlashInfer accuracy fix for Blackwell GPUs (needed for correctness)
- PR #35421 — Fixes tool call streaming with num_speculative_tokens > 1 (only if using tool calls + MTP)

---

## Recipe 10 — Ixtrix's Full Docker Alias with Patch Files Volume-Mounted (Feb 27)

> **Status:** Confirmed working. Includes all patches for tool calls, collective_fusion, and video input.
> **Note:** Uses cu130-nightly image. Patches are mounted from host paths.

```bash
alias qwen-vllm-mtp='
docker stop vllm-ava-397b 2>/dev/null;
docker rm vllm-ava-397b 2>/dev/null;
docker run -d \
  --name vllm-ava-397b \
  --runtime=nvidia \
  --ipc=host \
  --shm-size=16g \
  -p 8000:8000 \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
  -e NCCL_P2P_LEVEL=4 \
  -e NCCL_IB_DISABLE=1 \
  -e OMP_NUM_THREADS=6 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -v /models/Qwen3.5-397B-A17B-NVFP4-custom-config.json:/model/config.json:ro \
  -v /models/Qwen3.5-397B-A17B-NVFP4:/model:ro \
  -v ~/vllm-fix/fusion/collective_fusion.py:/usr/local/lib/python3.12/dist-packages/vllm/compilation/passes/fusion/collective_fusion.py:ro \
  -v ~/vllm-fix/chat_completion/serving.py:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py:ro \
  -v ~/vllm-fix/tool_parsers/qwen3coder_tool_parser.py:/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/qwen3coder_tool_parser.py:ro \
  vllm/vllm-openai:cu130-nightly \
  --model /model \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --media-io-kwargs '"'"'{"video": {"num_frames": -1}}'"'"' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --speculative-config '"'"'{"method":"qwen3_next_mtp","num_speculative_tokens":2}'"'"' \
  --served-model-name Qwen3.5-397B-A17B-NVFP4 Qwen3'
```

**Note from Ixtrix:** "i had to add some custom commands because GPUs dont like my VM setup"
- Uses custom config.json mounted separately (with mtp.fc in ignore list)
- Patch files mounted from ~/vllm-fix/ directory on host

**config.json snippet (required for MTP to work):**
```json
{
  "architectures": ["Qwen3_5MoeForConditionalGeneration"],
  "dtype": "bfloat16",
  "image_token_id": 248056,
  ...
}
```

---

## Critical Config Fix — mtp.fc in quantization_config.ignore (Feb 27)

> **Required for MTP=5 on NVFP4 models.** Without this, vLLM crashes with shape mismatch error.

Edit `config.json` in the model directory. Add **"mtp.fc"** to quantization_config.ignore:

```json
{
  "quantization_config": {
    "ignore": [
      ...existing entries...,
      "mtp.fc"
    ]
  }
}
```

**Why:** vLLM tries to load the MTP projection layer as NVFP4-quantized, but it is actually BF16 in the checkpoint. This causes a shape mismatch crash. Adding it to the ignore list tells vLLM to leave it in BF16.

---

## Thinking Mode Fix — enable_thinking Flag (Feb 27)

> vLLM does NOT enable thinking tags by default. Must pass explicitly via extra_body.

```python
# In your API call:
extra_body = {"chat_template_kwargs": {"enable_thinking": True}}

# Or for no-thinking mode:
extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
```

**Without this flag:** model thinks internally but outputs no think tags. The inference still works but thinking is suppressed in output.

---

## Useful Links — Updated (Feb 28, 2026)

- **PR #35219** (FlashInfer accuracy fix, Blackwell): https://github.com/vllm-project/vllm/pull/35219
- **PR #35421** (tool call streaming + MTP>1 fix): https://github.com/vllm-project/vllm/pull/35421
- **PR #35581** (MTP kernel fix, 2-8% throughput improvement): https://github.com/vllm-project/vllm/pull/35581
- **PR #35615** (Festr's new tool call fix): https://github.com/vllm-project/vllm/pull/35615
- **PR #33088** (Async TP sum reduction fix): https://github.com/vllm-project/vllm/pull/33088
- **EleutherAI lm-evaluation-harness:** https://github.com/EleutherAI/lm-evaluation-harness
- **Sehyo NVFP4 quants (122B popular):** https://huggingface.co/Sehyo/Qwen3.5-122B-A10B-NVFP4
- **YouTube: 397B vs 122B comparison (xCreate):** https://youtu.be/OE5KdF4spss?si=BXH8oMsRDNS6XE5N
