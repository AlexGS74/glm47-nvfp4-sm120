# MiniMax-M2.5 Working Recipes

> Captured from #minimax-m25 Discord thread (RTX6kPRO server) — Updated February 28, 2026
> Model: MiniMaxAI/MiniMax-M2.5 | Hardware: 4x or 8x RTX Pro 6000 (Blackwell SM120)

---

## Recipe 1 — Eric P Sheets' vLLM Official + Extended (4x GPU, Feb 13)

> **Status:** Confirmed working. First recipe in channel. Based on official HuggingFace deploy guide.
> Reference: https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/main/docs/vllm_deploy_guide.md

**Minimal official command:**
```bash
uv venv
source .venv/bin/activate
uv pip install vllm --torch-backend=auto

SAFETENSORS_FAST_GPU=1 vllm serve \
    MiniMaxAI/MiniMax-M2.5 --trust-remote-code \
    --tensor-parallel-size 4 \
    --enable-auto-tool-choice --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think
```

**Extended variant with override-generation-config:**
```bash
env VLLM_SLEEP_WHEN_IDLE=1 SAFETENSORS_FAST_GPU=1 vllm serve \
    MiniMaxAI/MiniMax-M2.5 \
    --tensor-parallel-size 4 \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --override-generation-config '{"temperature":0.7, "top_p": 0.9, "repetition_penalty": 1.15}'
```

**Notes:**
- vLLM reports: Model loading took 53.75 GiB
- Temperature 1.0 (official recommendation) causes looping. Eric recommends 0.7.
- Power consumption improved vs M2.1: M2.1 rarely hit 200W/card at 300W limit. M2.5 goes full tilt.

---

## Recipe 2 — CyySky's SGLang 8-GPU (FP8, ~86 tok/s, Feb 17)

> **Status:** Working ~86 tok/s on 8x GPUs. Includes fused_moe_triton tuning.
> **Optimized with:** kernel tuning script for RTX Pro 6000 Blackwell Max-Q.

**SGLang launch:**
```bash
sudo docker run -it --rm -v /home/gpusvr/:/home/gpusvr/ --ipc=host --shm-size=8g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all --network host \
    lmsysorg/sglang:dev-cu13 bash

python -m sglang.launch_server \
    --model-path /home/gpusvr/MiniMax-M2.5 \
    --tp-size 8 \
    --ep-size 8 \
    --mem-fraction-static 0.8 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --served-model-name llm_model \
    --host 0.0.0.0 \
    --port 9504 \
    --trust-remote-code \
    --fp8-gemm-backend triton \
    --moe-runner-backend triton
```

**Kernel tuning (run once before launch):**
```bash
python /sgl-workspace/sglang/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model /home/gpusvr/MiniMax-M2.5 \
    --tp-size 8 \
    --ep-size 8 \
    --dtype fp8_w8a8 \
    --tune
```

**For vLLM (FP8 on 8 GPUs):**
```bash
# First copy the tuned config file:
cp -v 'E=32,N=1536,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition,dtype=fp8_w8a8,block_shape=[128, 128].json' \
    /home/gpusvr/venv-vllm-nightly/lib/python3.13/site-packages/vllm/model_executor/layers/fused_moe/configs/'E=32,N=1536,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition,dtype=fp8_w8a8,block_shape=[128,128].json'

# Install nightly vLLM:
uv venv --python 3.13 venv-vllm-nightly
source venv-vllm-nightly/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cu130 -U --force-reinstall
uv pip install torch==2.10 torchvision torchaudio triton --index-url https://download.pytorch.org/whl/cu130 -U

# Launch vLLM:
vllm serve MiniMax-M2.5 \
    --served-model-name llm_model \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --enable-auto-tool-choice \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.9 \
    --max-model-len -1 \
    --trust_remote_code \
    --port 9504
```

**Optimal fused_moe config for RTX Pro 6000 (SM120) — E=32,N=1536:**
```json
{
    "1": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        ...
    }
}
```
> Full tuned config file: `E=32,N=1536,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition,dtype=fp8_w8a8,block_shape=[128,128].json`
> See: https://github.com/sgl-project/sglang/issues/18870 for the full tuned config.

---

## Recipe 3 — chisleu's vLLM Docker One-Liner (4x GPU, Feb 17)

> **Status:** Working. Simplest docker run for 4 GPUs.

```bash
docker run --rm --gpus 4 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "SAFETENSORS_FAST_GPU=1" \
    --env "VLLM_SLEEP_WHEN_IDLE=1" \
    -p 5000:5000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model MiniMaxAI/MiniMax-M2.5 \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --served-model-name model \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --port 5000 \
    --reasoning-parser minimax_m2_append_think \
    --trust-remote-code
```

**Performance:** 81 tok/s at 20k context, 61 tok/s at 100k context (chisleu).

---

## Recipe 4 — lukealonso NVFP4 REAP Quant (Single RTX 6000, ~81k context max)

> **Status:** Works on single RTX 6000 (96GB). First single-card capable quant.
> Model: https://huggingface.co/lukealonso/MiniMax-M2.5-REAP-139B-A10B-NVFP4

**Using SGLang (mudaG's recommendation for reasoning parser):**
```bash
# Note: Use --reasoning-parser minimax (not minimax-append-think for NVFP4)
python -m sglang.launch_server \
    --model lukealonso/MiniMax-M2.5-REAP-139B-A10B-NVFP4 \
    --reasoning-parser minimax \
    --tool-call-parser minimax-m2 \
    ...
```

**Notes:**
- Fits on single 96GB card (RTX Pro 6000 or DGX Spark).
- MAX_MODEL_LEN limited to ~81k on single card.
- Qu FRAG: "At 64 users it is FAST" — confirmed very fast at high concurrency.
- DGX Spark benchmarks: pp2048 = 3342 tok/s, tg32 = 16.7 tok/s (single card, limited).
- chisleu: NVFP4 runs at **90 tok/s (20k context) and 60 tok/s (100k)** on 2 GPUs.
- chisleu: "nvfp4 benchmarks well on 2 GPUs vs 4 GPUs in fp8"

**Gotcha:** `minimax-append-think` reasoning parser was "wonky" for some users with NVFP4. Use `minimax` instead (mudaG).

---

## Recipe 5 — Marky's AWQ (2x RTX 6000 Pro, ~114 tok/s low context)

> **Status:** Working. Uses mratsim/Minimax-M2.5-BF16-INT4-AWQ quant.
> **Performance:** ~114 tok/s at low context, ~50 tok/s at 130k+ context. (2x RTX 6000 Pro)

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="mratsim/Minimax-M2.5-BF16-INT4-AWQ"

GPU_UTIL=0.97
SAMPLER_OVERRIDE='{"temperature": 1, "top_p": 0.95, "top_k": 40, "repetition_penalty": 1.1, "frequency_penalty": 0.40}'

export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_USE_FLASHINFER_MOE_FP8=1
export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_FLASHINFER_MOE_BACKEND=throughput
export CUDA_VISIBLE_DEVICES=0,1
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=8
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_SLEEP_WHEN_IDLE=1
export TORCH_FLOAT32_MATMUL_PRECISION=high
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

uv run vllm serve "${MODEL}" \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8080 \
  --stream-interval 1 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --attention-backend FLASHINFER \
  --disable-custom-all-reduce \
  --override-generation-config "${SAMPLER_OVERRIDE}" \
  --block-size 16 \
  --max-num-seqs 64 \
  --max-model-len 196608 \
  --max_num_batched_tokens 16384 \
  --gpu-memory-utilization ${GPU_UTIL}
```

---

## Recipe 6 — Markets & Mayhem's SGLang NVFP4 (2x RTX 6000, Feb 27)

> **Status:** Working. Uses lukealonso NVFP4 quant with SGLang CUDA 13. Includes NSA backends.
> Note: kernel tuning takes 5-10 minutes on first run (AMD 5950x w/128GB RAM).

**Environment:**
```bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=PHB
export NCCL_ALLOC_P2P_NET_LL_BUFFERS=1
export NCCL_MIN_NCHANNELS=8
export OMP_NUM_THREADS=8
export SAFETENSORS_FAST_GPU=1
```

**SGLang command:**
```bash
python3 -m sglang.launch_server \
  --model lukealonso/MiniMax-M2.5-NVFP4 \
  --served-model-name MiniMax-M2.5 \
  --reasoning-parser minimax \
  --tool-call-parser minimax-m2 \
  --enable-torch-compile \
  --trust-remote-code \
  --tp 2 \
  --mem-fraction-static 0.95 \
  --max-running-requests 16 \
  --kv-cache-dtype fp8_e4m3 \
  --quantization modelopt_fp4 \
  --attention-backend flashinfer \
  --moe-runner-backend flashinfer_cutlass \
  --disable-custom-all-reduce \
  --enable-flashinfer-allreduce-fusion \
  --context-length 196608 \
  --dtype auto \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_auto \
  --chunked-prefill-size 16384
```

---

## Quality Benchmark: NVFP4 vs Original (Lavd, Feb 28)

> **Result: NVFP4 quant is slightly BETTER than original at overall accuracy.**
> Dataset: MMLU-style multi-category benchmark, 12032 questions total.

| Category | NVFP4 (lukealonso) | Original FP8/BF16 |
|---|---|---|
| Business | 90.6% | 90.1% |
| Law | 70.7% | 68.4% |
| Psychology | 84.5% | 84.7% |
| Biology | 94.1% | 93.3% |
| Chemistry | 91.4% | 91.3% |
| History | 73.2% | 74.5% |
| Other | 83.0% | 82.7% |
| Health | 83.9% | 82.3% |
| Economics | 88.6% | 89.1% |
| Math | 94.7% | 95.2% |
| Physics | 91.5% | 90.2% |
| Computer Science | 89.3% | 89.5% |
| Philosophy | 79.2% | 77.8% |
| Engineering | 81.5% | 81.8% |
| **ALL CATEGORIES** | **86.2%** | **85.8%** |

> **NVFP4 wins overall** (86.2% vs 85.8%). No meaningful degradation from quantization.

---

## Performance Summary (RTX Pro 6000 / SM120)

| Setup | Tok/s | Context | GPUs | Notes |
|---|---|---|---|---|
| SGLang FP8, kernel-tuned (CyySky) | ~86 | — | 8x | Tuned fused_moe_triton |
| vLLM FP8, expert parallel | — | — | 8x | Official config |
| NVFP4 (lukealonso), SGLang (Markets&Mayhem) | — | 196k | 2x | NSA backends, fp8 KV |
| NVFP4 (chisleu) | 90 | 20k | 2x | — |
| NVFP4 (chisleu) | 60 | 100k | 2x | — |
| AWQ (Marky/remichu_sm) | 114 | low | 2x | — |
| AWQ (remichu_sm) | 50 | 130k+ | 2x | — |
| vLLM (chisleu) | 81 | 20k | 4x | FP8/original |
| vLLM (chisleu) | 61 | 100k | 4x | FP8/original |
| DGX Spark, NVFP4 single card | 16.7 | 4096 | 1x | tg32; pp2048=3342 tok/s |

---

## Key Notes & Gotchas

- **Temperature:** Official recommendation is 1.0, but causes looping at high context. Use 0.7 (Eric P Sheets) or 0.95-1.0 with frequency_penalty.
- **Reasoning parser:** Use `minimax_m2_append_think` for FP8/BF16 models in vLLM. For NVFP4 with SGLang, use `minimax` (mudaG: "append_think was wonky for me").
- **SGLang vs vLLM for NVFP4:** NVFP4 loads in vLLM but is unreliable in SGLang for some users (Marky). Markets & Mayhem has it working in SGLang with CUDA 13.
- **SM120 NVFP4 gap:** luke: "There's a pretty annoying gap where much of the Blackwell support is conditioned on SM100." Performance will improve as better kernels are merged into flashinfer.
- **Expert parallel flag:** Add `--enable-expert-parallel` to vLLM for 8-GPU setups.
- **nothink mode:** Use `--reasoning-parser minimax_m2_append_think` to enable thinking separation. Without it, think tags may bleed into output breaking tool calls.
- **File edit performance:** M2.5 FP8 struggles to keep up with vscode format diffs in coding agents, often falling back to writing whole files (chisleu). Use lower context or faster variant.
- **SGLang crash:** "Rank 0 scheduler is dead, Exit code -9" — Marky on 2x RTX 6000 (32GB RAM only). Lowering gpu-util to 0.9 helped. Use bf16 KV cache or fp8 KV.

---

## Useful Links

- **Official deploy guide (vLLM):** https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/main/docs/vllm_deploy_guide.md
- **NVFP4 REAP quant (single card):** https://huggingface.co/lukealonso/MiniMax-M2.5-REAP-139B-A10B-NVFP4
- **AWQ quant:** https://huggingface.co/mratsim/Minimax-M2.5-BF16-INT4-AWQ
- **DGX Spark NVFP4 forum post:** https://forums.developer.nvidia.com/t/minimax-2-5-reap-nvfp4-on-single-dgx-spark/361248
- **vLLM benchmark suite (Qu FRAG):** https://github.com/shihanqu/vllm-benchmark-suite
- **RTX6kPRO SGLang fused_moe benchmark issue:** https://github.com/sgl-project/sglang/issues/18870
- **vLLM reasoning parser bug report:** https://github.com/vllm-project/vllm/issues/34625


## MMLU-Pro Evaluation Script (Marky, Mar 1)

> Community evaluation script used to run the quality benchmarks above.
> Based on MMLU-Pro repo, enhanced with kimi2.5 and deepseek3.2 improvements (Marky).
> KV cache: int16 (kv 16 per Festr). Temperature 0.1 (orangezed).

```bash
source /data1/llama.env/bin/activate
python /data1/MMLU-Pro/evaluate_from_apiX-.py \
  --url "[API_ENDPOINT_URL]" \
  -m "m25nv" \
  -n 16 \
  -o "/data1/MMLU-Pro/eval_results_m25nv/" \
  --retry 2 \
  --max_tokens 48000 \
  --retry_wrong 2

# View results:
python MMLU-Pro/evalshowpro.py -r /data1/MMLU-Pro/eval_results_m25nv/
```

**Inference params used in eval script (orangezed):**
```python
temperature=0.1,
max_tokens=args.max_tokens,
top_p=0.95,
frequency_penalty=0,
presence_penalty=0,
stream=True,
extra_body={"top_k": 40}
```

**Notes:**
- `--retry_wrong 2`: retries each wrong answer to reduce hallucination noise.
- Results are temperature-dependent (Lavd: "sometimes did better with abliterated version than stock" — noise at this level).
- Festr: benchmark margins (86.2% vs 85.8%) are within probability noise. Multiple runs recommended.
- Use `--n 16` concurrent workers for speed. Adjust based on your API throughput.
