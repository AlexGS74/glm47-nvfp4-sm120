# GLM-4.7 FP8 Recipes — RTX PRO 6000 (SM120)

Community-tested configurations for running GLM-4.7 FP8 on RTX PRO 6000 (Blackwell SM120).
Channel: #glm-47 | Server: RTX6kPRO
Last updated: March 1, 2026

---

## Recipe 1: Festr's SGLang Launch (4-GPU, Feb 15 2026)

**First community recipe — the definitive baseline**

```bash
python -m sglang.launch_server \
  --model-path THUDM/GLM-4.7-9B-0414 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e5m2 \
  --tp 4 \
  --enable-torch-compile \
  --chunked-prefill-size 512 \
  --mem-fraction-static 0.88 \
  --host 0.0.0.0 \
  --port 30000
```

Performance: ~100 tok/s generation on 4x RTX PRO 6000
Author: Festr
Notes: Use FP8 KV cache (fp8_e5m2) to maximize context length on 4 cards

---

## Recipe 2: chisleu's Docker Version (Feb 16 2026)

```bash
docker run --gpus all --rm \
  -p 30000:30000 \
  -v /path/to/models:/models \
  lmsysorg/sglang:latest \
  python -m sglang.launch_server \
    --model-path /models/GLM-4.7-9B-0414 \
    --quantization fp8 \
    --kv-cache-dtype fp8_e5m2 \
    --tp 4 \
    --chunked-prefill-size 512 \
    --mem-fraction-static 0.88 \
    --host 0.0.0.0 \
    --port 30000
```

Known issue: Triton OOM error => reduce --cuda-graph-max-bs
Fix: Festr noted missing fused_moe JSON config causes this
Author: chisleu

---

## Recipe 3: root-754B's vLLM FP8 Command (Feb 28 2026)

```bash
NCCL_P2P_DISABLE=1 \
NCCL_MIN_NCHANNELS=32 \
NCCL_MAX_NCHANNELS=64 \
NCCL_BUFFSIZE=67108864 \
CUDA_DEVICE_MAX_CONNECTIONS=64 \
python -m vllm.entrypoints.openai.api_server \
  --model THUDM/GLM-4.7-9B-0414 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e5m2 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --host 0.0.0.0 \
  --port 8000
```

Performance: 18.47 tok/s sequential, 24.97 tok/s parallel (1.35x speedup)
Hardware: 4x RTX PRO 6000 at 400W TDP
Author: root-754B
Notes: MTP disabled for maximum KV cache space; performance identical to MTP-enabled

---

## Recipe 4: Festr's Updated SGLang with dp 2 (Feb 27 2026, 8-GPU only)

```bash
python -m sglang.launch_server \
  --model-path THUDM/GLM-4.7-9B-0414 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e5m2 \
  --tp 4 \
  --dp 2 \
  --enable-torch-compile \
  --chunked-prefill-size 512 \
  --mem-fraction-static 0.88 \
  --host 0.0.0.0 \
  --port 30000
```

IMPORTANT: --dp 2 requires 8 GPUs (4 TP x 2 DP). Do NOT use on 4-card systems.
Performance: Higher throughput for multi-user concurrent inference
Author: Festr

---

## Recipe 5: root-754B's FP8 Quantization Script (Feb 27 2026)

Use this to quantize your own GLM-4.7 checkpoint to block-FP8 format.

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "THUDM/GLM-4.7-9B-0414"
OUTPUT_DIR = "./GLM-4.7-9B-FP8-block"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=["lm_head"],
    ignore_patterns=["re:.*mlp\\.gate$", "kv_a_proj_with_mqa", "q_a_proj", "model.embed_tokens"],
)

oneshot(model=model, recipe=recipe)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Quantization complete:", OUTPUT_DIR)
```

Ignored layers: lm_head, all MLP gate layers (mlp.gate), kv_a_proj_with_mqa, q_a_proj, model.embed_tokens
Format: Block FP8 (not KV quant)
Author: root-754B
Notes: RTX PRO 6000 4-card setup at 400W TDP

Output quantization_config.json:

```json
{
  "config_groups": {
    "group_0": {
      "input_activations": {
        "dynamic": true,
        "group_size": 128,
        "num_bits": 8,
        "strategy": "group",
        "type": "float"
      },
      "weights": {
        "block_structure": [128, 128],
        "dynamic": false,
        "num_bits": 8,
        "observer": "static_minmax",
        "strategy": "block",
        "type": "float"
      }
    }
  },
  "quant_method": "compressed-tensors",
  "quantization_status": "compressed"
}
```

---

## Performance Comparison (March 1, 2026)

### SGLang vs vLLM on 4x RTX PRO 6000

| Backend | Mode | Throughput | Notes |
|---------|------|------------|-------|
| SGLang | Single batch | ~100 tok/s | FP8 + FP8 KV cache, torch compile |
| SGLang | Multi-user | ~103 tok/s | chisleu report (Mar 1) |
| vLLM | Sequential | 18.47 tok/s | root-754B, no MTP |
| vLLM | Parallel | 24.97 tok/s | root-754B, 1.35x speedup |
| AWQ INT4 | Generation | 60-70 tok/s | AlexGS, 4-card, 93.6% prefix cache hit |

### Quantization Quality Comparison (Feb 28 2026, root-754B)

Cosine similarity between full precision and quantized outputs across 20 heads:

| Quantization | Cosine Similarity | Head-to-Head Win Rate | Notes |
|-------------|-------------------|----------------------|-------|
| 3s-FP8 (self-quantized) | 0.99980 | 55% (11/20) | Best overall |
| arch-FP8 (THUDM official) | 0.99984 | 45% (9/20) | Slightly higher avg cosine |
| NVFP4 | 0.99754 | 0% (0/20) | Noticeably lower quality |

Conclusion: NVFP4 quality is noticeably worse for code generation. FP8 (self-quantized or official) is preferred.

---

## EAGLE Speculative Decoding Notes (Feb 28 2026)

- SGLang's EAGLE (MTP) acceptance rate: ~6% per position
- Low acceptance rate = minimal throughput benefit for GLM-4.7 FP8
- root-754B: disabled MTP for more cache, performance was right about identical
- AlexGS humaneval: MTP on = 43%, MTP off = 45% (slight accuracy improvement with MTP off)
- Recommendation: Disable MTP to maximize KV cache space on 4-card setups

## MTP Weight Availability Across Checkpoints (March 3, 2026)

GLM-4.7 uses `num_nextn_predict_layers: 1` in config.json. The MTP draft layer is
stored as `model.layers.92.*` (layer 92, after the 92 regular layers 0-91) in a
separate `mtp.safetensors` file. It contains a full MoE layer (160 experts + attention +
shared_head) — same architecture as a regular transformer block.

| Checkpoint | MTP weights | mtp.safetensors | Keys |
|---|---|---|---|
| `zai-org/GLM-4.7` (BF16) | Yes | Yes (BF16) | ~15 keys (dense MoE layer) |
| `zai-org/GLM-4.7-FP8` | Yes | Yes (FP8 quantized) | 989 keys (with weight_scale) |
| `Salyut1/GLM-4.7-NVFP4` | **No** | Missing | 0 keys |
| `Tengyunw/GLM-4.7-NVFP4` | **No** | Missing | 0 keys |

**Official vLLM recipe** (https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM.html)
recommends `--speculative-config.num_speculative_tokens 1` with 90%+ acceptance rate.
@aabbccddwasd reports ~100 tok/s on GLM-4.7 with MTP enabled (vs ~60 tok/s without).

**To enable MTP on NVFP4**: graft `mtp.safetensors` from the BF16 original onto the
NVFP4 checkpoint and update `model.safetensors.index.json`. The MTP layer stays BF16.
See `scripts/graft_mtp_to_nvfp4.sh`. Also requires adding `model.layers.92` to
`quantization_config.ignore` in config.json (the script does this automatically).

**To enable MTP on FP8** (vLLM):
```bash
SPEC_TOKENS=1 ./scripts/serve_glm47_fp8_vllm.sh
```

### MTP Testing Results (March 3, 2026 — AlexGS, 4x RTX PRO 6000)

**Conclusion: MTP is useless on GLM-4.7 at any precision.** The official 90% acceptance
claim is not reproducible. The draft head is fundamentally weak — acceptance rates are
too low to overcome the overhead of running a full MoE draft layer.

| Config | Acceptance rate | Single-req tok/s | Notes |
|---|---|---|---|
| NVFP4 + BF16 MTP graft | 1% | ~10 | Grafted BF16 mtp.safetensors onto Salyut1 NVFP4 |
| FP8 native MTP | 12-50% | 54 | MTP weights are FP8, same precision as main model |
| NVFP4 no MTP | n/a | 60 | Best NVFP4 performance |
| FP8 no MTP | n/a | 56 | Slightly slower single-req than NVFP4 (larger model) |

### Benchmark: NVFP4 no MTP (March 3, 2026)

vLLM 0.16.1rc1, TRITON_ATTN, NCCL_P2P_LEVEL=4, 512in/256out, 32 prompts

| Concurrency | Sys tok/s | TPOT median ms | TTFT median ms | TTFT P99 ms |
|---|---|---|---|---|
| 1 | 60 | 16.58 | 147.40 | 192.96 |
| 2 | 90 | 22.16 | 92.68 | 139.16 |
| 4 | 159 | 25.19 | 110.89 | 172.10 |
| 8 | 234 | 34.21 | 131.51 | 209.57 |
| 16 | 360 | 44.49 | 158.91 | 261.15 |

### Benchmark: FP8 with MTP=1 (March 3, 2026)

| Concurrency | Sys tok/s | TPOT median ms | TTFT median ms | TTFT P99 ms |
|---|---|---|---|---|
| 1 | 54 | 18.36 | 169.00 | 601.86 |

Cancelled — slower than no-MTP due to draft overhead with low acceptance.

### Benchmark: FP8 no MTP (March 3, 2026)

| Concurrency | Sys tok/s | TPOT median ms | TTFT median ms | TTFT P99 ms |
|---|---|---|---|---|
| 1 | 56 | 17.82 | 152.54 | 470.30 |
| 2 | 87 | 22.93 | 103.13 | 143.82 |
| 4 | 141 | 28.27 | 127.54 | 289.96 |
| 8 | 206 | 38.77 | 153.91 | 599.09 |
| 16 | 316 | 50.57 | 191.65 | 664.94 |

---

## NCCL High-Throughput Config for Turin (Dual-NUMA, Feb 27 2026)

For dual-CPU (Turin/EPYC 9005) servers with split NUMA topology:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_MIN_NCHANNELS=32
export NCCL_MAX_NCHANNELS=64
export NCCL_BUFFSIZE=67108864
export CUDA_DEVICE_MAX_CONNECTIONS=64
```

Hardware: GPU0-3 on NUMA0, GPU4-5 on NUMA1 (dual Turin EPYC)
Purpose: Avoid P2P bottleneck across NUMA nodes; maximize channel utilization
Author: Festr

---

## Context Length Warnings

### 4-Card FP8 Context Limitation

Error encountered by chisleu (Mar 1):
  Input length (171455 tokens) exceeds maximum allowed (139297 tokens)

- 4x RTX PRO 6000 with FP8 model + FP8 KV cache: max context ~139k tokens
- Fix: Reduce context length, or disable MTP to free KV cache space

### FP8 KV Cache Warning (March 3, 2026 — CRITICAL)

**Do NOT use `--kv-cache-dtype fp8` on SM120 with multi-session long-context workloads.**

Symptom: Generation throughput drops to 6-9 tok/s (vs expected 60 tok/s) with 2+ concurrent
sessions at 50K+ context. GPUs show 100% utilization but produce almost no tokens. Single
sessions may appear acceptable; the problem manifests under concurrent load.

Root cause: FP8 KV cache adds quantize/dequantize overhead on every attention step. With
2 concurrent long-context sessions (50-100K tokens each), the FP8 quant/dequant per token
doubles and becomes the bottleneck — GPUs are busy doing FP8 conversions instead of useful
attention math.

Fix: Remove `--kv-cache-dtype fp8` and let vLLM use the default (auto/bf16). This uses more
VRAM per KV entry but eliminates the conversion overhead. Throughput recovers immediately.

The tradeoff: FP8 KV cache halves KV memory (fits longer context), but the conversion overhead
kills decode throughput under concurrent load on SM120. For Claude Code usage with multiple
sessions, default bf16 KV cache is significantly faster.

Note: prefix caching (enabled by default in vLLM V1) is NOT the cause — it was initially
suspected but the slowdown persisted with prefix caching on. The root cause is FP8 KV dtype.

Author: AlexGS (tested on 4x RTX PRO 6000, vLLM 0.16.1rc1, GLM-4.7 NVFP4)

---

### Expert Parallel Warning

- --enable-expert-parallel was always too slow for GLM-4.7
- Saturates PCIe communication too fast on 4-card setups
- Do NOT use --enable-expert-parallel with GLM-4.7 on 4 cards

---

## Quantization Recommendations

| Scenario | Recommended | Avoid |
|----------|-------------|-------|
| 4x RTX PRO 6000, quality priority | FP8 (official or self-quant) | NVFP4 |
| 4x RTX PRO 6000, speed priority | FP8 still faster on SM120 | NVFP4 (slower on Blackwell) |
| 8-card setup | FP8 with --dp 2 in SGLang | INT4/AWQ |
| Code generation | FP8 mandatory | NVFP4 (code quality noticeably worse) |

Key findings:
- NVFP4 is actually FASTER than FP8 on vLLM (60 vs 56 tok/s single-req, 360 vs 316 at c=16)
  - Smaller model = less memory bandwidth per token
  - FP8 advantage was SGLang-specific (earlier tests); vLLM tells a different story
- NVFP4 has noticeably worse code quality (root-754B, Mar 1)
- INT4/AWQ runs at 60-70 tok/s vs FP8's ~100 tok/s in SGLang
- GPTQ INT8 mixed INT4 was best quant for 4-card before FP8 (darkstar000, Feb 26)
- MTP is useless on GLM-4.7 at any precision (1-50% acceptance) — disable it

---

## See Also

- Thread summary: glm47-fp8-thread-summary.md
- Related channels: #sglang, #vllm (created by Festr on Feb 25)
- GLM-4.7 is multimodal; inference does not get slow for large context (Festr, Feb 25)
