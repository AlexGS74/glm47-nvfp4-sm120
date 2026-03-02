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
- NVFP4 is SLOWER than FP8 on RTX PRO 6000 (SM120) - contrary to expectations
- NVFP4 has noticeably worse code quality (root-754B, Mar 1)
- INT4/AWQ runs at 60-70 tok/s vs FP8's ~100 tok/s in SGLang
- GPTQ INT8 mixed INT4 was best quant for 4-card before FP8 (darkstar000, Feb 26)

---

## See Also

- Thread summary: glm47-fp8-thread-summary.md
- Related channels: #sglang, #vllm (created by Festr on Feb 25)
- GLM-4.7 is multimodal; inference does not get slow for large context (Festr, Feb 25)
