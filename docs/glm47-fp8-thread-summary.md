# GLM-4.7 FP8 Thread Summary -- RTX PRO 6000 (SM120)

Source: #glm-47, RTX6kPRO Discord | Period: Feb 15 - Mar 1, 2026 | Hardware: 4x/8x RTX PRO 6000 SM120

## TL;DR

Channel launched Feb 15 with Festr's SGLang recipe. FP8 ~100 tok/s via SGLang beats vLLM 18-25 tok/s; NVFP4 slower AND lower quality on Blackwell; EAGLE ~6% acceptance rate; critical Triton JSON config discovered March 1.

---

## Timeline

### Feb 15 -- Channel Launch

- SGLang: --quantization fp8 --kv-cache-dtype fp8_e5m2 --tp 4 --enable-torch-compile
- Performance: ~100 tok/s on 4x RTX PRO 6000 (Recipe 1)

### Feb 16 -- First Attempts

- chisleu Docker version gets Triton OOM -- fix: reduce --cuda-graph-max-bs
- Festr: missing fused_moe JSON config causes issues

### Feb 25 -- FP8 vs NVFP4 Debate

- AlexGS AWQ INT4: 60-70 tok/s, 93.6% prefix cache hit rate
- Festr: NVFP4 SLOWER than FP8 on SM120 -- contrary to expectations
- FP8 on 4 cards with FP8 KV cache: ~100 tok/s confirmed
- Festr: GLM 4.7 is the best model I ever used so far (on 4 GPUs)

### Feb 26 -- NVFP4 Verdict

- Festr: precision lost was noticeable compared to FP8
- darkstar000: GPTQ INT8 mixed INT4 was best quant before FP8
- root-754B: all non-FP8 quants fail or reject on RTX 6k
### Feb 27 -- vLLM vs SGLang; Quant Script

- JTazz vLLM: only 8 tok/s single batch on 4 cards
- SGLang: 100 tok/s -- vLLM always slower for MLA models
- Festr: --enable-expert-parallel too slow on 4 cards
- Updated SGLang with --dp 2 for 8-GPU ONLY (Recipe 4)
- root-754B FP8 quant script (Recipe 5): FP8_BLOCK, ignores lm_head/mlp.gate/kv_a_proj
- root-754B Turin dual-NUMA: GPU0-3 NUMA0, GPU4-5 NUMA1
- Festr NCCL: NCCL_P2P_DISABLE=1, NCHANNELS=32/64, BUFFSIZE=67108864

### Feb 28 -- Benchmarks and Cosine Similarity

root-754B vLLM FP8 (Recipe 3):
- Sequential: 18.47 tok/s; Parallel: 24.97 tok/s (1.35x)

root-754B cosine similarity (20 heads):

| Quantization | Cosine Similarity | Win Rate |
|---|---|---|
| 3s-FP8 (self-quantized) | 0.99980 | 55% (11/20) |
| arch-FP8 (THUDM official) | 0.99984 | 45% (9/20) |
| NVFP4 | 0.99754 | 0% (0/20) |

NVFP4 loses every head-to-head comparison.

- chisleu: 103 tok/s SGLang; TTFT high (5-9s)
- EAGLE MTP acceptance: ~6% per position (very low)
- AlexGS humaneval: MTP on=43%, MTP off=45%
- root-754B: code quality from NVFP4 noticeably worse than FP8
### Mar 1 (Day) -- Context Limits

- chisleu: Input length 171455 exceeds max 139297 tokens
- Festr: disable MTP to get more KV cache space
- AlexGS: vLLM PR #34053 anthropic response fix (thinking in CC)
- Community begins migrating to Qwen 3.5

### Mar 1 (Evening, 21:04-21:15 UTC) -- Critical SM120 Triton JSON Fix

root-754B discovers vLLM warning:
  Performance might be sub-optimal! Config file not found:
  ...E=40,N=1536,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,...json

Festr fix: copy the Server Edition JSON to missing paths:
  File: E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,dtype=fp8_w8a8.json
  Source: voipmonitor.org hosted
  Festr: does not matter what E/N values -- just use this file for any missing RTX json files

- chisleu: sm120.json file is necessary magic
- Festr (00:29 UTC): configs accelerate serving via custom Triton kernels
- Default configs MISSING for RTX PRO 6000 Blackwell
- Note: file must change for non-w8a8 quants

See Recipe 6 in glm47-fp8-recipes.md for exact fix.

### Mar 1 (Evening) -- MTP Discussion

- root-754B: EAGLE acceptance was only 15% -- draft model issue
- Festr: impossible -- something is wrong

---

## Key Findings

| Backend | Throughput | Notes |
|---|---|---|
| SGLang FP8 | ~100-103 tok/s | Best; torch compile; FP8 KV cache |
| vLLM FP8 sequential | 18.47 tok/s | root-754B |
| vLLM FP8 parallel | 24.97 tok/s | 1.35x speedup |
| AWQ INT4 | 60-70 tok/s | With prefix caching |

- FP8 wins code quality, precision, throughput vs NVFP4
- NVFP4 only advantage: slightly longer context on 4 cards
- 4-card FP8 max context: ~139k tokens; disable MTP for more headroom
- fused_moe Triton JSON required -- missing by default, copy Server Edition file
- NCCL config for Turin: P2P_DISABLE=1, 32-64 channels, large BUFFSIZE

---

## Community Notes

- GLM-4.7 multimodal -- inference does not slow for large context (Festr)
- Kimi K2.5: 8 GPUs minimum; avoid REAP (lobotomized) -- Festr
- Community migrated to Qwen 3.5 after March 1
