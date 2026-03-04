# GLM-5 NVFP4 Channel Summary
*#glm-5 | RTX6kPRO Discord | Feb 16 - Mar 4, 2026*

See also: [GLM-5 NVFP4 Recipes](./glm5-nvfp4-recipes.md)

---

## Feb 16-17, 2026 - Channel Launch: First Working Config

Channel opens with Festr posting lukealonso/GLM-5-NVFP4 on HuggingFace.

**Hardware:** 440GB VRAM min, 8x RTX 6000 Pro required. fp8_e4m3 KV cache garbled - use bf16.

**NCCL:** Add iommu=pt (amd_iommu=pt on AMD) to kernel command line for P2P hang fix.

**luke (lukealonso):** Self-compiled sglang, abandoned vLLM (keeps breaking NVFP4). Shared sampler patch.

---

## Feb 22-23, 2026 - Community Interest, Size Limits

- chisleu: "Everyone is raving about it"
- root-754B: NVFP4 too big for 4x RTX 6k
- JTazz: 35 tok/s bf16; quality "pretty bad ngl"
- Festr: "I don't like AWQ compared to nvfp4"
- Min 8x cards (441GB too large for TP4)

---

## Feb 24, 2026 - SGLang DCP, Sparse Attention, Frameworks

**Driven (GPU):** sm120 cannot run DeepGEMM (sm90/sm100 only). Working on sm89 MMA kernel for sparse attention.

**Grimulkan:** FlashMLA sparse compiled but too slow on sm120. Plans lightning indexer for dense prefill compatibility.

**Festr:** GLM-5 on SGLang at 35tok/sec nvfp4, fp8 KV garbled, bf16 only.

**SGLang DCP (antgroup fork):** Export SGLANG_DCP=8 - works with all parallelism types.

**Framework comparison:**
- Festr: vLLM better throughput in high concurrency, TTFT unbeatable vs SGLang
- Driven: sm120 probably better with trtllm
- Grimulkan: vLLM faster via marlin/triton; nvfp4 PTQ model too much tradeoff

---

## Feb 26-28, 2026 - Docker Recipe Pinned, MTP Planning, Power

**Alex (8x RTX 6000 PRO WS, 768GB):** Got GLM5 working via Festr's Docker. NVFP4 crashes in complex coding tasks (Kilo Code). "CUDA error: device-side assert triggered". Stable at 1-2 concurrency.

**Festr's pinned Docker setup (Feb 28):**

```bash
docker pull lmsysorg/sglang:dev-cu13
docker run -it --rm --cpuset-cpus "0-63" -v /root/.cache/huggingface:/root/.cache/huggingface -v /mnt:/mnt/ --ipc=host --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all --network host lmsysorg/sglang:dev-cu13 bash
pip install --upgrade transformers
```

**CRITICAL: MTP weights warning (vLLM PR #35548):**
lukealonso/GLM-5-NVFP4 does NOT include MTP weights. Serving with MTP loads garbage (torch.empty()), zero acceptance rates. Use festr2/GLM-5-NVFP4-MTP.

**Feb 28 benchmark (no MTP):** 0 context = 44tok/s, 15k context = 30tok/s

**Power:** 8 cards at 400W each in prefill, 4000tok/s prefill throughput. Peak 600-640W per card.

**Community:**
- Festr: "k2.5 is strong no doubt but GLM models has something which always delivered"
- Festr: REAP "lobotomised too much unless someone rips off carefully all chinese stuff"
- Shibe: "Nvfp4 is so close to fp8 performance anyway so not worth imo"
- Festr: "fp8 is faster at least for single batch"

---

## Mar 1, 2026 - MTP+EAGLE Breakthrough: ~100 tok/sec

**Festr's breakthrough (5:15 PM):**
"@Grimulkan FUUUUUUUUCK - I just added MTP layer and enabled the EAGLE. this is speed 0 context. this is absolutely insane. jump from 32tok/sec. like double the speed."

200k context: 39-57 tok/s.

**Complete MTP+EAGLE recipe (5:52 PM, model: festr2/GLM-5-NVFP4-MTP):**
See [GLM-5 NVFP4 Recipes](./glm5-nvfp4-recipes.md) for full commands.

**Coding speed (7:14 PM):** 95-99 tok/s fast requests; 77-92 tok/s sustained.

**Architecture analysis (Grimulkan):**
SGLang runs GLM-5 as DeepSeek V3.1 model with nvfp4:
1. FA2 in prefill ignoring MLA - "backwards compatible" via lightning indexer
2. sm120 stuck on FA2 sm89 - missing TMEM/tcgen05 and FA4
3. MTP head is bf16 (same as DSv3.2) - why EAGLE works so well
4. FA4 PR #19428 in SGLang (bf16 for sm120)
5. nvfp4 moe gemm should work in GLM5 vllm (DCP compatible) but needs wiring
Grimulkan: "I can compute prefill dense for compatibility. Just need something to test sparse mla decode in vllm."

**claude-relay proxy (Festr's tool):**
Stateless OpenAI-compatible proxy, no API key, proper token counting, vision via MCP playwright.
"I have now >3 hours session with the GLM5 - rock solid stable and very capable, I'm slightly excited"

---

## Mar 2-3, 2026 - MMLU-Pro Results, AWQ, Stability

**MMLU-Pro accuracy (Festr, Mar 3):**
- NVFP4: 87.3% vs benchmark 87.7% - gap only -0.4%
- NVFP4 wins on: formal_logic +3.2%, hs_math +2.3%, hs_physics +2.0%, sociology +2.0%

**AWQ:** Alex OOM during weights loading. Festr: "did not tried, nvfp4 is superior to awq"

**vLLM still broken for GLM-5** - SGLang only.

**orangezed NUMA/PCIe question (Mar 3):**
Festr: "I dont have pcie switch and performance long context is same. The pcieswitch has 10-15tok advantage with single batch but fades at higher context or concurrency. On turin - almost identical to PCIE switch."

---

## Mar 4, 2026 - Latest Recipe

Festr's complete docker+sglang with --max-running-requests 64 and --cuda-graph-max-bs 32. See Recipes file.

Final performance: ~100 TOK/SEC with MTP+EAGLE, 60-80 tok/sec long context, 32 tok/sec at 150k context with Grimulkan DCA patches.

---

## Key People

| Person | Role |
|---|---|
| Festr | Primary - configs, quants (festr2/GLM-5-NVFP4-MTP), EAGLE recipe |
| luke (lukealonso) | Original GLM-5-NVFP4 quant, sglang patches |
| Grimulkan | Architecture analysis, DCA long-context patches |
| Driven (GPU) | SGLang DCP, sm89 sparse kernel |
| Alex | 8x RTX 6000 PRO WS user, stability testing |
| root-754B | vLLM testing, MTP weights PR #35548 |
| orangezed | Long context / NUMA / PCIe questions |

---

## Performance Summary

| Config | Context | Speed |
|---|---|---|
| NVFP4 bf16 KV, no MTP | 0 | ~50 tok/s |
| NVFP4 bf16 KV, no MTP | 15k | ~30 tok/s |
| NVFP4 + MTP + EAGLE | 0 | ~100 tok/s |
| NVFP4 + MTP + EAGLE | 100k | ~double baseline |
| NVFP4 + MTP + EAGLE | 200k | 39-57 tok/s |
| NVFP4 + MTP + EAGLE + DCA | 150k | 32 tok/s |
| NVFP4 + MTP + EAGLE | long | 60-80 tok/s |

**MMLU-Pro:** NVFP4 = 87.3% vs benchmark 87.7% (gap: -0.4%)
