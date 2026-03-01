# RTX6kPRO Discord — #general Channel Summary

> **Source:** #general channel, RTX6kPRO Discord server
> **Period covered:** February 12 – February 26, 2026
> **Channel purpose:** General discussion of running large models on RTX Pro 6000 Blackwell (SM120) hardware
> **Model-specific recipes:** See model-specific docs in this repo

---

## TL;DR

The RTX6kPRO community formed on February 12, 2026, started by Festr with a mission to document how to run large open-weight models on RTX 6000 Pro Blackwell (SM120) GPUs. In two weeks the community grew from a handful of members comparing MiniMax M2.1 configs to a 100+ member server covering multiple frontier models, deep hardware topology debugging, and community-built tooling. The dominant themes are: which quantization format runs natively on SM120, how to configure NCCL for best P2P performance across different motherboard/PCIe topologies, and which models are worth running locally vs cloud.

---

## Timeline

### Feb 12, 2026 — Server Launch

- **Festr** creates the server and sets the mission: "I want to build an actual database about how to run models on RTX 6000 PRO with proven best parameters and inference frameworks."
- First members: Alfonso, chisleu, mickg, Eric P Sheets, Ixtrix.
- **Eric P Sheets** shares first working MiniMax M2.1 vLLM config for 4-GPU deployment. Key flags: `--tool-call-parser minimax_m2`, `--reasoning-parser minimax_m2_append_think`, `--enable-auto-tool-choice`, `--tensor-parallel-size 4`, `--enable-expert-parallel`.
- chisleu: "Are you running FP8?" — the MiniMax M2.1 official release is 8-bit.

### Feb 12–13, 2026 — MiniMax M2.1/M2.5 Focus

- **CyySky** shares a full systemd service config for MiniMax M2.1 on 8x Pro 6000 (8 GPUs, tp=8, expert-parallel, port 9504). Gets **70–122 tok/s**, 1,700,000 token KV cache.
- **CyySky** confirms MiniMax M2.5 full-precision FP8 runs on vLLM with 8x Pro 6000. Peak memory: 728GB. KV cache: 1,700,000 tokens.
- Festr: "the int4 quant is >15% worse in coding, so int4 is a no go." FP8 is the only viable path.
- Deep discussion of GLM 5 on SM120: DS3.2 not supported on SM120 in vLLM. "nvfp4 will be the only way for GLM5." SGLang output garbled.

### Feb 14, 2026 — NVFP4 Quality Benchmarks (Festr)

Festr runs benchmark comparison: **Original vs INT4 vs NVFP4**:

| Benchmark | Original | INT4 | NVFP4 |
|---|---|---|---|
| IFEval | 100% | 90% | **100%** |
| MMLU | 100% | 100% | **100%** |
| BBH | 100% | 100% | **100%** |
| GSM8K | 100% | 100% | **100%** |
| HumanEval | 87.5% | 50% | **100%** |
| HellaSwag | 100% | 100% | **100%** |
| ARC | 100% | 100% | **100%** |
| TruthfulQA | 100% | 100% | **100%** |
| WinoGrande | 90% | 90% | 90% |
| Overall | 90.2% | 84.5% | **90.3%** |

NVFP4 matches or exceeds original on all benchmarks. INT4 shows significant degradation on HumanEval (50%). This data explains the community's preference for NVFP4 over INT4.

chisleu: "This Art Agent is badass at prompting." (sharing an AI-generated image as a demo of local inference quality)

### Feb 14, 2026 — vLLM KV Cache FP8 Bug Discussion (luke)

- **luke** (NVIDIA employee, knowledge from public sources only) shares a critical insight: models with KV scales (other than 1.0) produce gibberish output if you turn on FP8 KV caching (`--kv-cache-dtype fp8_e4m3`) without a fix.
- SGLang PR #18904 fixes this. vLLM doesn't have the equivalent fix yet.
- Festr and luke discuss: the fix is important for models with non-1.0 KV scales. Models luke has produced all have either no KV scales or 1.0 KV scales, so they're safe.
- luke: "I work for NVIDIA but 100% of my knowledge here is from public sources." Also: "SM120 is basically SM89 with some little extra bits bolted on."

### Feb 16, 2026 — SM120 Backend Compatibility Table (Stew Tong tweet, shared by CyySky)

From @stewtong on X: "RTX PRO 6000 is SM120 (Blackwell Server Edition) — not SM100 (B200) or SM90 (H100). Default FP8 MoE backends crash."

| Component | Backend | SM120 Status | Notes |
|---|---|---|---|
| FP8 GEMM | DeepGemm | FAIL | RuntimeError: Assertion error |
| FP8 GEMM | CUTLASS | FAIL | Not supported on SM128 |
| FP8 GEMM | Triton | WORKS | No SM gate |
| Attention | FlashInfer | OK | Default, no issues |
| MoE runner | Triton | WORKS | |

Required flags: `--fp8-gemm-backend triton --moe-runner-backend triton`

### Feb 17, 2026 — Model Ecosystem Discussion

- **OpenRouter LLM Leaderboard** (screenshot shared by Festr): MiniMax M2.5 #1 (2.45T tokens, NEW), Kimi K2.5 #2, GLM 5 #3 (NEW), Gemini 3 Flash Preview #4, DeepSeek V3.2 #5, Claude Sonnet 4.5 #6, Claude Opus 4.6 #7, Grok 4.1 Fast #8.
- destroyed: "MiniMax M2.5 is better than Opus 4.5 is crazy. im honestly very impressed with it @ nvfp4 — it's crushing large agent tasks."
- Festr: "I'm considering to create public paid service for the M2.5 inference — cheaper with nvfp4 — I don't see anyone offering it cheap using nvfp4."
- Community consensus: **fp8 is almost lossless, nvfp4 probably 1–2% accuracy drop.**
- Tensor parallel best practices: TP is power-of-2 (2, 4, 8). 6 cards not a natural split. For 8 cards, TP=8 gets bigger KV cache (20M tokens at DCP=8) vs two independent TP=4 instances.
- Grimulkan: "On my table I lose tput on 8 vs 16 cards: 79 tok/s → 64 tok/s. Despite consuming a LOT more power and having more VRAM surface area. **The reason for 16 cards: KV cache goes up to 20M tokens — absurd amounts of prompts for mass processing and 1000+ tok/s aggregate tput.**"
- Kimi K2.5 on 8 RTX 6000 Pro (native int4 experts, Marlin gemm, Triton MLA):
  - 8 cards, TP=8, DCP=8, FP8 KV: 3M tok KV, 68 tok/s
  - 8 cards, TP=8, DCP=1, FP8 KV: 380K tok, 79 tok/s
  - 16 cards, TP=16, DCP=16, FP8 KV: 20M tok, 43 tok/s
  - 16 cards, TP=16, DCP=1, FP8 KV: 1.25M tok, 64 tok/s

### Feb 17, 2026 — Grimulkan Joins; vLLM SM120 Source Build

- **Grimulkan** joins and becomes a key technical contributor. Has an 8-GPU PCIe switch setup.
- Grimulkan recompiled vLLM from nightly cu130 with 2 custom PRs applied. Runs Kimi K2.5 at 70–77 tok/s with specific env vars.
- Key env vars for SM120 performance (Grimulkan's full set):

```
NCCL_CUMEM_ENABLE=0
NCCL_WIN_ENABLE=0
NCCL_P2P_LEVEL=SYS
NCCL_ASYNC_ERROR_HANDLING=1
NCCL_IB_DISABLE=1
NCCL_DEBUG=INFO
TORCH_DISTRIBUTED_DEBUG=INFO
VLLM_SLEEP_WHEN_IDLE=30
VLLM_TEST_FORCE_FP8_MARLIN=1
VLLM_MARLIN_USE_ATOMIC_ADD=1
VLLM_MARLIN_INPUT_DTYPE=fp8
VLLM_VIDEO_FETCH_TIMEOUT=300
VLLM_AUDIO_FETCH_TIMEOUT=60
VLLM_MEDIA_LOADING_THREAD_COUNT=32
LLM_ALLREDUCE_USE_SYMM_MEM=0
ENABLE_SM120=1
```

- "It has to be the NCCL ones or the allreduce one. It made a significant latency difference vs docker defaults."

### Feb 19, 2026 — Qwen 3.5 Arrives; #qwen-35 Channel Opens

- destroyed: "Anyone thinking of trying Qwen3.5-397B-A17B-nvfp4?"
- Festr: "yes, I'm curious how it will stand against GLM-4.7."
- Festr: "but the glm-4.7 with the mtp is generating >100 tokens for single batch — its so fast."
- chisleu: "I want to run Qwen 3.5, but I'm going to have to run it at 4 bit. Waiting on vLLM to catch up."
- darkstar000 joins, came from the Qwen 3.5 thread.
- Festr creates #qwen-35 channel. Community splits to tackle the new model.
- chisleu on MiniMax M2.5: "Having a really good time with MiniMax M2.5 though in the mean time."

### Feb 20, 2026 — Hardware Show & Tell + Community Expands

- **Grimulkan** shares photos of his GPU cluster (open frame rack, 8x Pro 6000 with PCIe switches). "Janky pic from when I was originally building the cluster (you should never do this)." Later cleaner rack shots.
- **darkstar000**: open-frame build with workstation card cooling system, top fans to pull heat out.
- **chisleu**: "This is my build. I don't need the supplemental fan anymore. I learned to accept the Blackwell's heat."
- **mudaG** reveals his setup: 8x Pro 6000 Max-Q in a SuperMicro chassis on 4x PSUs at 120v. Peak power draw last week: **1989W**. Uses Prometheus + Grafana + DCGM for monitoring. SGLang/vLLM pods on k8s + litellm.
- mudaG shares link to SuperMicro AS-4124GS-TNR 4U Server (supports 8x PCIe 4.0 NVIDIA GPU). chisleu: "holy crap $20k for 2TB of RAM."
- **Veratu** joins, shares a custom server health metrics dashboard with GPU status per card (temps, power, memory, PCIe link speeds). Built with btop. chisleu: "I built an MCP server so my Agent can check the GPU temps after sustained GPU use."
- **darkstar000**: "I need better metrics, haven't time to set it up. Are you using Prometheus / Grafana?" mudaG: "Yep."
- Power discussion: 300W vs 600W card variants. Community consensus: "the performance difference between 300W and 600W on these cards is really negligible" (darkstar000). Qu: "Drop to 300w is more than 10% at high concurrency." Reference: https://shihanqu.github.io/Blackwell-Wattage-Performance/

### Feb 20, 2026 — Multi-GPU Topology Deep Dive (Festr + luke + Grimulkan)

The longest and most technical thread in the channel. Key participants: Festr (AMD EPYC Turin, 8 GPUs, PCIe fabric), luke (NVIDIA, topology expert), Grimulkan (8 GPU, PCIe switches).

**The core problem:** Festr has AMD Turin with 8 GPUs on PCIe fabric. Getting only 53–66 tok/s. Grimulkan gets 70–77 tok/s on same vLLM command. Why?

**Key findings:**
- With NCCL_P2P_LEVEL=4 → 55–61 tok/s. With NCCL_P2P_LEVEL=SYS → 61 tok/s (+6 boost).
- With DCP enabled → 31.2 tok/s (significantly worse).
- The AMD EPYC Turin system has two NUMA nodes. GPUs 0–3 are on NUMA 0, GPUs 4–7 on NUMA 1. Cross-NUMA P2P traffic has to cross the CPU QPI/IF interconnect.
- luke: "No one designs systems that rely on the SMP interconnect for the bulk of communication. In fact we avoid it like the plague in datacenter deployments. Schedulers like SLURM are topology-aware and don't place workloads on sides."
- luke: "With 2 PCIe switches connected to the root complex, you'll just do a hairpin turn through the I/O die — still much better than traversing the IF."
- Grimulkan has 4x PCIe switches, 4 cards per switch. Ring comms are excellent at PCIe speeds. **79 tok/s with 8 cards when staying on one NUMA node.**
- Festr: **NCCL_P2P_LEVEL=SYS gives him a boost from 55 to 61 tok/s.**
- With Grimulkan's full env vars added to Festr's setup: **77 tok/s** (previously 55 tok/s).
- **Veratu** (Intel platform experience): "The issue I had was on Intel platform, it simply couldn't go faster than about 60% of spec no matter what I did. As soon as I switched to AMD and the WX I didn't have anymore issues."
- DeepEP (deepseek-ai/DeepEP hybrid-ep branch): An efficient expert-parallel communication library with PCIe topology support. luke interested.
- Grimulkan: "NCCL does NOT do the snake topology above by default. NVIDIA does not prioritize large PCIe systems."
- P2P bandwidth matrix from Festr's Turin system showed all cross-die traffic ~100 GB/s bidirectional (PCIe Gen5 x16). Within switch: full ~1476 GB/s.
- P2P latency matrix showed cross-die latency ~2.5–3.2 µs vs within-switch latency ~2.07 µs.

### Feb 21, 2026 — vLLM Source Build Deep Dive (Grimulkan)

- Grimulkan shares his approach: "nightly cu130 pull, locally recompiled and my 2 PRs applied." For recompiling: `pip install -e . --no-build-isolation` with torch 2.10.x cu130.
- chisleu: "you guys are super heros."
- Deep discussion about SLURM topology awareness, NTB (non-transparent bridging) for cross-NUMA.
- Grimulkan confirms with Grimulkan's env vars and Festr's docker: same speed. "It has to be the NCCL ones."
- chisleu builds **MCP server for GPU temperature monitoring** — Cline can check GPU temps mid-task.

### Feb 21–23, 2026 — Qwen 3.5 Work Splits to #qwen-35

The detailed Qwen 3.5 work moved to the dedicated channel. In #general, occasional cross-references:
- Festr: first impressions on Qwen 3.5 performance (see #qwen-35 for full details)
- Multiple members joining after seeing Qwen 3.5 performance

### Feb 23, 2026 — Server Renamed to RTX6kPRO

Server renamed from earlier name to **RTX6kPRO**. Channel sidebar now shows dedicated channels: `#general`, `#kimi-k25`, `#glm-47`, `#glm-5`, `#minimax-m25`, `#qwen-35`, `#hardware`, `#sglang`, `#vllm`, `#kimi-k25-forum`, `#benchmarking`, voice channel.

### Feb 23–24, 2026 — More Hardware Topology Discussions

Continued P2P bandwidth testing with various NCCL env combinations:
- Festr's Turin system bidirectional bandwidth matrices shared (multiple screenshots).
- Key takeaway: **NCCL_P2P_LEVEL=SYS critical for multi-NUMA AMD systems.**
- Grimulkan confirms Turin vs Genoa difference in PCIe topology.
- Veratu: "Yes our setups are now the same. 4 switches, 4 cards per switch. Just different MB/CPU."
- chisleu and Grimulkan both use SLURM at work. "And making it topology aware was one of the biggest speeds we got in multi-CPU systems."

### Feb 25, 2026 — Qwen 3.5 275 tok/s Moment Ripples Through #general

The 275 tok/s news from #qwen-35 gets referenced in #general.
- mudaG: "Will be testing Qwen3.5-122B FP8 on 2x PRO 6000 today."
- Community celebrating the milestone.

### Feb 26, 2026 — Video Input & Ongoing Development

- Ixtrix shares vLLM video input support via `--media-io-kwargs '{"video": {"num_frames": -1}}'`.
- mudaG testing Qwen 3.5-122B FP8 on 2x Pro 6000 — results pending.
- Tool call parser fix (`qwen35_coder`) announced in #qwen-35.

---

## Community Members & Their Setups

| Member | Hardware | Framework | Notes |
|---|---|---|---|
| Festr | 2x AMD EPYC Turin, 8x RTX Pro 6000 (2 nodes) | vLLM cu130-nightly | Server operator, channel founder |
| Grimulkan | 8x RTX Pro 6000, 4x PCIe switches | vLLM (recompiled) | Multi-GPU topology expert, 2 custom PRs |
| mudaG | 8x RTX Pro 6000 Max-Q, SuperMicro 4U, 4x PSUs | SGLang + k8s + litellm | Work server room |
| chisleu | 4x RTX Pro 6000 (Max-Q) | vLLM | btop+MCP GPU monitoring |
| Veratu | 4x PCIe switches, 4 GPUs each | vLLM/SGLang | Custom server health dashboard |
| darkstar000 | 4x RTX Pro 6000 (open frame) | vLLM (k8s) | Fan cooling mods |
| CyySky | 8x RTX Pro 6000 | SGLang, vLLM | First SGLang FP8 recipe |
| Ixtrix | 4x RTX Pro 6000 | SGLang | Vision tasks, video input |
| kcramp | 4x RTX Pro 6000 MAX-Q | SGLang | AMD Threadripper 5975WX |
| Qu | Multiple Blackwells | Mixed | Power/wattage benchmarking |
| destroyed | 4x RTX Pro 6000 | vLLM (docker-compose) | 275 tok/s record holder |
| luke | (NVIDIA employee) | — | Low-level topology, NCCL, PCIe expertise |

---

## Models Discussed

| Model | Status on SM120 | Framework | Notes |
|---|---|---|---|
| MiniMax M2.1 (FP8) | Working | vLLM | First model discussed, 8-GPU, 70–122 tok/s |
| MiniMax M2.5 (FP8) | Working | vLLM | 70–122 tok/s, 1.7M tok KV on 8 GPUs |
| Kimi K2.5 (INT4+Marlin) | Working | vLLM (recompiled) | 68–79 tok/s on 8 GPUs, Marlin+Triton MLA |
| GLM 4.7 | Working | vLLM cu130-nightly | >100 tok/s with MTP, production use |
| GLM 5 | Blocked | vLLM/SGLang | DS3.2 not supported; nvfp4 only path; SGLang garbled output |
| Qwen 3.5-397B NVFP4 | Working | vLLM cu130-nightly | 275 tok/s peak (see #qwen-35 for full docs) |
| Qwen 3.5-122B FP8 | Testing | SGLang/vLLM | mudaG testing on 2x Pro 6000, results pending |
| Qwen 3 VL | Working | — | chisleu using for vision tasks |
| Qwen 3 Coder Next | Working | — | chisleu: "super powerful coder for its size, easily runs on 1–2 Blackwells" |
| Gemini 3 Flash Preview | Cloud | — | OpenRouter #4 |
| DeepSeek V3.2 | Untested | — | OpenRouter #5 |

---

## Key Technical Insights

### NCCL Configuration for SM120 Multi-GPU

The most-discussed and most-impactful configuration finding. Key env vars in priority order:

1. `NCCL_P2P_LEVEL=SYS` — forces all P2P, critical on AMD multi-NUMA (55 → 61 tok/s)
2. `NCCL_CUMEM_ENABLE=0` and `NCCL_WIN_ENABLE=0` — disables Windows memory features
3. `NCCL_IB_DISABLE=1` — disable InfiniBand unless you have it
4. `ENABLE_SM120=1` — enables SM120-specific kernels
5. `VLLM_TEST_FORCE_FP8_MARLIN=1`, `VLLM_MARLIN_USE_ATOMIC_ADD=1`, `VLLM_MARLIN_INPUT_DTYPE=fp8` — Marlin FP8 gemm path

### PCIe Topology Tradeoffs

- **Single PCIe switch per 4 GPUs (Grimulkan style):** Best intra-group performance, but inter-group traffic goes through CPU root complex
- **AMD multi-NUMA (Festr style):** Cross-NUMA GPU communication adds QPI/IF latency. NCCL_P2P_LEVEL=SYS essential
- **8 GPUs in one NUMA (ideal):** No multi-NUMA penalty. Ring comms work well at PCIe speeds
- **DCP (Decode Context Parallel):** Halves tok/s for single requests but enables 20M token KV cache for mass processing

### Quantization Format Guide

Based on community testing (Festr's Feb 14 benchmarks and general discussion):

- **FP8:** Near-lossless (99%+ benchmark parity). Native SM120 support via Triton backend. Best for 8-GPU systems with full VRAM
- **NVFP4:** 1–2% accuracy drop max. Only viable format for 4-GPU MoE models (fits where FP8 doesn't). Matches original on most benchmarks per Festr's data
- **INT4/Marlin:** 15%+ degradation on HumanEval. Only use for models where FP8/NVFP4 not available (e.g., Kimi K2.5 native int4 experts)
- **BF16:** Not feasible for large models on 4 GPUs (too large). FP8 is the sweet spot

### SM120-Specific Backend Requirements

From Stew Tong's writeup and community testing:
- FP8 GEMM: Triton only (DeepGemm and CUTLASS fail on SM120)
- Attention: FlashInfer works (default)
- MoE runner: Triton works
- Required flags: `--attention-backend FLASHINFER --moe-runner-backend flashinfer_cutlass --fp4-gemm-backend flashinfer_cudnn`

### 8-GPU vs 4-GPU Tradeoffs

Grimulkan's data: 8 GPUs → 64 tok/s single request vs 4 GPUs → 79 tok/s. **Why use 8?**
- KV cache grows from 3M tokens (8-GPU, DCP=8) to 20M tokens
- Aggregate throughput for batch processing: 1000+ tok/s
- For mass prompt processing / high-concurrency: 8+ GPUs wins

---

## Community-Built Tools

| Tool | Creator | Description |
|---|---|---|
| Server Health Metrics dashboard | Veratu | Custom btop-based GPU monitoring per card (temps, power, memory, PCIe) |
| GPU MCP Server | chisleu | MCP server for AI agents to check GPU temps mid-task |
| Prometheus + Grafana + DCGM | mudaG | Full observability stack with SGLang/vLLM metrics dashboards |
| Custom vLLM build with SM120 PRs | Grimulkan | pip install -e with 2 SM120-specific PRs + full env var config |

---

## Notable Quotes

> *"I want to build an actual database about how to run models on RTX 6000 PRO with proven best parameters and inference frameworks."* — Festr (channel founding message)

> *"you guys are super heros"* — chisleu (on Grimulkan recompiling vLLM with SM120 PRs)

> *"No one designs systems that rely on the SMP interconnect for the bulk of communication. In fact we avoid it like the plague in datacenter deployments."* — luke (NVIDIA)

> *"The only other person I've met online who has squeezed 16 GPUs into 1 machine is @Veratu. It is not common."* — Grimulkan

> *"I'm going to ride my 4xR6000s until the fans die. This is plenty of compute for what I need."* — chisleu

> *"god damnit I need to be doing my taxes right now not dicking around with mtp"* — kcramp

> *"NVFP4 probably 1–2% accuracy drop. fp8 is almost lossless"* — community consensus

> *"the performance difference between 300W and 600W on these cards is really negligible"* — darkstar000

> *"I smell there will be nothing for next 12 months"* — Festr (on the 8x Blackwell build market)

---

## Open Questions & Active Threads

1. **GLM 5 on SM120** — DS3.2 not supported; nvfp4 path requires framework support not yet available
2. **Qwen 3.5-122B FP8 benchmarks** — mudaG testing on 2x Pro 6000, results pending
3. **Optimal NCCL config for AMD Turin 2-node setup** — still investigating best multi-NUMA settings
4. **Grimulkan's SM120 vLLM PRs** — potential to upstream to main vLLM
5. **DeepEP hybrid-ep PCIe support** — luke mentioned wanting to try for non-InfiniBand topologies
6. **Video inference benchmarks** — Ixtrix testing vLLM video input on NVFP4 models

---

## Resources

| Resource | Link |
|---|---|
| Qwen 3.5 NVFP4 recipes | [qwen35-397b-nvfp4-recipes.md](./qwen35-397b-nvfp4-recipes.md) |
| Qwen 3.5 thread summary | [qwen35-397b-thread-summary.md](./qwen35-397b-thread-summary.md) |
| SM120 Backend Compatibility (Stew Tong) | https://x.com/stewtong/status/2023519911910535521?s=20 |
| Blackwell Wattage Performance | https://shihanqu.github.io/Blackwell-Wattage-Performance/ |
| chisleu's btop gist | https://gist.github.com/chisleu/3beb0bf03764e8b75c8e118ec1c9a7df |
| Grimulkan's level1techs setup thread | https://forum.level1techs.com/t/wip-blackwell-rtx-6000-pro-max-q-quickie-setup-guide-on-ubuntu-24-04-lts-25-04/230521 |
| convergence.ninja Max-Q build guide | https://convergence.ninja/post/blogs/000021-ToasterOnline.md |
| DeepEP hybrid-ep (PCIe) | https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep |
| SuperMicro AS-4124GS-TNR (8x GPU) | https://www.theserverstore.com/supermicro-as-4124gs-tnr-4u-server.html |


---

## Feb 26, 2026 (continued) — New Members, Channel Expansion

Following Festr's creation of #sglang and #vllm dedicated channels (responding to mudaG's suggestion at 12:55 PM), the afternoon brought a stream of new members.

**Server growth:** chisleu posted the Discord link on r/BlackwellPerformance subreddit, triggering a flood of joins (including scammer bots, promptly banned). Notable new members:
- **someguyinthesky**: Single RTX Pro 6000 + 9950X3D + 256GB DDR5-6000. "Y'all are really making me want to get another 6000 pro."
- **t3rmina1**: Single GPU, only 512GB DDR4 2400 to pair.
- **mtone**: Dual PCIe 5.0 x8 (7950X3D) setup. Notes nvtop bandwidth hits 25GB/s in rare heavy concurrent requests with large prompts. Recommends TR/EPYC for 4 cards or RAM offload.

Community reaches critical mass. MOGUL suggests a #welcome channel to keep #general clean.

---

## Feb 27, 2026 — GPU Market Discussion; SM120 MXFP4; Model Recommendations

### NVIDIA Rubin and NVLink Frustration

**awgross** joins and becomes a key technical voice. Initial topic: NVIDIA's GPU roadmap.
- "There's going to be an even bigger divergence between general compute GPU and AI GPUs based on how Rubin is shaping up."
- **Festr**: "they really should get back the nvlink for prosumers — those fairytails that they wanted to add more cuda instead of nvlink.pffff"
- **awgross**: "Warp MMA instruction — The segmenting of the SM100 vs 120 is really annoying... Leaves RTX 6000 as this weird stepchild. P2P yes but lacking other instructions. Going to have to learn to make my own to get MXFP4 working properly."

awgross notes SM120 MXFP4 kernel status:
> "Full speed PCIe5 is ok for training but don't really matter for inference"
> "does anyone having working MXFP4 kernels for GPT-OSS on SM120? I've had a hell of a time with vLLM and SGLang due to the Blackwell SM confusion issues"

**Festr** on PCIe performance vs model size: "it depends. Kimi 2.5 in prefill pushes 25GB/sec (50GB total) when doing prefill on 8 cards — but decoding 800MB/sec per card for a SINGLE request."

### Procurement Discussion

**blue** (building 8x RTX Pro 6000 in Canada): Asked about sourcing. Newegg works but 1 at a time. Community recommends Markets & Mayhem's supplier link.

**ploopmaster**: Trying to get educational discount from NVIDIA — "giving me the run around."

### Model Recommendations for 1-4 GPU Users (Festr)

**Hurricane** asks: "I currently only have 1 RTX Pro 6000 but can maybe upgrade to 2 or even 4. What model and quantization are you recommending for 1x, 2x, 3x... RTX Pro 6000? I need 128k tokens context and at least 5 concurrent requests."

**Festr's comprehensive response:**
- 2 cards: MiniMax M2.5 with 100+ concurrent requests
- 4 cards: Qwen 3.5-385B with at least 100 concurrent calls (not all in 128k context, but 5 for sure)
- Also: GLM 4.7
- Summary: "currently qwen35, minimax and glm47 are worth running"

**Hurricane** follows up: "Is that smarter than Qwen3-Coder-Next 80B Q6_K_XL?" Festr: "absolutely" — points to architecture description for nvidia/Qwen3.5-397B-A17B-NVFP4.

Hurricane reports: "Currently I'm running this in 1 card with awesome speed with 9x 128K context."

**Festr**: "On 1 card I think that the Qwen3.5 122B-10A (or whatever it is) in NVFP4 will supersede it."

---

## Feb 28, 2026 — AWQ vs NVFP4; System Crashes; Evaluation Tools; AI Coding Agents

### AWQ vs NVFP4 Activation Width Debate

**orangezed**: "How do people feel about 16 bit vs 8 bit activations? AWQ vs NVFP4 experts?"

**chisleu**: "I love my tokens.... tokens go brrrr" (preferring FP8 for more tokens)

**orangezed**: "so you prefer 8 bit activations?"

**Festr**: "When I tested AWQ vs NVFP4 — the later wins. But it's slower in single batch. I was testing MiniMax — but I might have flawed the testing, I let it evaluate Claude Code by crafting some basic tests."

Community consensus solidifies: NVFP4 wins on quality, but AWQ may have a throughput advantage in some single-batch scenarios.

### chisleu's System Crash + Ansible Migration

**chisleu** reports a serious system crash:
> "My system crashed hard last night. It would boot up but about 5 seconds after the console login would appear, the screen would go black and the system would be unresponsive."
> "I hope it wasn't malware, but I nuked the whole system. Finally getting around to playing with Ansible (which is really cool BTW). I'm going to use Ansible to make rebuilding the system and getting models running easier since this platform has had 2 hard system crashes now..."

chisleu shares Ansible playbook output (NVIDIA CUDA 13.0 + driver 590 installation). "Livin on the edge!"

**Festr** on the crash: "nah, that's too paranoid. it will be your RAM issue"

**chisleu**: "AMD Threadripper Pro 7995WX (96 cores of pure computational fury)" — confirms their system spec. Later analysis: BIOS reset fixed first crash; second crash suspected RAM issue, not malware.

### Evaluation Framework Discussion

**Festr** shares EleutherAI/lm-evaluation-harness: "so we can use this framework for evaluating various quants."

**fearnworks**: "EleutherAI lm-eval harness is a very solid benchmark."

**Eric P Sheets**: "Which benchmarks/evals do you typically use?" — sparking discussion about MMLU-Pro and other community eval scripts (connected to activity in #qwen-35 and #minimax-m25).

### AI Coding Agents: Claude Code Takes Over

A significant shift in the community conversation — multiple members discussing how AI coding agents (Claude Code specifically) are replacing manual infrastructure work.

**Ixtrix**: "I can't believe I manually setup 2 VMs when I could just let Claude SSH in and do it for me..."

**destroyed**: "lol been really hitting me today how it's been replacing everything I normally do. felt like yesterday laughing at scripts pasted from gpt3.5"

**Festr**: "I have exactly same, I'm just doing some manual stuff / preparing dockers / virtuals realising I'm just losing time as the claude code do it instantly and more precisely"

**Ixtrix**: "yeah it just setup a SearXNG, Crawl4AI, and Browser-Use isolated VM for my MCP stack"

**destroyed**: "it's really the next batch of models after opus 4.6 has been out that we need — thing is just on a diff tier than anything else"

**Ixtrix**: "I think the coding ability will trickle down into smaller models, but the harness / orchestration layer is where the closed source models will shine"

---

## Mar 1, 2026 — Browser-Use with Local Models; MTP Stability; Secondary Market Scams

### Browser-Use: Qwen 3.5 397B as Driver Model

Discussion of using self-hosted models for browser automation.

**0ftf**: "What model are you using to drive your browser use?"

**Ixtrix**: "Qwen 3.5 397B" (primary recommendation for browser-use tasks)

**0ftf**: "Have you tried smaller models? Hoping to get browser-use working on a single 6k pro."

**destroyed**: "You should try Qwen 3.5 35b or the 122b whatever you can fit. From what I see they should work based on benchmarks — the 397b crushes browser MCP usage currently for me."

**Ixtrix**: "I was going to use the 122b and have my other 2 GPUs for OCR/TTS but I still need to build eval sets to see if 122b will work or not for my use case. The OCR Bench score, IF, long context just make the 397b too good not to use."

**fearnworks**: "For the grounding thing, I've found sam3 as a tool to be REALLY effective — they added text conditioning for agent interaction."

### MTP Speculative Decoding Stability (Festr)

**chisleu**: "I'm waiting for the patches to get merged to fix tool calling, spec decoding, etc."

**Festr**: "You do not need to — using mtp with :1 is still almost same gain and completely stable. And there is still an unresolved bug when it is >2 anyway (stability issue) so I would not go above 2 (like :2 max). Which is recommended on the model page anyway."

**chisleu**: "I know. But I'm really satisfied with myself for getting GLM 4.7 with 200k context stood up with SGLang as a systemd service. I'm waiting for spec decode 5 to switch to Qwen 3.5."

### Secondary Market: RTX 6000 Pro Pricing and Scams

As hype grows, RTX 6000 Pro secondary market becomes active — with associated scams.

**Sicario#2055**: "I saw an RTX 6000 on eBay in the $5500 range."

**chisleu**: "I imagine that lots of people that don't really need one are going to buy one and sell it shortly after. There is so much hype — and openclaw made it worse. Keeping an eye out on eBay and such will probably yield some savings for those inclined to 2nd hand markets."

**Festr** on the $5500 listing: "not bad price, but it's underpriced — I'm curious why they're selling for this price. It literally cost 10k."

**someguyinthesky**: "I'd be careful, I see some listings for 6000 pro on eBay for $3000. Obvious scam."

**root-754B-NVFP4.safetensors** (a member with an amusing username): "Scammers buy or steal old or abandoned eBay accounts that look legit and have some history before launching a bunch of high value items."

Community advice: Use eBay's buyer protection. Never wire transfer. Verify seller feedback history.
