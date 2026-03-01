# MiniMax-M2.5 — Community Thread Summary

> **Source:** #minimax-m25 channel, RTX6kPRO Discord server
> **Period covered:** February 13 – February 28, 2026
> **Hardware context:** Primarily 2x–8x RTX Pro 6000 (Blackwell SM120) GPUs
> **Working recipes:** See [minimax-m25-recipes.md](./minimax-m25-recipes.md)

---

## TL;DR

MiniMax-M2.5 (456B MoE model) arrived in the server on Feb 13, 2026, and the community quickly got it running. The key story: the official vLLM deploy guide works out of the box, FP8 on 8 GPUs gives ~86 tok/s with kernel tuning, and the NVFP4 REAP variant by lukealonso lets the whole model fit on a single RTX Pro 6000 (96GB). NVFP4 quality benchmark shows 86.2% accuracy vs 85.8% for the original — essentially no degradation from quantization. The main frustration is the SM120 NVFP4 kernel gap (conditioned on SM100), which limits NVFP4 performance, and temperature/reasoning parser quirks.

---

## Timeline of Progress

### Feb 13 — Channel Opens; First Working Recipe

**Eric P Sheets** opens the channel with the first working recipe. The official HuggingFace deploy guide for vLLM works with 4 cards:

```
SAFETENSORS_FAST_GPU=1 vllm serve MiniMaxAI/MiniMax-M2.5 --trust-remote-code --tensor-parallel-size 4 --enable-auto-tool-choice --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think
```

Key observations from Eric:
- Model loading: 53.75 GiB.
- **Temperature 1.0 causes looping.** Recommend 0.7 with repetition_penalty 1.15.
- Power improvement vs M2.1: M2.1 rarely hit 200W/card at 300W limit; M2.5 goes full tilt.


### Feb 13-14 — Initial Testing

**chisleu** downloads and tests immediately:
- **81 tok/s at 20k context, 61 tok/s at 100k context** on 4 GPUs (FP8).
- "I think I got a little spoiled by Qwen 3 Coder Next" — acknowledging the speed difference.
- Note: Official recommendation is temp 1.0, but Eric recommends 0.7 to avoid loops.

Also: **destroyed** (WWM) and **Festr** testing SGLang on NVFP4 — getting ~85-89 tok/s with 2 cards on NVFP4.


### Feb 16-17 — SGLang Deep Dive + CyySky's 8-GPU Setup

**CyySky** (8x GPU, FP8) shares the full SGLang recipe with kernel tuning:

- docker image: `lmsysorg/sglang:dev-cu13`
- SGLang params: `--tp-size 8 --ep-size 8 --mem-fraction-static 0.8 --fp8-gemm-backend triton --moe-runner-backend triton`
- **Kernel tuning** with `tuning_fused_moe_triton.py`: tuning takes 5-10 minutes, results in optimized `E=32,N=1536` config for RTX Pro 6000 Blackwell Max-Q.
- **Result: ~86 tok/s** (FP8, 8x GPU, kernel-tuned).

**CyySky for vLLM** (8 GPUs, nightly):
- Install: `uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cu130`
- Requires copying tuned kernel config file to vLLM's fused_moe/configs/ directory.
- Uses `--enable-expert-parallel` flag for 8-GPU setups.

**chisleu** shares a simpler docker one-liner for 4 GPUs (vllm/vllm-openai:latest).

**luke** notes the SM120 kernel gap:
> "Hopefully NVFP4 inference perf will improve as better kernels for SM120 are merged into flashinfer, right now there's a pretty annoying gap where much of the Blackwell support is conditioned on SM100."

**Festr** adds concern:
> "sm120 is missing some crucial features, I'm curious if we get some proper nvfp4 implementation which will be faster than fp8. From what I'm reading — it might actually never happen as the nvfp4 is just more complex in attention — it has to unpack to fp8 or bf16 anyway."

**luke** clarifies: "It's only missing some relatively minor tensor memory related instructions, you can definitely make some faster kernels that are a combination of the SM90 and SM100 kernels."


### Feb 17 — Reasoning Parser Issues

**chisleu** hits tool call problems with smolagents:
> "Is there a way to put minimax 2.5 into nothink mode? It's breaking smolagent's tool calls."

luke: "make sure you have the minimax reasoning parser enabled."

Discussion of vLLM issue #34625 — reasoning parser bug with MiniMax M2.5. Some users find switching to deepseek_r1 parser works better for tool calls.

**mudaG** later (Feb 20): For the NVFP4 quant with SGLang, use `--reasoning-parser minimax` (not `minimax-append-think`). "append_think was wonky for me — minimax properly separates thinking into reasoning_content field."


### Feb 20 — NVFP4 REAP Quant: Single Card Breakthrough

**luke** announces:
> "FYI I uploaded an NVFP4 quant of the REAP variant of the model — it fits on a single 96GB card"

Model: lukealonso/MiniMax-M2.5-REAP-139B-A10B-NVFP4

**Community reaction:**
- **mudaG**: "I believe im trying out your nvfp4 quant now on dp=2"
- **chisleu**: "Damn minimax m2.5 is a good model"
- **Qu FRAG**: "Could be groundbreaking. If it's not too lobotomized it might be the smartest model that can run on a single 6000 or Spark"
- **Qu FRAG**: "Link please!" → luke shares the HuggingFace URL.

**NVIDIA Developer Forums post** shared by Qu FRAG: MiniMax 2.5 REAP NVFP4 on single DGX Spark:
- pp2048 = 3342.54 tok/s, tg32 = 16.71 tok/s (limited by single card but proves feasibility).


### Feb 21 — NVFP4 Speed Testing

**chisleu** gets NVFP4 running on 2 GPUs:
> "I got the nvfp4 running on 2 gpus at a reasonable clip!! Now I gotta try running something tiny to see if I can get higher throughput out of the other two GPUs"
> **90 tok/s (20k context) ↔ 60 tok/s (100k context)**

"nvfp4 benchmarks well on 2 GPUs vs on 4 GPUs in fp8" — confirming efficiency advantage.

**Qu FRAG** single-card speed test (1x RTX 6000, MAX_MODEL_LEN ~81k):
> "Finally speed tested this. At 64 users it is FAST."
- Test suite: fork of vllm-benchmark-suite by shihanqu simulating agentic tasks with kilo code.
- Issue noted: outputs sometimes incoherent. luke couldn't reproduce.

luke debug: provides curl command to test coherence directly.


### Feb 21 — Code Agent Performance Frustrations

**chisleu**:
> "I don't know if I have the patience to run code summarization agents locally... Minimax-M2.5 takes like 15-20 minutes to finish summarizing DSPy."
> "I wish cline could run MCP servers in the background and get results back async"


### Feb 22 — NVFP4 Coherence Debugging

luke provides direct curl test for NVFP4 coherence. Mysterious issue: some users see incoherent outputs that luke can't reproduce.


### Feb 27 — AWQ Variant Testing; Marky's Setup

**Marky** (AMD 5950x, 32GB RAM, 2x RTX 6000 Pro) shares his AWQ setup:
- Model: mratsim/Minimax-M2.5-BF16-INT4-AWQ
- Using vLLM with `--tensor-parallel-size 2`, numerous FlashInfer flags.
- Performance: ~114 tok/s at low context, ~50 tok/s at 130k+ context.
- Comparison view: "AWQ also a bit dumber" — Qu FRAG on quality tradeoff.

**Markets & Mayhem** shares complete SGLang NVFP4 recipe (CUDA 13, 2x GPU):
- Uses lukealonso/MiniMax-M2.5-NVFP4.
- Flags include: NSA prefill/decode backends (`flashmla_auto`), `--enable-flashinfer-allreduce-fusion`, `--disable-custom-all-reduce`.
- **Environment:** `NCCL_P2P_LEVEL=PHB`, `NCCL_ALLOC_P2P_NET_LL_BUFFERS=1`, `NCCL_MIN_NCHANNELS=8`.
- Kernel tuning takes 5-10 minutes on AMD 5950x w/128GB RAM.

**Marky encounters SGLang crash:**
```
[2026-02-27 22:52:09] Rank 0 scheduler is dead. Exit code: -9
```
- Confirmed: NVFP4 loads in vLLM but unreliable in SGLang for Marky.
- Using SGLang 0.5.9 and sgl_kernel 0.3.21.
- Memory: at gpu_util=0.9, GPUs at 88-89k MiB each.
- Marky: "I can get nvfp4 loaded with vllm, just can't do it with sglang reliably, but the performance hit with nvfp4 is real — the time before it starts thinking feels so much slower too"

**Markets & Mayhem** has SGLang working: "sglang w/cuda 13 works well for me on two rtx 6ks. 12.9 was slower."

Marky vs Markets: CPU and RAM differences (5950x 32GB vs 5950x 128GB) may be the factor.


### Feb 28 — Quality Benchmark: NVFP4 vs Original

**Lavd** posts the definitive quality benchmark comparing:
- **lukealonso/MiniMax-M2.5-NVFP4** vs **MiniMaxAI/MiniMax-M2.5** (original)
- Benchmark: MMLU-style, 12032 questions across 14 categories.

**Result: NVFP4 wins overall — 86.2% vs 85.8% accuracy.** No meaningful degradation from quantization.

Category highlights (NVFP4 / Original):
- Law: 70.7% / 68.4% — NVFP4 clearly better
- Biology: 94.1% / 93.3% — NVFP4 better
- Math: 94.7% / 95.2% — Original slightly better
- Engineering: 81.5% / 81.8% — About equal

Full table in recipes file.

**remichu_sm** (2x RTX 6000 Pro): Confirms 114 tok/s at low context, 50 tok/s at 130k+ with AWQ.

---

## Key Discoveries & Insights

**Performance ceiling on 4x Blackwell:**
- FP8 (4 GPU): 61-81 tok/s depending on context
- NVFP4 (2 GPU): 60-90 tok/s — roughly comparable efficiency per GPU
- NVFP4 (1 GPU): 16.7 tok/s decode (DGX Spark), but excellent throughput (pp2048=3342 tok/s)

**The SM120 NVFP4 gap:** Most Blackwell attention kernel optimizations are coded for SM100. SM120 (RTX Pro 6000) uses a fallback path, which limits NVFP4 gains. This is likely to improve as flashinfer merges better SM120 support.

**Reasoning parser choice matters:**
- vLLM FP8: Use `minimax_m2_append_think` 
- SGLang NVFP4: Use `minimax` (not append_think — "wonky" per mudaG)
- Without any reasoning parser: think tags may bleed into output, breaking tool calls

**Temperature quirk:** Official recommendation is temp 1.0, but this causes output loops at longer contexts. Practical range: 0.7-0.95 with repetition_penalty 1.1-1.15.

**Code agent use:** M2.5 is slow for code summarization tasks (15-20 min for DSPy). Better suited as a high-quality reasoning/chat model than a fast coding agent.

**Expert parallel flag:** Essential for 8-GPU vLLM setups (`--enable-expert-parallel`).

**NVFP4 vs AWQ vs FP8:**
- NVFP4: Best quality (86.2% benchmark), good single/dual GPU efficiency, tricky SGLang setup
- AWQ: Fast at low context (114 tok/s), easier to load, slight quality tradeoff
- FP8 (original): Most reliable, best SGLang support, lower speed per GPU

---

## Notable Quotes

- **Qu FRAG:** "Could be groundbreaking. If it's not too lobotomized it might be the smartest model that can run on a single 6000 or Spark"
- **chisleu:** "Damn minimax m2.5 is a good model"
- **luke:** "There's a pretty annoying gap where much of the Blackwell support is conditioned on SM100"
- **Marky:** "The time before it starts thinking feels so much slower too" (NVFP4 vs FP8 TTFT)
- **Markets & Mayhem:** "sglang w/cuda 13 works well for me on two rtx 6ks. 12.9 was slower."

---

## Open Questions / Next Steps (as of Feb 28, 2026)

- SGLang NVFP4 reliability issues on low-RAM systems (Marky, 32GB RAM crash) — root cause unclear.
- SM120 NVFP4 kernel improvements coming to flashinfer — ETA unknown.
- NVFP4 coherence issues on single card (Qu FRAG) — intermittent and unreproducible.
- Formal benchmark comparing NVFP4 vs AWQ speed/quality tradeoffs at scale.
- MiniMax M3 / next version — no timeline in channel.
