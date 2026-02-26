# Qwen3.5-397B-A17B-NVFP4 — Community Thread Summary

> **Source:** #qwen-35 channel, RTX6kPRO Discord server
> **Period covered:** February 19 – February 26, 2026 (~8:09 AM)
> **Hardware context:** Primarily 4x RTX Pro 6000 (Blackwell SM120) GPUs
> **Working recipes:** See [qwen35-397b-nvfp4-recipes.md](./qwen35-397b-nvfp4-recipes.md)

---

## TL;DR

A small group of researchers cracked running nvidia/Qwen3.5-397B-A17B-NVFP4 on 4x Blackwell GPUs over ~7 days, starting with zero framework support and ending with 275 tok/s peak in production workloads. The journey involved fighting vLLM crashes, a broken config file, a missing quantization patch, and ultimately discovering that MTP speculative decoding is a double-edged sword that breaks tool calls — and that the fix required a new tool call parser name (qwen35_coder) introduced quietly in a vLLM PR.

---

## Timeline of Progress

### Feb 19 — First Blood: SGLang (Festr + CyySky)

- Channel opens. No framework supports NVFP4 on Blackwell natively yet.
- **CyySky** posts the **first working recipe** — SGLang FP8 model on 8 GPUs using lmsysorg/sglang:dev-cu13. Gets **75-125 tok/s** with NEXTN speculative decoding.
- **Festr** builds SGLang from a custom branch (feat/transformers-v5-qwen35-nvfp4 by joninco) and gets the NVFP4 model running at **~85 tok/s**, but notes it is **unstable** (memory access violations).
- Credit to a HuggingFace discussion by vincentzed-hf for the initial config.
- **vLLM** at this point: nobody can get it to work. Multiple people report crashes on every attempt.

### Feb 20-21 — vLLM Attempts Begin; SGLang Stability Problems

- Multiple users attempt vLLM — all crash. Common error: weight assertion failure (size mismatch [512, 4096] vs [512, 2048]).
- **luke** posts the minimal SGLang model card command that works on paper.
- **Festr** identifies SGLang PR #18937 (merged into main) as the required patch. Pull the nightly docker image.
- **chisleu** reports: "Qwen 3.5 isn't supported on vLLM yet. vLLM is the only software I've got running models successfully."
- **Ixtrix** posts working SGLang alias (non-docker) with NEXTN speculative decoding — **stable at ~51 tok/s** on 4x Blackwell.
- **kcramp** shares GitHub Gist: Qwen 3.5 NVFP4 Setup SM120 (https://gist.github.com/catid/87cca824963f17fe7479a0ed26221397)

### Feb 21-22 — Festr's SGLang Recipe Solidifies

- **Festr** posts a comprehensive SGLang docker + full params. Gets **~42 tok/s** stable on 4 GPUs.
- Key env vars: NCCL_P2P_LEVEL=4, SGLANG_DISABLE_DEEP_GEMM=1, NCCL_IB_DISABLE=1, OMP_NUM_THREADS=8
- **Ixtrix** documents two working SGLang aliases (with/without speculative). Key finding: omitting --speculative-draft-model-quantization unquant is **20% faster**.
- **kcramp** gets SGLang working with NEXTN spec decode. ~35 tok/s at 155k context. Refines env to push ~42 tok/s.
- kcramp's system: AMD Threadripper 5975WX 32-core + ASUS WRX80E + 256GB RAM + 4x RTX Pro 6000 MAX-Q on RAID0.

### Feb 22-23 — The Critical vLLM Unlock (aabbccddwasd's Pinned Message)

- **aabbccddwasd** drops the key insight that unlocks vLLM:
  > "success with nvidia's model on vllm, 150tps on coding with mtp3 — just add model.language_model.layers..mlp.gate and mtp.fc to the ignore list and run it normally with vllm"
- The mlp.gate layers are not marked as excluded from quantization in the checkpoint. Hardcoding them as non-quantizable fixes the crash.
- **Festr** pins this message immediately.
- **vLLM PR #35156** (https://github.com/vllm-project/vllm/pull/35156): [BUGFIX][Qwen3.5] Hardcode mlp.gate as not quantizable — the official fix now merged.
- **chisleu** gets vLLM running using vllm/vllm-openai:qwen3_5-cu130 image.
- **darkstar000** achieves **~77 tok/s** on vLLM via Kubernetes deployment (Triton backend).

### Feb 23 — SGLang vs vLLM Discussion; 150 tok/s Rumour Circulates

- Discussion about whether the 150 tok/s number is real. Ixtrix: "SGLang has worked first try on every model — MiniMax, Qwen 3.5, GLM 4.7 etc."
- Qu notes: "Problem with SGLang is terrible high concurrency speeds."
- vLLM PR #35156 confirmed as the path forward. Config.json hack no longer needed once merged.

### Feb 24 — kcramp's SGLang Refinement + Performance Screenshot

- kcramp posts refined SGLang command with NEXTN + cuda graph + chunked prefill + flashinfer_cudnn backend.
- Achieves ~42 tok/s at 155k context. Screenshot shared showing decode logs.
- Festr testing the same — comparable results.

### Feb 25 AM — vLLM cu130-nightly Becomes the Recommended Path

- Festr establishes the vllm/vllm-openai:cu130-nightly image as the go-to.
- Confirmed: no config.json edits needed with this image — just run it.
- Festr tries vllm serve Qwen3.5-397B-A17B-FP8 (FP8 variant) with --speculative-config {"method":"qwen3_next_mtp","num_speculative_tokens":2} — working.
- Ixtrix: "SGLang has worked perfectly, did you try with 4 Blackwells?" — both paths now solid.

### Feb 25 PM — vLLM Performance Explosion

**6:30-7:00 PM:**
- Festr loads via cu130-nightly WITHOUT config.json edit. **64 tok/s baseline** (no MTP).
- Switches to MTP (method: mtp, tokens=1) — **100-105 tok/s**.
- Tries qwen3_next_mtp, tokens=2 — stays 100-105 but more stable.
- At 100 concurrent sessions: **2400 tok/sec aggregate**. At 200k context: still 100-105 tok/s.
- destroyed confirms his docker-compose (cu130-nightly) working "flawlessly" — **98-100 tok/s** normal, **125-140 tok/s** code gen.

**7:14-7:28 PM:**
- destroyed switches to qwen3_next_mtp, tokens=2 — nice speed bump, **125-140 tok/s** code gen.
- Festr tests --decode-context-parallel-size 2 — 1.8M KV cache, but drops to 80-85 tok/s.
- destroyed tests tokens=5 on qwen next mtp — "200+ tps single request code gen"
- Festr: "it has insane instruction following benchmark — almost 80% compared to 67% for glm4.7"
- destroyed: "it's a monster for one-prompt stuff with tool calls"

**7:29-7:33 PM — The 275 tok/s Moment:**
- destroyed posts a screenshot of his vLLM logs showing peak speeds.
- Festr: **"275tok/sec — this is insane"**
- destroyed: "I thought I was seeing things lmao — its fkn cruising writing html and js"
- Festr: "nobody talks about this yet"
- kcramp: "wtf"
- Ixtrix: "this qwen model is literally the best release I could ask for — beyond getting Opus 4.6 locally I don't think it can be beaten in how much it's worth for me"
- Festr: "in production I'm using glm4.7 for web chatbots, this qwen could replace it easily as 250 tokens/sec is INSANE. With the mtp 5 it bumped from 100 to 150 tok/sec"

**7:44-7:53 PM — The Tool Call Problem Discovered:**
- kcramp reports: "now that I'm using spec decode I keep getting tool failures"
- Festr's vLLM output shows IndexError: list index out of range in tool parser.
- kcramp shares a detailed debug screenshot covering 4 root causes:
  1. Using --enable-auto-tool-choice but client sends no tools
  2. Tool-call format mismatch (OpenCode vs vLLM tool parser)
  3. **Speculative decoding + tool calls = malformed/incomplete calls** — remove --speculative-config
  4. Model outputs tool call with missing fields under some prompts
- kcramp: "the answer was disable speculative decoding lol"
- Festr (7:53 PM): "yep, turning off mtp fixed the tool calling for me"

**8:00-8:20 PM — OpenCode Workarounds Investigated:**
- destroyed: "qwen3_next_mtp tool calling in OpenCode doesn't work — but when I switch to mtp it does"
- kcramp: "you get like 3 bad calls in a row but then it figures it out lol"
- kcramp shares OpenCode PR #14786 (https://github.com/anomalyco/opencode/pull/14786): add streaming boolean option
- kcramp merges non-streaming PR but ultimately: "ok I really don't like it without streaming"

**8:36-9:31 PM — MTP Limits Confirmed:**
- darkstar000 asks if MTP is working with cu130-nightly
- chisleu gave up due to **13 second TTFT** on zero context, switched back to older models
- kcramp at 8:48 PM: using it, steaming-off fixes tool calls but "you just see nothing for like 7 minutes, hopefully it didn't delete anything"
- darkstar000 at 8:54 PM: "tool calls worked pretty flawlessly in charm crush with streaming on through litellm — i didnt ever get the MTP to work though"
- chisleu: "tool calls worked for me, but all my benchmarks showed extremely slow prompt processing"
- destroyed at 8:58 PM confirms: **num_speculative_tokens > 1 bricks tool calling** — still getting 170-200 tok/s with qwen3_next_mtp and tokens=1
- chisleu: "I'm literally shaking and crying right now"
- destroyed at 9:01 PM: "follow this guys if you want mtp to work and switch method to qwen3_next_mtp / greater than 1 spec tokens is fkn amazing for single request code gen but tool calling is busted"
- darkstar000: "1 is good enough for me"
- destroyed at 9:08 PM: "this is true but I already miss seeing the consistent 175tps single request minimum"
- darkstar000 at 9:31 PM: "yeah mine just won't start with that config with mtp enabled"

### Feb 26 — The Tool Call Parser Fix

**4:52 AM — New Hope:**
- Festr shares vLLM PR #35347 "Fix Qwen 3.5 tool calling problem" by sunqingn7: https://github.com/vllm-project/vllm/pull/35347
- "I'm gonna to try it — this should fix our problem"

**5:47 AM — First Attempt Fails:**
- Festr: "no, its not fixed" — traceback: IndexError: list index out of range / Error in preprocessing prompt inputs
- Was still running old --tool-call-parser qwen3_coder with num_speculative_tokens=5

**6:04-6:16 AM — The Critical Discovery:**
- Ixtrix: "PR was for 2 so maybe it's only lower numbers"
- Festr tries VLLM_USE_V2_MODEL_RUNNER=1 — not implemented
- Festr at 6:16 AM: **"HA! I have missed that the PR added new parser --tool-call-parser qwen35_coder / I was still using --tool-call-parser qwen3_coder / retesting"**

**6:26 AM — It Works:**
- Festr: "@Ixtrix working! with 2, now trying 5"

**6:34-6:48 AM — Narrowing Token Limits:**
- tokens=5: not working; tokens=3: not working; tokens=2: WORKING; tokens=4: not working
- Ixtrix: "Haven't checked the code change but even numbers are favoured in most things"
- Festr at 6:48 AM: "4 does not work — so only 2 is working"
- Ixtrix: "I mean 2 is still nice at least"

**7:10-7:16 AM — Qwen3.5-122B FP8 Testing Announced:**
- mudaG: "Will be testing Qwen3.5-122B FP8 on 2x PRO 6000 today, any benchmarks y'all wanna see?"
- Festr requests: single batch speeds at 0 context, 40k context, and 100-batch parallel
- mudaG: "Most likely SGLang but if I have time I'll get to both"
- Festr confirms: with speculative decoding and qwen35_coder parser, tool calls work for tokens=1 and 2 but not >2 (in vLLM; SGLang untested for NVFP4)

**8:09 AM — Video Input on vLLM:**
- Ixtrix: "anyone tested video input on vLLM with NVFP4?"
- Shares how to enable: launch vLLM with --media-io-kwargs '{"video": {"num_frames": -1}}'
- API example uses extra_body with mm_processor_kwargs: {"fps": 2, "do_sample_frames": True}
- "This feature is currently supported only in vLLM" (not SGLang)

---

## Key Discoveries & Insights

### 1. The config.json Fix (superseded by vLLM PR #35156)
The mlp.gate and mtp.fc layers aren't marked non-quantizable in the checkpoint. Add to ignore list in both config.json and hf_quant_config.json. With cu130-nightly this is no longer needed.

### 2. MTP Speculative Decoding — Speed vs. Correctness

| Config | Speed | Tool Calls | Use Case |
|---|---|---|---|
| No MTP | ~64 tok/s | Perfect | High concurrency, agentic |
| method: mtp, tokens=1 | ~100 tok/s | Usually OK | General use |
| qwen3_next_mtp, tokens=2 + qwen35_coder parser | ~125-140 tok/s | Working (Feb 26) | Single-user code gen |
| qwen3_next_mtp, tokens=3-5 | 170-275 tok/s | Broken | Benchmarking/demos only |

- "For each spec step it's a 91% chance of acceptance. Doing 4 steps has a compounding failure rate." (destroyed)
- MTP gains vanish under concurrent load — best for single-user setups.

### 3. The Tool Call Parser Fix (Feb 26)
Key: vLLM PR #35347 introduced a new parser name **qwen35_coder** (note: 35, not just qwen3). Using the old qwen3_coder parser causes IndexError with speculative decoding. Using qwen35_coder restores tool call functionality at num_speculative_tokens=2. Tokens 3-5 remain broken regardless of parser.

### 4. MTP Internal Mechanics
- For Qwen3.5-397B specifically, MTP loads a **28B draft model on top** of the main model.
- method: mtp = standard (stable); method: qwen3_next_mtp = enhanced (breaks tool calls above tokens=2)
- num_speculative_tokens=5 gives the draft model 4 tokens per step plus base model's 1.

### 5. NVFP4 Precision Quality
- NVFP4 beats FP8 by 2 points in some evals. Precision drop vs bf16: "supposedly negligible" (kcramp).
- Instruction following: 80% vs 67% for GLM 4.7 — noted as key strength.

### 6. Tool Call Fix Summary (Updated Feb 26)
- **For tool calls + speed:** use --tool-call-parser qwen35_coder with num_speculative_tokens=2 and method=qwen3_next_mtp
- **For max reliability:** disable MTP entirely (remove --speculative-config)
- **OpenCode:** expects tools format, not legacy; ensure it waits for final tool call payload before executing

### 7. Video Input (vLLM only, Feb 26)
Launch vLLM with --media-io-kwargs '{"video": {"num_frames": -1}}'. Configure fps via extra_body in API calls. Currently vLLM-only.

---

## Production Users Summary

| Person | Framework | Speed | Notes |
|---|---|---|---|
| Festr | vLLM cu130-nightly, qwen3_next_mtp tokens=2, qwen35_coder | 125-140 tok/s | Replacing GLM 4.7 web chatbots |
| destroyed | vLLM cu130-nightly, docker-compose, mtp tokens=1 | 98-140 tok/s | 200+ tok/s code gen w/ tokens=5 (no tool calls) |
| darkstar000 | vLLM Kubernetes Triton | ~77 tok/s | k8s production |
| kcramp | SGLang NVFP4 fp8_e5m2 KV | ~42 tok/s | AMD Threadripper 5975WX |
| Ixtrix | SGLang NEXTN speculative | ~51 tok/s | 4x Blackwell, vision tasks |
| chisleu | vLLM 4x Blackwell | ~85 tok/s | Abandoned MTP due to 13s TTFT |

---

## Notable Quotes

> *"I feel more complete today — qwen3 is working. Now the ultimate goal is to run GLM5"* — Festr

> *"275tok/sec — this is insane. Nobody talks about this yet"* — Festr

> *"I thought I was seeing things lmao — its fkn cruising writing html and js"* — destroyed

> *"This qwen model is literally the best release I could ask for — beyond getting Opus 4.6 locally I don't think it can be beaten"* — Ixtrix

> *"the answer was disable speculative decoding lol"* — kcramp

> *"god damnit I need to be doing my taxes right now not dicking around with mtp"* — kcramp

> *"HA! I have missed that the PR added new parser --tool-call-parser qwen35_coder / I was still using --tool-call-parser qwen3_coder"* — Festr (the moment the fix was found)

> *"I'm literally shaking and crying right now"* — chisleu (on inconsistent MTP performance)

> *"greater than 1 spec tokens is fkn amazing for single request code gen but tool calling is busted"* — destroyed

---

## Open Problems

1. ~~MTP + tool calls~~ **PARTIALLY RESOLVED (Feb 26):** Use qwen35_coder parser + max 2 spec tokens
2. Why only tokens=1 and 2 work with qwen35_coder but 3-5 break — likely draft token accumulation issue
3. Dynamic spec token count based on server load — not in vLLM yet
4. GLM 5 support — the community's next target
5. 13s TTFT issue some users see with cu130-nightly (chisleu) — may be hardware config specific
6. Qwen3.5-122B FP8 on 2x PRO 6000 benchmarks — mudaG testing, results pending

---

## Resources

| Resource | Link |
|---|---|
| Working Recipes (full configs) | [qwen35-397b-nvfp4-recipes.md](./qwen35-397b-nvfp4-recipes.md) |
| NVFP4 model (nvidia) | https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4 |
| Community HF model + discussion | https://huggingface.co/vincentzed-hf/Qwen3.5-397B-A17B-NVFP4/discussions/1 |
| vLLM mlp.gate bugfix PR | https://github.com/vllm-project/vllm/pull/35156 |
| vLLM Qwen3.5 tool call fix PR | https://github.com/vllm-project/vllm/pull/35347 |
| SGLang NVFP4 support PR | https://github.com/sgl-project/sglang/pull/18937 |
| SGLang SM120 setup Gist | https://gist.github.com/catid/87cca824963f17fe7479a0ed26221397 |
| Benchmark comparison 122b vs 397b | https://qwen122-vs-397-20260224-1154.surge.sh |
| OpenCode non-streaming PR | https://github.com/anomalyco/opencode/pull/14786 |
| vLLM official Qwen3.5 recipe | https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html |
