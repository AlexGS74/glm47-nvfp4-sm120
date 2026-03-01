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


### Feb 26 (Continued: 8:09 AM – 1:30 PM) — Unsloth Joins; MTP=5 Confirmed at 150 tok/s

**8:09 AM:** Ixtrix asks about video input on vLLM with NVFP4, shares documentation and code example (see recipes file).

**~8:30 AM — Unsloth NVFP4 quants available (Shibe/UnslothAI):**
- Shibe joins and shares: Sehyo/Qwen3.5-122B-A10B-NVFP4 (71B model, 37.3k downloads!), Sehyo/Qwen3.5-35B-A3B-NVFP4, Sehyo/Qwen3.5-397B-A17B-NVFP4
- "Seems like 122B is most popular qwen3.5 model"
- Festr: "have u updated the qwen with mtp layer?" — Shibe: "Not yet, still in Japan lol, until sunday"

**~12:24 PM — MTP=5 confirmed at ~150 tok/s:**
- A docker-compose config with `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":5}'` shows: **~150 tok/s decode at batch 1 on 4x Blackwell (vs ~75 tok/s without MTP)**
- "updated forum" posted

### Feb 27, 2026 — Tool Call PRs, Think Tags Fix, Benchmarks

**1:32 PM — February 27 begins with new technical threads**

**Qu gets abysmal speeds:**
- 500-token speedtest on Qwen3.5-397B-A17B-NVFP4: 12 tok/s at 1 req / 0 context; 164 tok/s at 8 req. Something wrong with their setup.
- Ixtrix: PCIe matters for TP. Comparison: Threadripper PRO 9975WX + 4x Max Q's on proxmox (4 NUMA nodes) shows significantly lower speeds.
- kcramp: "iirc the pcie only matters for loading the models" — Ixtrix: "no it matters for TP"
- Ixtrix posts MTP-3 (fixed numa nodes) vs MTP-5 benchmark tables (side-by-side screenshots)

**No thinking output — critical fix (Ixtrix, ~5:17 PM):**
- darkstar000: "has anyone encountered no thinking output with vllm?"
- Ixtrix: must pass `extra_body["chat_template_kwargs"] = {"enable_thinking": thinking}` — otherwise vLLM does NOT enable thinking by default
- Key note from Ixtrix: "vLLM does NOT enable thinking by default — without this flag the model always outputs `<think>...</think>` tags regardless of intent. Setting enable=True activates reasoning mode (temp 0.6, top_p 0.95). Setting enable=False activates instruct mode (temp 0.7, top_p 0.8)."

**Ixtrix MTP=5 tool calling works:**
- darkstar000: "MTP and tool calling is resolved now?"
- chisleu: "I haven't heard that it was."
- Ixtrix: **"mine works at mtp 5"** — a significant update; Ixtrix's specific setup (Threadripper PRO 9975WX + patched vLLM) gets tool calls working at MTP=5

**Video input confirmed working:**
- Qu: "Just want to confirm that this actually accepts video/image input?"
- Ixtrix: "video works great"

**Ixtrix shares full updated docker run command (~5:20 PM):**
Full alias with custom patches mapped in via -v flags:
- Maps `collective_fusion.py`, `chat_completion/serving.py`, and `qwen3coder_tool_parser.py` from PR files into the cu130-nightly container
- Includes `--media-io-kwargs '{"video": {"num_frames": -1}}'`
- Uses `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`
- "I had to add some custom commands because GPUs don't like my VM setup"
- Related: "Both PR files to add and map: https://github.com/vllm-project/vllm/pull/33088"

**Festr creates new vLLM PR #35615 (~11:08 AM):**
- "vibe coded better patch for the tool calling which was still failing"
- https://github.com/vllm-project/vllm/pull/35615

**Festr posts new PR #35581 (Feb 28, 4:55 AM):**
- https://github.com/vllm-project/vllm/pull/35581

**Chat template discovery (~7:21 PM):**
- Unsloth (Daniel Han) LinkedIn post: "Qwen3.5 is now updated with improved tool-calling & coding performance after we fixed the model's chat template. Improves all variants, no matter quant type."
- Festr: "they literally fixed chat template"
- darkstar000: "qwen updated their chat template 4 days ago and nvidia is 11 days old, giving that one a shot"
- Community investigating whether nvidia's NVFP4 model uses the updated chat template

**AWQ quant discussion:**
- darkstar000: "the AWQ has been more reliable and consistent for me so far lol"
- Festr: "I think we need exact prove for this"
- darkstar000: "the AWQ seems to capture more details when copying a screenshot of a UI for example"
- Festr: "who did the AWQ?" — darkstar000: "QuantTrio"
- darkstar000: "theoretically the nvfp4 should be better, its probably just lack of optimizations"

**gguf discussion:**
- Qu: "Friends don't let friends run ggufs on rtx6k"
- Ixtrix: "If it had as good processing for parallel requests and batching I would use ggufs because they have Q4-Q8 rather than FP4 and FP8 only"
- destroyed: "using that with these gpus is like buying a $12k stove when most dinners you make are hamburger helper or some shit"
- Festr: "vllm / sglang is the only way here"
- darkstar000: "yeah I won't even bother with a gguf anymore"

**EleutherAI LM eval harness (~2:17 PM):**
- Festr shares https://github.com/EleutherAI/lm-evaluation-harness
- "if anybody has time to evaluate this framework for testing — would love to use it for evaluation qwen35 nvfp4 quant we use currently and compare to the FP8"
- fearnworks: "eleuther lm eval harness is a very solid benchmark"
- kcramp: "by my vibes, minimax is better than this — keeps losing context and being confidently incorrect"

### Feb 28, 2026 — Today

**Qu shares YouTube video:**
- "Is Bigger Better? EVERY Qwen 3.5 Local AI Compared - 397B vs 122B v..." (xCreate channel)

**MTP numa node fix confirmed:**
- Ixtrix at 8:45 AM posts "latest run of mtp 3 (with fixed numa nodes) vs mtp 5" — side-by-side benchmark tables showing significantly improved performance with fixed NUMA node configuration

**Benchmark data (MTP 3, fixed NUMA nodes):**

| Scenario | Concurrency | Time | Throughput |
|---|---|---|---|
| Algorithm (2048 tokens) | 1 req | 17.9s | 114.4 tok/s |
| Algorithm (2048 tokens) | 4 req | 25.9s | 316.0 tok/s |
| Debugging (1024 tokens) | 1 req | 8.8s | 116.9 tok/s |
| Debugging (1024 tokens) | 8 req | 16.4s | 498.2 tok/s |
| System Design (1536 tokens) | 1 req | 14.0s | 109.7 tok/s |
| System Design (1536 tokens) | 4 req | 20.3s | 302.2 tok/s |
| Code Review (512 tokens) | 1 req | 4.6s | 111.5 tok/s |
| Mixed Load (4x1024 tokens) | 4 req | 13.2s | 309.8 tok/s |

**Thinking is disabled by default in vLLM (Ixtrix note):**
kcramp: "im not using thinkign at all fyi — only nothink — i saw some benchmark that said think was worse"

**Vision routing discussion (~5:17 PM):**
- Festr: "my claude code is now connected to my GLM5 and I'm trying to solve via transparent routing if the vision can be routed to the qwen3.5 small model"
- kcramp: "I just have it as an MCP for OCR to a different PC but still"
- Festr: "I don't like it as MCP — it eats tokens. I'm vibing fully transparent proxy which detects image → replaces it with placeholder → if the GLM wants to see what's in the image it calls new tools which redirects the image to the vision model"
- Ixtrix: "you could setup a proxy you connect to before GLM5 receives the request, then every request you send can filter the image data and if it's detected it routes it to the image model, then when the response is generated you inject that into the request context"
- Festr: "next glm will have vision, impossible to not have vision in next glm release"

---

## Updated: Key Discoveries & Insights (as of Feb 28, 2026)

### 8. Think Tags Require Explicit Flag (vLLM)

vLLM does NOT enable thinking by default for Qwen3.5. Must pass:

```python
extra_body["chat_template_kwargs"] = {"enable_thinking": thinking}
```

- `enable_thinking=True`: Reasoning mode (temp 0.6, top_p 0.95)
- `enable_thinking=False`: Instruct mode (temp 0.7, top_p 0.8)
- Without this flag: model always outputs `<think>...</think>` tags regardless (BUG-814)

### 9. NUMA Topology Critical for MTP Performance

Ixtrix (Threadripper PRO 9975WX + 4x Max Q's on proxmox) was getting lower speeds than expected. Root cause: 4 NUMA nodes on the VM. Fixing NUMA node configuration dramatically improved throughput — MTP-3 with fixed NUMA shows 109–498 tok/s across scenarios.

### 10. Chat Template Update (Qwen + nvidia)

Both Qwen and nvidia have updated the Qwen3.5 chat template to fix tool-calling:
- Qwen updated 4 days before Feb 27, 2026
- nvidia updated 11 days before Feb 27, 2026
- Unsloth: "Qwen3.5 is now updated with improved tool-calling & coding performance after we fixed the model's chat template. Improves **all variants**, no matter quant type."

### 11. Ixtrix's Tool Calling Approach (MTP=5 Working)

Ixtrix gets tool calling working at MTP=5 by:
1. Mapping custom PR patch files into the container via -v flags (PR #33088 + PR #35615 files)
2. Using the updated qwen3coder_tool_parser.py
3. Updated config.json with correct architectures field
Full config in recipes file under "Recipe 9 — Ixtrix's Full Video + Tool Call Docker Config".

---

## Updated Open Problems (as of Feb 28, 2026)

1. **Tool calls with MTP=5:** Working for Ixtrix (with PR patches + patched container). Still broken for most users on stock cu130-nightly. PRs #35615 and #35581 (Festr) pending merge.
2. **Think tags not outputting:** RESOLVED — must pass `extra_body["chat_template_kwargs"] = {"enable_thinking": True}` in vLLM
3. **AWQ vs NVFP4 quality:** Community anecdotally prefers AWQ for some tasks (UI screenshots). NVFP4 theoretically better but unoptimized. Needs systematic benchmark (EleutherAI lm-eval suggested).
4. **Dynamic spec token count based on server load** — not in vLLM yet
5. **GLM 5 support** — next target; "next glm will have vision, impossible to not have vision in next glm release"
6. **Qu's 12 tok/s mystery** — likely NUMA/PCIe topology issue on their setup
7. **Unsloth NVFP4 with MTP layer** — Shibe building, ETA: Sunday (after Japan trip)
8. **Video routing proxy** — Festr building transparent vision proxy for GLM5 + Qwen3.5 vision model

---

## Updated Resources (as of Feb 28, 2026)

| Resource | Link |
|---|---|
| Working Recipes (full configs) | [qwen35-397b-nvfp4-recipes.md](./qwen35-397b-nvfp4-recipes.md) |
| NVFP4 model (nvidia) | https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4 |
| Community HF model + discussion | https://huggingface.co/vincentzed-hf/Qwen3.5-397B-A17B-NVFP4/discussions/1 |
| vLLM mlp.gate bugfix PR | https://github.com/vllm-project/vllm/pull/35156 |
| vLLM Qwen3.5 tool call fix PR | https://github.com/vllm-project/vllm/pull/35347 |
| Festr's tool call fix PR #35615 | https://github.com/vllm-project/vllm/pull/35615 |
| Festr's fix PR #35581 | https://github.com/vllm-project/vllm/pull/35581 |
| vLLM container patch PR #33088 | https://github.com/vllm-project/vllm/pull/33088 |
| SGLang NVFP4 support PR | https://github.com/sgl-project/sglang/pull/18937 |
| SGLang SM120 setup Gist | https://gist.github.com/catid/87cca824963f17fe7479a0ed26221397 |
| Unsloth Qwen3.5 NVFP4 122B | https://huggingface.co/Sehyo/Qwen3.5-122B-A10B-NVFP4 |
| Unsloth Qwen3.5 NVFP4 35B | https://huggingface.co/Sehyo/Qwen3.5-35B-A3B-NVFP4 |
| Unsloth Qwen3.5 NVFP4 397B | https://huggingface.co/Sehyo/Qwen3.5-397B-A17B-NVFP4 |
| EleutherAI LM Eval Harness | https://github.com/EleutherAI/lm-evaluation-harness |
| Benchmark comparison 122b vs 397b | https://qwen122-vs-397-20260224-1154.surge.sh |
| OpenCode non-streaming PR | https://github.com/anomalyco/opencode/pull/14786 |
| vLLM official Qwen3.5 recipe | https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html |



### Feb 26 — Unsloth NVFP4 Quants Released + MTP=5 at 150 tok/s Confirmed

**Quants available (Sehyo on HuggingFace):**
- Sehyo/Qwen3.5-122B-A10B-NVFP4 (71B VRAM, 37k+ downloads — most popular)
- Sehyo/Qwen3.5-35B-A3B-NVFP4 (4.6k downloads)
- Sehyo/Qwen3.5-397B-A17B-NVFP4 (2.45k downloads)

**MTP=5 at 150 tok/s confirmed by WWM:**
- qwen3_next_mtp with spec_tokens=2 gives a noticeable speed bump.
- 125-140 tok/s for code gen.
- At 100 concurrent sessions: 2400 tok/sec aggregate. 200k context: 100-105 tok/s still solid.

**Festr's complete working vLLM docker command (cu130-nightly + all patches):**
- Full docker alias with collective_fusion patch, tool parser fix, and all recommended flags.
- Key patches volume-mounted from host: collective_fusion.py, serving.py, qwen3coder_tool_parser.py
- Includes --media-io-kwargs for video input and --speculative-config qwen3_next_mtp with num_speculative_tokens=2.
- See recipes file for the full docker command.

**Tool call streaming fix merged:**
- Festr: New PR #35615 created to replace older PR (closed old one), fixing Qwen3Coder tool call parser.
- "[Bugfix] Fix Qwen3Coder tool call streaming with speculative decoding" — fixes broken tool call JSON when using qwen3_coder parser + MTP.
- Also relevant: "[Bugfix] Use 'sum' reduction instead of 'avg' in Async TP reduce-sc..." (PR #33088) for correct multi-GPU reduce.


### Feb 27 AM — orangezed's Clean Recipe + Critical Config Fix

**Key insight: mtp.fc must be added to quantization_config.ignore in config.json:**
```json
"ignore": [...existing entries..., "mtp.fc"]
```
Without this, vLLM tries to load the MTP projection layer as NVFP4-quantized but it is actually BF16, causing a shape mismatch crash.

**orangezed's consolidated MTP=5 recipe (clean python launch):**
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
Result: ~150 tok/s decode (batch 1) on 4x Blackwell, up from ~75 tok/s without MTP.

**vLLM build (orangezed):**
- Base: any recent main (ec8f943db, ~Feb 26 2026).
- Cherry-pick PR #35219 (FlashInfer accuracy fix for Blackwell) and PR #35421 (tool call streaming fix).
- PR #35219 needed for correctness on Blackwell. PR #35421 only needed for MTP>1 + tool calling.

**Ixtrix docker-compose.yml variant (also confirmed working):**
- Uses custom vllm-qwen35-mtp Docker image, NVIDIA_VISIBLE_DEVICES=4,5,6,7 (GPUs 4-7).
- Includes --tool-call-parser qwen3_coder, --reasoning-parser qwen3, --mm-encoder-tp-mode data.
- --speculative-config with qwen3_next_mtp, num_speculative_tokens=5.
- Result: ~150 tok/s decode at batch 1 on 4x Blackwell (vs ~75 tok/s without MTP).

**Video/image input confirmed by Ixtrix at 4:58 PM**: "video works great"
- orangezed confirms images working too. Video not tested by orangezed.

**Thinking/reasoning tag fix:**
- Ixtrix: vLLM does NOT enable thinking by default. Must pass explicitly:
  `extra_body["chat_template_kwargs"] = {"enable_thinking": thinking}`
- Without this flag, model outputs no think tags even though it thinks internally.

**Ixtrix MTP=3 vs MTP=5 benchmark (from fixed NUMA nodes):**
- MTP=3: Algorithm 2048 tokens, 1 req = 17.9s / 114.4 tok/s; 4 req = 25.9s / 316.0 tok/s; Debugging 1024 tok, 1 req = 8.8s / 116.9 tok/s
- MTP=5: Algorithm 2048 tokens, 1 req = 40.5s / 41.7 tok/s; 4 req = 26.2s / 313.0 tok/s; Debugging 1024 tok, 8 req = 8.0s / 641.9 tok/s
- MTP=5 wins at higher concurrency / batched scenarios.

**Shibe still in Japan (confirmed 1:30 PM)**: "Not yet Im still in Japan lol Until sunday" — MTP layer update for Unsloth NVFP4 quants pending.

**kcramp's docker setup shared**: configs posted publicly, search config.json in channel history.

**PR #35581 committed (Feb 28 3:37 AM)**: "Fix Qwen3_5MTP packed_modules_map..." — fixes the packed_modules_mapping in Qwen3_5MTP class where gate_up_proj incorrectly included down_proj instead of gate_proj. Enables the intended fused MLP kernel path, improving throughput by 2-8% depending on batch size and sequence length.


### Feb 27 PM — Performance Comparison; Unsloth Chat Template Fix

**Qu FRAG's abysmal speeds explained (port 1235 config):**
- 500-token speedtest: 0 context 1 req = 12.0 tok/s, 8 req = 164.0 tok/s; 16k context 1 req = 70.6 tok/s, 8 req = 216.6 tok/s
- Root cause: PCIe Gen 3 (not Gen 5). kcramp confirms: "my 50-70 tok/s is with PCIe3 lol — iirc the pcie only matters for loading the models"
- Ixtrix: on Threadripper PRO 9975WX with 4x Max Q's, proxmox host, 4 NUMA nodes on the VM.

**Compare: Ixtrix's good config (port 8000):**
- 500-token speedtest: 0 context 1 req = 59.1 tok/s, 8 req = 568.7 tok/s; 16k context 1 req = 108.8 tok/s, 8 req = 466.6 tok/s

**VLLM_USE_V2_MODEL_RUNNER=1 flag** — Festr asks if anyone tried it. Unclear if beneficial.

**Unsloth chat template fix (7:19-7:37 PM):**
- darkstar000 spots Unsloth LinkedIn post: "Qwen3.5 is now updated with improved tool-calling & coding performance!"
- Unsloth: "Qwen3.5 should now produce better tool-calling & coding outputs after we fixed the model's chat template. Improves all variants, no matter quant type or..."
- Festr: "they literally fixed it, worth try compare their and vanilla"
- darkstar000: "qwen updated their chat template, will give it a shot"
- Qu FRAG: "Unsloth fixes might only apply to their ggufs"

**Festr benchmark comparison table posted (image, 7:31 PM):**
- Qwen vs GLM5 vs Kimi vs "Leader" across GPQA Diamond, HLE, IFBench, AA-LCR, GDPval-AA, CriPr, SciCode, Terminal-Bench Hard, AA-Omniscience, AA-Hallucination Rate.
- Qwen leads on GPQA Diamond (69.2%), IFBench (78.8%), AA-LCR (65.7%), GDPval-AA (35.4%).
- kcramp: "i really dont like how it ignored plan mode sometimes" — Festr: "is it because of the nvfp4?"

**Mixed precision insight (Festr, 7:33 PM):**
> "Quantizing any attn_* is especially sensitive for hybrid architectures, and so leaving them in higher precision works well — maybe worth trying to create mixed precision quant leaving sensitive layers in high precision and nvfp4 only for those which are not that sensitive"

**Qu FRAG: "Friends don't let friends run ggufs on rtx6k"** — community motto established.


### Feb 27 Night — Tool Call Fix Posted; NUMA Topology; Performance Debugging

**darkstar000 at 10:34 PM**: "fixed the vllm thinking, getting 80 token/s with the newer cu130-nightly"

**Tool calls broken with MTP confirmed by darkstar000 at 10:37 PM.**
- Ixtrix at 11:00 PM: "there has been a fix posted for this"
- kcramp: "im not using thinking at all fyi — only nothink. I saw some benchmark that said think was worse"

**NUMA topology affecting performance:**
- orangezed at 10:51 PM: "wow just tried two different computers, both TP=4 with rtx6k and one goes through CPU interconnect (can't do full TP=4 through PCIe host bridge) — it gets 60 tok/s VS the 100-150 when on same PCIe interconnect"
- Confirms: GPU interconnect topology is critical. Direct PCIe connections dramatically outperform CPU-bridged setups.

**orangezed at 11:54 PM**: "what are these patches to collective_fusion, tool_parser, etc?" — still learning the patch system.


### Feb 28 Early Morning — YouTube Review + Kernel Fix

**Qu FRAG at 2:19 AM** shares YouTube video: "Is Bigger Better? EVERY Qwen 3.5 Local AI Compared — 397B vs 122B v..." (xCreate channel).

**PR #35581 merged**: Fix Qwen3_5MTP packed_modules_mapping — 2-8% throughput improvement for MTP models.

**Ixtrix at 8:45 AM** posts MTP=3 vs MTP=5 benchmark (with fixed NUMA nodes):
- MTP=5 wins at higher concurrency. MTP=3 better for single-request latency in some scenarios.
- Mixed Load (4x1024 tokens): MTP=3 = 15.2s / 309.8 tok/s; MTP=5 = 11.4s / 360.8 tok/s.


### Feb 28 AM — PR #35615 + AWQ Quality Debate

**Festr at 11:08 AM**: "vibe coded better patch for the tool calling which was still failing. I have closed the old one, created new PR: https://github.com/vllm-project/vllm/pull/35615"
- chisleu: "I think @Festr is a clawbot running some secret government AGI"

**AWQ quality vs NVFP4 debate:**
- darkstar000: "the AWQ has been more reliable and consistent for me so far lol"
- Festr: "I think we need exact proof for this — doing all the SWE bench verified or whatever replicated tests would be nice"
- darkstar000: "could run some benchmarks against each — the AWQ seems to capture more details when copying a screenshot of a UI"
- AWQ quant done by QuantTrio. Festr: "unsloth does not have something similar to the AWQ? I'm curious why unsloth is not providing nvfp4 quants — it looks like they know what they are doing."
- darkstar000: "yeah I won't even bother with a gguf anymore" (after Qu FRAG's "friends don't let friends run ggufs")
- Shibe at 1:01 PM: "Running gguf on gpu? Wtf" (Shibe returned from Japan)

**Festr at 2:17 PM** shares EleutherAI lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- fearnworks at 4:05 PM: "eleuther lm eval harness is a very solid benchmark"


### Feb 28 PM — Vision Routing Proxy + Claude Code Architecture

**Vision routing discussion (5:17-7:42 PM):**

Festr's goal: route vision requests from Claude Code (running against GLM5) transparently to a smaller vision model (Qwen3.5-27B) without modifying the client.

**Ixtrix approach (shared at 7:30 PM):**
- When conversation exceeds 30 images or hits context window limit, model unloads pictures but injects a table: `| filename | short description | status (loaded|forgotten) |`
- Model can call a tool to pull an image back into context if needed.
- "also its only 1000 tokens per image, at 256k thats not a problem with smart management"

**Festr's Claude Code with split text/vision architecture via claude-code-router (CCR) — posted 7:41 PM:**

Architecture: "split-brain" — text model handles all reasoning, vision model only called on demand.

How it works:
1. User sends message with image (e.g., screenshot from Claude Code Read tool)
2. CCR strips the image, caches it, injects [Image #N] placeholder + analyzeImage tool
3. Text model (Qwen3.5-397B, GLM5, or any OpenAI-compatible model via sglang/vllm) sees placeholder, calls analyzeImage with task + context
4. CCR's streaming interceptor catches the tool call mid-stream, sends image + task to vision model
5. Vision model (e.g., Qwen3.5-27B) returns text description
6. CCR makes follow-up request to text model with description
7. Client receives one seamless streaming response — thinking → answer, no visible round-trip

Status: "this is now my design — just testing it. It is working well so far"

**Context on GLM5 + next steps:**
- Ixtrix: "until GLM5.5 comes out and you need a vision model again"
- Festr: "next glm will have vision, impossible to not have vision in next glm release"
- Festr: "the images are polluting context pretty quickly when working with mcp playwright and some longer debug sessions"


---

## Open Questions / Next Steps (as of Feb 28, 2026 ~7:42 PM)

- PR #35615 (new Festr tool call fix) — needs community testing and merge.
- Unsloth NVFP4 quants with MTP layer — Shibe said "Until Sunday" (now back from Japan, should be available).
- AWQ vs NVFP4 quality benchmark — formal SWE-bench comparison still pending.
- Festr's CCR vision routing proxy — working prototype, community interested.
- mudaG's Qwen3.5-122B FP8 benchmark on 2x Pro 6000 — still pending results.


## Mar 1, 2026 — CCR Vision Routing Architecture; vLLM Patch Dockerfile; Speed Records

### Feb 28, Evening — Model Comparisons and Vision Discussion

Following the CCR/routing work Festr had been developing, broader model comparison discussion begins:

**kcramp**: "by my vibes, minimax is better than this [Qwen 3.5], keeps losing context and being confidently incorrect. Maybe system prompt issue — idk if I can do like a 'if you are not sure, ASK, do not assume'"

**Qu FRAG**: "If only Minimax had native vision" — sparking a thread about MiniMax 3.

**Ixtrix**: "They said MiniMax 3 might [have vision], if I recall their twitter post correctly"

**Ixtrix**: "how hard is it to just add a vision model for routing though?" — foreshadowing the architecture Festr is building.

**Festr** announces his approach:
> "this is exactly what I'm trying to solve - my claude code is now connected to my GLM5 and I'm trying to solve via transparent routing if the vision can be routed to the qwen3.5 small model"

### Mar 1, 00:41 AM — Festr's CCR Split-Brain Architecture Announcement

**Festr posts a comprehensive write-up** of his Claude Code vision routing design:

> **Claude Code with split text/vision architecture via claude-code-router**
>
> Running Claude Code against local/self-hosted models through @musistudio/claude-code-router (CCR). The setup uses a split-brain architecture for image handling:
>
> **Text model (main brain)** — any OpenAI-compatible model (via sglang, vllm, etc.) that handles all reasoning, code generation, tool use. Doesn't need to be multimodal. Images in the conversation are replaced with [Image #N] placeholders.
>
> **Vision model (eyes)** — a smaller multimodal model (e.g. Qwen3.5-27B) running on a separate machine. Only gets called when the text model needs to look at an image.
>
> **How it works:**
> 1. User sends a message with an image (e.g. a screenshot from Claude Code's Read tool)
> 2. CCR strips the image, caches it, injects [Image #N] placeholder + analyzeImage tool
> 3. Text model sees the placeholder, decides it needs to look at the image, calls analyzeImage with a task ("describe the UI layout") and context ("user is asking about a CSS bug")
> 4. CCR's streaming interceptor catches the tool call mid-stream, sends the image + task to the vision model
> 5. Vision model returns a text description
> 6. CCR automatically makes a follow-up request to the text model with the description
> 7. Client receives one seamless streaming response — thinking → answer, no visible round-trip

**Festr**: "this is now my design - just testing it. It is working well so far"

Discussion about vision + image context management followed:

- **kcramp**: Uses an MCP tool calling a different PC for OCR instead of routing
- **Festr**: Prefers transparent proxy over MCP ("it eats tokens and is not universal as it could be")
- **el8**: "I think LiteLLM can go a long way to doing this"
- **Ixtrix**: Shares their own stateful image forgetting architecture — maintains a table of image descriptions injected into context, model can request to "pull back" an image when needed

Festr's concern: "images are polluting context pretty quickly when working with MCP playwright and longer debug sessions — I was always interested how claude code maintains multiturn with images"

Ixtrix on their approach: "when the model has over 30 pictures uploaded or the conversation has gone over a certain context count, it unloads the pictures, injecting [file | short description | stated (loaded|forgotten)] then based on context if it decides it needs the image it can call a tool that pulls it back into main context"

**Festr**: "maybe I should really wait for the DS4 release so I'm not chasing ghosts" — reflecting on whether to wait for DeepSeek 4 instead

### Mar 1, Morning — Unsloth Update + Community Traction

**Marky**: "I'd love to figure this out for opencode/pi" (CCR vision routing)

**Festr**: "I will release mine vibe coded proxy once this design will be proven to work"

**Marky** reports: Tried Qwen3.5-27B on Strix Halo for vision, but "just too slow" — can't co-run with M2.5 NVFP4 on the same GPUs.

**Marky** shares: "Unsloth updated their quants of 35, and improved the chat template" — Unsloth NVFP4 getting better.

**Festr** to Marky: "what was the speed? Im getting 70 tok/sec" — referring to Qwen3.5 on their setup.

### Mar 1, Afternoon — vLLM Patch Dockerfile; Speed Records

**Context:** Qwen 3.5 still requires cherry-picked patches to work reliably in vLLM nightly.

**orangezed** reminds: "Qwen35 requires two PRs cherry picked last I checked, see forum"

**Lavd**: Confirms — `git cherry-pick PR #35219 PR #35421` — but main branch may have changed, introducing new challenges.

**orangezed** shares their full **production Dockerfile** for patched vLLM:

```dockerfile
FROM vllm/vllm-openai:cu130-nightly

# Install patch utility
RUN apt-get update && apt-get install -y --no-install-recommends patch && rm -rf /var/lib/apt/lists/*

# Apply PR #35219 (FlashInfer accuracy fix - zero freed KV cache blocks)
COPY pr35219.patch /tmp/
RUN cd /usr/local/lib/python3.12/dist-packages && patch -p1 < /tmp/pr35219.patch

# Apply PR #35581 (Fix Qwen3_5MTP packed_modules_mapping for gate_up_proj)
COPY pr35581.patch /tmp/
RUN cd /usr/local/lib/python3.12/dist-packages && patch -p1 < /tmp/pr35581.patch

# Apply PR #35615 (Fix Qwen3Coder streaming tool parser for speculative decode)
COPY vllm-fix/tool_parsers/qwen3coder_tool_parser.py /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/qwen3coder_tool_parser.py
COPY vllm-fix/chat_completion/serving.py /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py

# Auto-patch config.json to add mtp.fc to quantization ignore list
COPY patch_config.py /opt/patch_config.py
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENTRYPOINT ["/opt/entrypoint.sh"]
```

**PRs applied:**
- **#35219**: FlashInfer accuracy fix — zero freed KV cache blocks
- **#35581**: Fix Qwen3_5MTP `packed_modules_mapping` for gate_up_proj
- **#35615**: Fix Qwen3Coder streaming tool parser for speculative decode

**chisleu** posts a speed benchmark — Qwen 3.5 benchmarks "profoundly faster" than SGLang GLM 4.7:

```
📊 Success Rate: 3/3 (100.0%)
⚡ Performance Metrics:
  TTFT: 52.37s avg (38.99s min, 77.18s max) — 🔴 Slow (high context queries)
  TPS:  68.4 tok/s avg (68.1 min, 68.5 max) — 🟢 Excellent
  Tokens: 671 generated
Context: ~170k tokens per query
```

**Context window:** Qwen 3.5 max_model_len = 262,144 tokens (confirmed by chisleu and orangezed).
