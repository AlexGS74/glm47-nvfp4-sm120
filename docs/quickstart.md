# GLM-4.7 Quick Start — SM120 (4× RTX PRO 6000 Blackwell)

Assumes everything is already installed (vLLM 0.16.0, models cached, patches applied).
Just the steps to get running.

---

## AWQ (recommended — faster, all features work)

**Terminal 1 — vLLM:**
```bash
cd ~/mllm/glm47-nvfp4-sm120
./scripts/serve_glm47_awq.sh
```

**Terminal 2 — buster-ripper:**
```bash
uv run ~/mllm/buster-ripper/buster_ripper.py \
  --upstream http://localhost:30000 --port 30001 --verbose
```

**Terminal 3 — Claude Code:**
```bash
glm47
```

---

## NVFP4 (native FP4 Tensor Cores, currently slower due to vLLM SW immaturity)

**Terminal 1 — vLLM:**
```bash
cd ~/mllm/glm47-nvfp4-sm120
./scripts/serve_glm47_nvfp4_vllm.sh
```

**Terminal 2 — buster-ripper:**
```bash
uv run ~/mllm/buster-ripper/buster_ripper.py \
  --upstream http://localhost:30000 --port 30001 --verbose
```

**Terminal 3 — Claude Code:**
```bash
glm47
```

---

## Optional flags

| What | How |
|------|-----|
| Disable prompt diffing | `uv run buster_ripper.py ... --dump-dir ""` — omit `--dump-dir` entirely |
| Enable prompt diffing | add `--dump-dir ~/mllm/prompt-diffs-awq` |
| Persist session stats | add `--stats-db ~/mllm/buster-ripper-stats.db` |
| Override model name | `glm47 claude-opus-4-5-20251001` |
| Run benchmark | `cd ~/mllm/glm47-nvfp4-sm120 && SPEC_TOKENS_LABEL=awq-no-mtp ./scripts/bench_serving.sh` |
| Run quality eval | `cd ~/mllm/glm47-nvfp4-sm120 && LABEL=awq ./scripts/eval_quality.sh` |

---

## Quality Evaluation (lm-eval)

Runs HumanEval, MBPP, and GSM8K against a live vLLM server. Uses buster-ripper
in `--eval-mode` on port 30002 (leaves your port 30001 Claude Code proxy untouched).

### Install

```bash
uv tool install "lm-eval[api]" --with transformers
```

### Run (server must already be up)

```bash
cd ~/mllm/glm47-nvfp4-sm120

# NVFP4
LABEL=nvfp4 ./scripts/eval_quality.sh

# AWQ
LABEL=awq ./scripts/eval_quality.sh

# Quick smoke test — 10 samples, math only
LABEL=nvfp4 NUM_SAMPLES=10 TASKS=gsm8k_cot_zeroshot ./scripts/eval_quality.sh
```

Results land in `./evals/<LABEL>/` with per-sample logs.

### Tasks

| Task | Dataset | Metric | What it tests |
|------|---------|--------|---------------|
| `humaneval_instruct` | OpenAI HumanEval (164) | pass@1 | Code generation — write a function from docstring |
| `mbpp_instruct` | Google MBPP (500) | pass@1 | Code generation — practical coding problems |
| `gsm8k_cot_zeroshot` | GSM8K (1319) | exact_match | Math reasoning — grade-school word problems |

### How it works

1. Script starts buster-ripper on **port 30002** with `--eval-mode`
2. `--eval-mode` injects `chat_template_kwargs: {enable_thinking: false}` into every
   `/v1/chat/completions` request — prevents GLM-4.7 from putting the answer in the
   `reasoning` field instead of `content` (where lm-eval reads it)
3. lm-eval runs 16 requests concurrently against the proxy
4. Proxy is killed automatically when the script exits

### Environment variables

| Var | Default | Description |
|-----|---------|-------------|
| `LABEL` | `no-label` | Run label — used for output dir (`evals/<LABEL>/`) |
| `TASKS` | `humaneval_instruct,mbpp_instruct,gsm8k_cot_zeroshot` | Comma-separated lm-eval task names |
| `NUM_SAMPLES` | `100` | Samples per task (0 = full dataset) |
| `NUM_CONCURRENT` | `16` | Parallel API requests to vLLM |
| `EVAL_MAX_TOKENS` | `0` | max_tokens cap (0 = no limit, model default) |
| `SERVER_BASE` | `http://localhost:30000` | vLLM server URL |
| `PROXY_PORT` | `30002` | buster-ripper eval proxy port |

---

## Key facts

| | AWQ | NVFP4 |
|-|-----|-------|
| Quant | QuantTrio/GLM-4.7-AWQ (`awq_marlin`) | Salyut1/GLM-4.7-NVFP4 |
| Attention | FlashInfer | TRITON_ATTN (flashinfer broken on SM120 for FP4 MoE) |
| MTP | off (slower at low concurrency) | off (0% acceptance rate) |
| Thinking | renders in Claude Code (model name spoofed to Opus) | same |
| C=1 decode | ~80 tok/s | ~54 tok/s |
| C=16 decode | ~492 tok/s | ~356 tok/s |
| Tool calls | working | working |
| Served model name | `claude-opus-4-5-20251001` | `claude-opus-4-5-20251001` |
| vLLM port | 30000 | 30000 |
| buster-ripper port | 30001 | 30001 |
| eval proxy port | 30002 | 30002 |
