# GLM-4.7 Quick Start — SM120 (4× RTX PRO 6000 Blackwell)

Assumes everything is already installed (vLLM 0.15.1, models cached, patches applied).
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

---

## Key facts

| | AWQ | NVFP4 |
|-|-----|-------|
| Quant | QuantTrio/GLM-4.7-AWQ (`awq_marlin`) | Salyut1/GLM-4.7-NVFP4 |
| Attention | FlashInfer | TRITON_ATTN (flashinfer broken on SM120 for FP4 MoE) |
| MTP | off (slower at low concurrency) | off (0% acceptance rate) |
| Thinking | renders in Claude Code (model name spoofed to Opus) | same |
| C=1 decode | ~68 tok/s | ~48 tok/s (benchmarked at lower batch size) |
| Tool calls | working | working |
| Served model name | `claude-opus-4-5-20251001` | `claude-opus-4-5-20251001` |
| vLLM port | 30000 | 30000 |
| buster-ripper port | 30001 | 30001 |
