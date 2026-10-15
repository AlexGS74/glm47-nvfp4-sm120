# Qwen3.8-Flash-Next on 2x RTX PRO 6000 Blackwell (SM120)

Working recipe as of 2026-10-15. Serves `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`
on two 96 GB Blackwell cards via the Local Inference Lab (LIL) "karmic-kraken" vLLM
image, TP2/DCP1 with MTP3 speculation, OpenAI- and Anthropic-compatible endpoints.

This is the same stack that runs daily Claude Code / opencode agent sessions:
~97–98% prefix-cache hit rate, MTP acceptance ~3.0/3 tokens, 1M-token context.

Upstream recipe (image contract, qualification): [rtx6kpro `models/qwen38-flash-next.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/qwen38-flash-next.md)

## Hardware / software

| | |
|---|---|
| GPU | 2x NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, CC 12.0, 96 GiB each) |
| Driver | 595.45.04 (hold: 595.84 has XID 120 GSP panics, 615 startup OOMs) |
| Image | `ghcr.io/local-inference-lab/vllm@sha256:bf2c8f5da6f1de82b345367102b528692231a23eadc1b49c180c82c397c77522` (karmic-kraken-beta-20260922) |
| Checkpoint | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`, revision `qad-step5500-ple1000` |
| Container | `qwen38flash`, host network, port 30000 |

## Quickstart

```bash
# 1. Prefetch the checkpoint revision into your HF cache (one-time, ~99G for main
#    + ~10.5G incremental for the QAD branch; HF blob dedup reuses identical files):
HF_HOME=<your-hf-cache> hf download \
  local-inference-lab/Qwen3.8-Flash-Next-NVFP4 --revision qad-step5500-ple1000

# 2. Point the launcher at your cache and start (detached):
HUB=<your-hf-cache> CACHE_DIR=<writable-dir> \
  ./scripts/run_qwen38.sh up qad-step5500-ple1000

# 3. Watch the first boot (download check + autotune, 10-30 min):
./scripts/run_qwen38.sh --logs
#    ready when the log shows: Application startup complete.

# 4. Test:
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sloppus", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'
```

`--status` / `--stop` manage the container. Env overrides: `GPUS` (default `0,1`),
`TP` (2), `DCP` (1), `PORT` (30000), `MAX_MODEL_LEN` (1048576), `SERVED_MODEL_NAME`
(`sloppus`), `MODE` (`mtp` | `off`), `IMAGE` (digest-pinned default).

## The two gotchas that matter

### 1. QAD revisions need a marlin MTP draft

The QAD checkpoint branches carry **block32-MXFP8 MTP experts**. The profile's
`moe_backend=b12x` draft crashes at speculator load on them
(`b12x is not supported for MXFP8 MoE`). The launcher handles this automatically
when you pass `up <revision>`: it adds an explicit speculative config with a
**marlin** draft backend (trunk stays b12x) and repeats the revision inside the
JSON, because an explicit config bypasses the runtime's draft-revision derivation.
Stock `main` keeps the profile default (NVFP4 draft, b12x) and needs nothing.

### 2. Only one served model name

The KK launcher types `served-model-name` as a single string, so only
`SERVED_MODEL_NAME` (default `sloppus`) is served — canonical-name requests 404.
Alias routing happens at your proxy/gateway, not the engine.

## Configuration notes

- **Context**: `MAX_MODEL_LEN=1048576` (1M). The profile default is 262144; the
  KV pool (~4.6M tokens) supports 4–5 concurrent max-context sessions, or run
  262144 for more headroom.
- **MTP3**: acceptance ~3.0/3 tokens on agent workloads. `MODE=off` disables
  speculation entirely (also sidesteps the QAD draft gotcha).
- **Anthropic endpoint**: the image's `/v1/messages` handler keeps inline system
  messages in place (auto-detects template ordering) and strips Claude Code's
  per-request billing header — both matter for prefix-cache hit rate when pointing
  Claude Code at the server.
- **Single-card trial**: `GPUS=0 TP=1 ./scripts/run_qwen38.sh` works for quick tests.

## Quality / performance reference points

- Aider polyglot whole-edit (225 cases): **91.1 pass@2 / 50.2 pass@1** at 255.6 s/case
  for `qad-step5500-ple1000` — community best for this model family
  (previous best 88.9).
- Live agent serving (2–3 concurrent Claude Code sessions): prefix-cache hit
  97–98%, MTP acceptance ~3.0/3, no engine errors over multi-day runs.
- MLLU-Pro knowledge eval: 90.7% (vs GLM-5.3 full EXL3 91.0, DS4.1-Flash 90.3).

## Why the image is pinned to 0922 and not newer

Newer karmic-kraken betas currently regress:

- `20261004-bdd7e579` — CUDA OOM with native CPU KV offload
- `20261005-e1ccbaf0` — A4 prefill silently disabled on MXFP8-CSF-QAD
- daily builds have shown import breakage (`b12x.loader._csf_source`,
  `CsfScalePlanes`)

`karmic-kraken-beta-20261003-a717f13e` is qualified for the step-5500 checkpoint
(+25% decode with MTP per the lab's qualification) and is the next swap target,
but the 0922 digest above is the config this recipe vouches for.

## Files

- [`scripts/run_qwen38.sh`](../scripts/run_qwen38.sh) — launcher (self-contained
  docker-run wrapper; guards against GPU rival containers)

A frozen compose variant exists locally but is not published: it pins the
`-3221ccac` rebuild of the same tag and depends on the lab's launch yaml; the
wrapper above is the maintained path.
