# vLLM Anthropic Prefix-Cache Proxy

Normalizing HTTP proxy that sits between Claude Code and vLLM, applying
cache-stabilizing fixes to Anthropic `/v1/messages` requests before they reach
the model.

## Problem

vLLM V1 prefix caching hashes consecutive token blocks. Any token that changes
at position N invalidates the KV cache for all positions > N. For GLM-4.7,
tools are injected at position 0 of the formatted prompt — before the system
prompt — so a single reordered tool definition causes a full cache miss.

Observed cache busters (identified by audit, 2026-02-21):

| Cache buster | Scope | Fixed here |
|---|---|---|
| MCP tool reordering (reconnects return tools in arbitrary order) | Per-turn | ✅ sort by name |
| `currentDate` injection (`Today's date is YYYY-MM-DD.`) appended to MEMORY.md by the Claude Code framework | Daily | ✅ `--strip-date` |
| `gitStatus` block (branch, recent commits) | Per-conversation | ✗ changes once per session, acceptable |
| `<system-reminder>` CLAUDE.md / skills list | Per-turn if content changes | ✗ framework-level |

## Normalizations

1. **Sort tools by name** — always on. Makes the tool block byte-identical
   across MCP reconnects and different server orderings.

2. **Strip `currentDate`** — opt-in (`--strip-date`). Removes the line
   `Today's date is YYYY-MM-DD.` from user message content before forwarding.
   Omit if your prompts rely on the model knowing the current date.

## Install

No install needed — `uv run` fetches dependencies inline.

```bash
# Dependencies declared in proxy.py with PEP 723 inline metadata:
# fastapi, uvicorn[standard], httpx
```

## Run

```bash
# Default: listen on 127.0.0.1:30001, forward to localhost:30000
bash proxy/serve_proxy.sh

# With date stripping
STRIP_DATE=1 bash proxy/serve_proxy.sh

# Verbose (log every normalization applied)
VERBOSE=1 bash proxy/serve_proxy.sh

# Custom upstream / port
UPSTREAM=http://localhost:30000 PORT=30001 bash proxy/serve_proxy.sh
```

Or directly:
```bash
uv run proxy/proxy.py --help
```

## Configure Claude Code

Update the `glm47` function in `~/.bashrc` to point at the proxy port:

```bash
function glm47() {
  ANTHROPIC_BASE_URL="http://localhost:30001" \   # ← proxy, not vLLM directly
  ANTHROPIC_API_KEY="sk-local" \
  ANTHROPIC_MODEL="glm4.7" \
  ANTHROPIC_DEFAULT_OPUS_MODEL="glm4.7" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="glm4.7" \
  ANTHROPIC_DEFAULT_HAIKU_MODEL="glm4.7" \
  CLAUDE_CODE_SUBAGENT_MODEL="glm4.7" \
  /home/alex/.local/bin/claude "$@"
}
```

## Verify it's working

Run with `VERBOSE=1` and watch for normalization log lines:

```
10:32:01 [proxy] tools reordered: [Bash, Edit, Glob, ...] → [atlassianUserInfo, browser_click, ...]
```

If tools are already in a stable order, nothing is logged (no unnecessary work done).

## Architecture

```
Claude Code  →  proxy (:30001)  →  vLLM (:30000)
                  │
                  ├─ sort tools[] by name
                  ├─ (optional) strip currentDate
                  └─ pass through all other endpoints unchanged
```

Only `/v1/messages` POST requests are modified. All other endpoints
(`/v1/models`, `/health`, etc.) are proxied transparently.
