#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "uvicorn[standard]",
#   "httpx",
# ]
# ///
"""
Anthropic /v1/messages normalizing proxy for vLLM prefix-cache stability.

Normalizations applied before forwarding to vLLM:

  1. Sort tool definitions alphabetically by name.
     GLM-4.7's chat template injects tools at position 0 of the formatted
     prompt (before the system message). Any ordering change invalidates the
     entire KV cache. MCP servers reconnect and return tools in arbitrary
     order — sorting makes the tool block byte-identical across turns.

  2. (Optional, --strip-date) Remove the Claude Code framework-injected
     "Today's date is YYYY-MM-DD." line from user message content.
     This is appended to MEMORY.md content inside <system-reminder> blocks
     and changes daily, busting the cache at midnight. Pass --strip-date to
     strip it; omit to leave it (model may use date awareness).

  3. (Optional, --dump-dir PATH) Session-aware prompt diffing.
     Saves the full normalized request body for turn 0 of each session, then
     a unified diff vs the previous turn for all subsequent turns.  Useful for
     identifying remaining cache busters after normalization.

     Session ID = first 12 hex chars of SHA-256(system_prompt_text).
     Stable within a Claude Code session; changes between sessions.

     Output layout:
       PATH/<session_id>/turn_000.json   — full body (turn 0)
       PATH/<session_id>/turn_001.diff   — unified diff vs turn 000
       PATH/<session_id>/turn_002.diff   — unified diff vs turn 001
       ...
       PATH/<session_id>/index.txt       — one-line summary per turn

Usage:
  uv run proxy.py [--upstream URL] [--port PORT] [--strip-date] [--verbose]
                  [--dump-dir PATH]

  # default: listen on 127.0.0.1:30001, forward to localhost:30000
  uv run proxy.py

  # enable prompt diffing
  uv run proxy.py --dump-dir ~/mllm/prompt-diffs

Point Claude Code at the proxy:
  ANTHROPIC_BASE_URL=http://localhost:30001 claude ...
"""

import argparse
import dataclasses
import difflib
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

# ── Config (overridden by CLI args) ──────────────────────────────────────────
UPSTREAM = "http://localhost:30000"
STRIP_DATE = False
VERBOSE = False
DUMP_DIR: Path | None = None

# Matches the framework-injected currentDate line, e.g.:
#   Today's date is 2026-02-21.\n
_DATE_RE = re.compile(r"Today's date is \d{4}-\d{2}-\d{2}\.\n?")

# Hop-by-hop headers that must not be forwarded from upstream response
_HOP_BY_HOP = frozenset(
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    ]
)

log = logging.getLogger("proxy")

app = FastAPI()


# ── Normalization ─────────────────────────────────────────────────────────────


def normalize(body: dict) -> tuple[dict, list[str]]:
    """Apply cache-stabilizing normalizations. Returns (body, list of changes)."""
    changes: list[str] = []

    # 1. Sort tools by name
    if isinstance(body.get("tools"), list) and body["tools"]:
        original_names = [t.get("name", "") for t in body["tools"]]
        body["tools"] = sorted(body["tools"], key=lambda t: t.get("name", ""))
        sorted_names = [t.get("name", "") for t in body["tools"]]
        if original_names != sorted_names:
            changes.append(
                f"tools reordered: [{', '.join(original_names[:4])}{'...' if len(original_names) > 4 else ''}]"
                f" → [{', '.join(sorted_names[:4])}{'...' if len(sorted_names) > 4 else ''}]"
            )

    # 2. Strip currentDate injection from user message content
    if STRIP_DATE and isinstance(body.get("messages"), list):
        for msg in body["messages"]:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and _DATE_RE.search(content):
                msg["content"] = _DATE_RE.sub("", content)
                changes.append("stripped currentDate from user message")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        if _DATE_RE.search(block["text"]):
                            block["text"] = _DATE_RE.sub("", block["text"])
                            changes.append("stripped currentDate from user content block")

    return body, changes


# ── Session-aware prompt diffing ──────────────────────────────────────────────


@dataclasses.dataclass
class SessionState:
    turn: int = 0
    prev_text: str = ""          # rendered text of previous turn (for diffing)
    session_dir: Path = None     # type: ignore[assignment]


# session_id → SessionState
_sessions: dict[str, SessionState] = {}


def _session_id(body: dict) -> str:
    """Stable session ID from the system prompt content."""
    system = body.get("system", "")
    if isinstance(system, list):
        # Anthropic multi-block system
        system = " ".join(
            b.get("text", "") for b in system if isinstance(b, dict)
        )
    return hashlib.sha256(system.encode()).hexdigest()[:12]


def _render_body(body: dict) -> str:
    """Human-readable rendering of a request body for diffing.

    Renders as labelled sections so diffs are easy to read:
      [tools]       sorted list of tool names + full defs
      [system]      system prompt text
      [messages]    each message as role: content
    """
    parts: list[str] = []

    # Tools section — names first for a quick overview, then full defs
    tools = body.get("tools") or []
    if tools:
        names_line = "# " + ", ".join(t.get("name", "?") for t in tools)
        defs = json.dumps(tools, indent=2, ensure_ascii=False)
        parts.append(f"[tools] ({len(tools)} total)\n{names_line}\n{defs}")

    # System section
    system = body.get("system", "")
    if isinstance(system, list):
        system_text = "\n".join(
            b.get("text", "") for b in system if isinstance(b, dict)
        )
    else:
        system_text = system or ""
    if system_text:
        parts.append(f"[system]\n{system_text}")

    # Messages section — one block per message
    for i, msg in enumerate(body.get("messages") or []):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            content_text = "\n".join(
                b.get("text", json.dumps(b, ensure_ascii=False))
                if isinstance(b, dict)
                else str(b)
                for b in content
            )
        else:
            content_text = str(content)
        parts.append(f"[message {i} / {role}]\n{content_text}")

    return "\n\n" + ("\n\n" + "─" * 80 + "\n\n").join(parts) + "\n"


def _dump_turn(body: dict) -> None:
    """Save full body (turn 0) or unified diff (turn N) for this session."""
    assert DUMP_DIR is not None
    sid = _session_id(body)
    state = _sessions.get(sid)
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    n_msgs = len(body.get("messages") or [])
    n_tools = len(body.get("tools") or [])

    if state is None:
        session_dir = DUMP_DIR / sid
        session_dir.mkdir(parents=True, exist_ok=True)
        state = SessionState(turn=0, prev_text="", session_dir=session_dir)
        _sessions[sid] = state
        log.info("dump: new session %s → %s", sid, session_dir)

    current_text = _render_body(body)
    turn = state.turn

    if turn == 0:
        # Full save
        out = state.session_dir / "turn_000.json"
        out.write_text(
            json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        size_kb = len(current_text) / 1024
        summary = f"turn 000  {now}  FULL  {n_msgs} msgs  {n_tools} tools  {size_kb:.1f} KB"
        log.info("dump: %s", summary)
    else:
        # Diff vs previous turn
        prev_lines = state.prev_text.splitlines(keepends=True)
        curr_lines = current_text.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                prev_lines,
                curr_lines,
                fromfile=f"turn_{turn - 1:03d}",
                tofile=f"turn_{turn:03d}",
                lineterm="",
            )
        )
        out = state.session_dir / f"turn_{turn:03d}.diff"
        out.write_text("".join(diff_lines), encoding="utf-8")
        added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
        summary = (
            f"turn {turn:03d}  {now}  +{added}/-{removed} lines  "
            f"{n_msgs} msgs  {n_tools} tools"
        )
        log.info("dump: %s", summary)

    # Append to index
    index = state.session_dir / "index.txt"
    with index.open("a", encoding="utf-8") as f:
        f.write(summary + "\n")

    state.prev_text = current_text
    state.turn += 1


# ── Request forwarding ────────────────────────────────────────────────────────


def _forward_headers(request: Request) -> dict[str, str]:
    """Build headers to send upstream, dropping hop-by-hop and host."""
    return {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", *_HOP_BY_HOP)
    }


def _response_headers(upstream_headers: httpx.Headers) -> dict[str, str]:
    """Strip hop-by-hop headers from upstream response."""
    return {
        k: v
        for k, v in upstream_headers.items()
        if k.lower() not in _HOP_BY_HOP
    }


@app.post("/v1/messages")
async def proxy_messages(request: Request) -> Response:
    body = await request.json()
    body, changes = normalize(body)

    if changes and VERBOSE:
        log.info("normalized: %s", "; ".join(changes))
    elif changes:
        log.debug("normalized: %s", "; ".join(changes))

    if DUMP_DIR is not None:
        try:
            _dump_turn(body)
        except Exception:
            log.exception("dump failed")

    headers = _forward_headers(request)
    stream = body.get("stream", False)

    if stream:
        async def generate() -> AsyncIterator[bytes]:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{UPSTREAM}/v1/messages",
                    json=body,
                    headers=headers,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{UPSTREAM}/v1/messages",
                json=body,
                headers=headers,
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=_response_headers(resp.headers),
        )


# ── Passthrough for all other endpoints (health, /v1/models, etc.) ────────────


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD"])
async def passthrough(request: Request, path: str) -> Response:
    body = await request.body()
    headers = _forward_headers(request)
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.request(
            method=request.method,
            url=f"{UPSTREAM}/{path}",
            content=body,
            headers=headers,
            params=dict(request.query_params),
        )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=_response_headers(resp.headers),
    )


# ── Entrypoint ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--upstream", default="http://localhost:30000",
        help="vLLM base URL (default: http://localhost:30000)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=30001,
        help="Listen port (default: 30001)",
    )
    parser.add_argument(
        "--strip-date", action="store_true",
        help="Strip framework-injected 'Today's date is ...' from user messages",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Log every normalization applied",
    )
    parser.add_argument(
        "--dump-dir", metavar="PATH",
        help=(
            "Enable session prompt diffing. Saves full body for turn 0 and "
            "unified diffs for subsequent turns under PATH/<session_id>/."
        ),
    )
    args = parser.parse_args()

    UPSTREAM = args.upstream
    STRIP_DATE = args.strip_date
    VERBOSE = args.verbose
    if args.dump_dir:
        DUMP_DIR = Path(args.dump_dir).expanduser().resolve()
        DUMP_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [proxy] %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Listening on %s:%d → upstream %s", args.host, args.port, UPSTREAM)
    log.info(
        "Tool sorting: enabled | Date stripping: %s | Dump dir: %s",
        "enabled" if STRIP_DATE else "disabled",
        DUMP_DIR or "disabled",
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
