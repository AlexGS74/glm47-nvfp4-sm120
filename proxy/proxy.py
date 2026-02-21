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

Usage:
  uv run proxy.py [--upstream URL] [--port PORT] [--strip-date] [--verbose]

  # default: listen on 127.0.0.1:30001, forward to localhost:30000
  uv run proxy.py

  # custom upstream / port
  uv run proxy.py --upstream http://localhost:30000 --port 30001

Point Claude Code at the proxy:
  ANTHROPIC_BASE_URL=http://localhost:30001 claude ...
"""

import argparse
import logging
import re
import sys
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

# ── Config (overridden by CLI args) ──────────────────────────────────────────
UPSTREAM = "http://localhost:30000"
STRIP_DATE = False
VERBOSE = False

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
                f"tools reordered: [{', '.join(original_names[:4])}{'...' if len(original_names)>4 else ''}]"
                f" → [{', '.join(sorted_names[:4])}{'...' if len(sorted_names)>4 else ''}]"
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", default="http://localhost:30000", help="vLLM base URL (default: http://localhost:30000)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=30001, help="Listen port (default: 30001)")
    parser.add_argument("--strip-date", action="store_true", help="Strip framework-injected 'Today's date is ...' from user messages")
    parser.add_argument("--verbose", action="store_true", help="Log every normalization applied")
    args = parser.parse_args()

    UPSTREAM = args.upstream
    STRIP_DATE = args.strip_date
    VERBOSE = args.verbose

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [proxy] %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Listening on %s:%d → upstream %s", args.host, args.port, UPSTREAM)
    log.info("Tool sorting: enabled | Date stripping: %s", "enabled" if STRIP_DATE else "disabled")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
