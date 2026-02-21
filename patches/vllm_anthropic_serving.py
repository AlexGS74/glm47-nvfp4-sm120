"""
Patch: vllm/entrypoints/anthropic/serving.py
Target: ~/.local/share/uv/tools/vllm/lib/python3.12/site-packages/vllm/entrypoints/anthropic/serving.py

Bug: Two NoneType crashes in the Anthropic messages endpoint that prevent
tool calls from being returned to Anthropic clients (e.g. Claude Code).

Fix 1 — messages_full_converter (non-streaming), line ~282:
    Before: for tool_call in generator.choices[0].message.tool_calls:
    After:  for tool_call in (generator.choices[0].message.tool_calls or []):
    Reason: message.tool_calls is None when there are no tool calls, causing
            TypeError: 'NoneType' is not iterable and dropping the response.

Fix 2 — message_stream_converter (streaming), line ~403:
    Before: elif len(origin_chunk.choices[0].delta.tool_calls) > 0:
    After:  elif origin_chunk.choices[0].delta.tool_calls and len(...) > 0:
    Reason: delta.tool_calls is None for non-tool-call chunks, causing
            TypeError: object of type 'NoneType' has no len().
"""

import sys
from pathlib import Path

PATCH_MARKER = "tool_calls or []"

TARGET_SUBPATH = (
    "vllm/entrypoints/anthropic/serving.py"
)

FIX1_OLD = "for tool_call in generator.choices[0].message.tool_calls:"
FIX1_NEW = "for tool_call in (generator.choices[0].message.tool_calls or []):"

FIX2_OLD = "elif len(origin_chunk.choices[0].delta.tool_calls) > 0:"
FIX2_NEW = "elif origin_chunk.choices[0].delta.tool_calls and len(origin_chunk.choices[0].delta.tool_calls) > 0:"


def find_target_file() -> Path:
    import vllm
    base = Path(vllm.__file__).parent
    target = base / TARGET_SUBPATH.replace("vllm/", "", 1)
    if not target.exists():
        print(f"ERROR: target not found: {target}", file=sys.stderr)
        sys.exit(1)
    return target


def is_patched(target: Path) -> bool:
    return PATCH_MARKER in target.read_text()


def apply() -> bool:
    target = find_target_file()
    if is_patched(target):
        print(f"Already patched: {target}")
        return True
    text = target.read_text()
    if FIX1_OLD not in text:
        print(f"ERROR: Fix1 anchor not found in {target}", file=sys.stderr)
        return False
    if FIX2_OLD not in text:
        print(f"ERROR: Fix2 anchor not found in {target}", file=sys.stderr)
        return False
    text = text.replace(FIX1_OLD, FIX1_NEW)
    text = text.replace(FIX2_OLD, FIX2_NEW)
    target.write_text(text)
    print(f"Patched: {target}")
    return True


def revert() -> bool:
    target = find_target_file()
    if not is_patched(target):
        print(f"Not patched: {target}")
        return True
    text = target.read_text()
    text = text.replace(FIX1_NEW, FIX1_OLD)
    text = text.replace(FIX2_NEW, FIX2_OLD)
    target.write_text(text)
    print(f"Reverted: {target}")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    if args.check:
        print("Patched" if is_patched(find_target_file()) else "Not patched")
    elif args.revert:
        sys.exit(0 if revert() else 1)
    else:
        sys.exit(0 if apply() else 1)
