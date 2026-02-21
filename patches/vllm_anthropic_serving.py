"""
Patch: vllm/entrypoints/anthropic/serving.py
Target: ~/.local/share/uv/tools/vllm/lib/python3.12/site-packages/vllm/entrypoints/anthropic/serving.py

Bug 1 — messages_full_converter (non-streaming):
    message.tool_calls is None when there are no tool calls, causing
    TypeError: 'NoneType' is not iterable and dropping the response.
    Fix: add `or []` guard.

Bug 2 — message_stream_converter (streaming):
    delta.tool_calls is None for non-tool-call chunks, causing
    TypeError: object of type 'NoneType' has no len().
    Fix: add None guard before len().

Bug 3 — message_stream_converter (streaming):
    vLLM tool parser emits complete tool calls in a single chunk (id + args
    together). The original code only emitted content_block_start but not the
    follow-up content_block_delta for the arguments.
    Fix: emit content_block_delta when args are present in the same chunk.

Bug 4 — message_stream_converter (streaming):
    vLLM sends delta.content="" (empty string, not None) alongside tool_calls.
    The content check `if ... content is not None` passes for empty string,
    enters the text branch, and the `elif tool_calls` is never reached, so
    all tool calls are silently dropped.
    Fix: check tool_calls BEFORE content.
"""

import sys
from pathlib import Path

PATCH_MARKER = "check BEFORE content; vLLM sends delta.content"

TARGET_SUBPATH = "vllm/entrypoints/anthropic/serving.py"

FIX1_OLD = "for tool_call in generator.choices[0].message.tool_calls:"
FIX1_NEW = "for tool_call in (generator.choices[0].message.tool_calls or []):"

# Fixes 2, 3, 4 are applied as one block replacement (streaming path)
FIX_STREAMING_OLD = """\
                        # content
                        if origin_chunk.choices[0].delta.content is not None:
                            if not content_block_started:
                                chunk = AnthropicStreamEvent(
                                    index=content_block_index,
                                    type="content_block_start",
                                    content_block=AnthropicContentBlock(
                                        type="text", text=""
                                    ),
                                )
                                data = chunk.model_dump_json(exclude_unset=True)
                                yield wrap_data_with_event(data, "content_block_start")
                                content_block_started = True

                            if origin_chunk.choices[0].delta.content == "":
                                continue
                            chunk = AnthropicStreamEvent(
                                index=content_block_index,
                                type="content_block_delta",
                                delta=AnthropicDelta(
                                    type="text_delta",
                                    text=origin_chunk.choices[0].delta.content,
                                ),
                            )
                            data = chunk.model_dump_json(exclude_unset=True)
                            yield wrap_data_with_event(data, "content_block_delta")
                            continue

                        # tool calls
                        elif len(origin_chunk.choices[0].delta.tool_calls) > 0:
                            tool_call = origin_chunk.choices[0].delta.tool_calls[0]
                            if tool_call.id is not None:
                                if content_block_started:
                                    stop_chunk = AnthropicStreamEvent(
                                        index=content_block_index,
                                        type="content_block_stop",
                                    )
                                    data = stop_chunk.model_dump_json(
                                        exclude_unset=True
                                    )
                                    yield wrap_data_with_event(
                                        data, "content_block_stop"
                                    )
                                    content_block_started = False
                                    content_block_index += 1

                                chunk = AnthropicStreamEvent(
                                    index=content_block_index,
                                    type="content_block_start",
                                    content_block=AnthropicContentBlock(
                                        type="tool_use",
                                        id=tool_call.id,
                                        name=tool_call.function.name
                                        if tool_call.function
                                        else None,
                                        input={},
                                    ),
                                )
                                data = chunk.model_dump_json(exclude_unset=True)
                                yield wrap_data_with_event(data, "content_block_start")
                                content_block_started = True

                            else:\
"""

FIX_STREAMING_NEW = """\
                        # tool calls — check BEFORE content; vLLM sends delta.content=""
                        # alongside tool_calls, so content check must not run first
                        if origin_chunk.choices[0].delta.tool_calls and len(origin_chunk.choices[0].delta.tool_calls) > 0:
                            tool_call = origin_chunk.choices[0].delta.tool_calls[0]
                            if tool_call.id is not None:
                                if content_block_started:
                                    stop_chunk = AnthropicStreamEvent(
                                        index=content_block_index,
                                        type="content_block_stop",
                                    )
                                    data = stop_chunk.model_dump_json(
                                        exclude_unset=True
                                    )
                                    yield wrap_data_with_event(
                                        data, "content_block_stop"
                                    )
                                    content_block_started = False
                                    content_block_index += 1

                                chunk = AnthropicStreamEvent(
                                    index=content_block_index,
                                    type="content_block_start",
                                    content_block=AnthropicContentBlock(
                                        type="tool_use",
                                        id=tool_call.id,
                                        name=tool_call.function.name
                                        if tool_call.function
                                        else None,
                                        input={},
                                    ),
                                )
                                data = chunk.model_dump_json(exclude_unset=True)
                                yield wrap_data_with_event(data, "content_block_start")
                                content_block_started = True

                                # Arguments may arrive in the same chunk as the id
                                # (vLLM tool parser emits complete tool calls at once)
                                if tool_call.function and tool_call.function.arguments:
                                    delta_chunk = AnthropicStreamEvent(
                                        index=content_block_index,
                                        type="content_block_delta",
                                        delta=AnthropicDelta(
                                            type="input_json_delta",
                                            partial_json=tool_call.function.arguments,
                                        ),
                                    )
                                    data = delta_chunk.model_dump_json(exclude_unset=True)
                                    yield wrap_data_with_event(data, "content_block_delta")

                            else:\
"""

FIX_CONTENT_OLD = """\
                            continue
                else:"""

FIX_CONTENT_NEW = """\
                            continue

                        # content
                        elif origin_chunk.choices[0].delta.content is not None:
                            if not content_block_started:
                                chunk = AnthropicStreamEvent(
                                    index=content_block_index,
                                    type="content_block_start",
                                    content_block=AnthropicContentBlock(
                                        type="text", text=""
                                    ),
                                )
                                data = chunk.model_dump_json(exclude_unset=True)
                                yield wrap_data_with_event(data, "content_block_start")
                                content_block_started = True

                            if origin_chunk.choices[0].delta.content == "":
                                continue
                            chunk = AnthropicStreamEvent(
                                index=content_block_index,
                                type="content_block_delta",
                                delta=AnthropicDelta(
                                    type="text_delta",
                                    text=origin_chunk.choices[0].delta.content,
                                ),
                            )
                            data = chunk.model_dump_json(exclude_unset=True)
                            yield wrap_data_with_event(data, "content_block_delta")
                            continue
                else:"""


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
    if FIX_STREAMING_OLD not in text:
        print(f"ERROR: Streaming fix anchor not found in {target}", file=sys.stderr)
        return False
    text = text.replace(FIX1_OLD, FIX1_NEW)
    text = text.replace(FIX_STREAMING_OLD, FIX_STREAMING_NEW)
    text = text.replace(FIX_CONTENT_OLD, FIX_CONTENT_NEW)
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
    text = text.replace(FIX_STREAMING_NEW, FIX_STREAMING_OLD)
    text = text.replace(FIX_CONTENT_NEW, FIX_CONTENT_OLD)
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
