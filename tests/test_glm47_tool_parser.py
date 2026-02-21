"""
Unit tests for GLM-4.7 tool call parser.

Tests the key difference from GLM-4.5: GLM-4.7 emits tool calls without a
newline between the function name and the first arg tag.

Run with: python -m pytest tests/test_glm47_tool_parser.py -v
Requires: vllm installed (uv tool install vllm==0.15.1)
"""

import json
import pytest


def make_parser():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "patches"))
    from unittest.mock import MagicMock
    from glm47_moe_tool_parser import Glm47MoeModelToolParser

    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        "<tool_call>": 1001,
        "</tool_call>": 1002,
    }
    return Glm47MoeModelToolParser(tokenizer)


def make_request(tools):
    from unittest.mock import MagicMock
    req = MagicMock()
    req.tools = tools
    return req


# ── GLM-4.7 format: no newline between function name and args ─────────────────

def test_single_tool_no_newline():
    """GLM-4.7 emits <tool_call>Bash<arg_key>... without a newline."""
    parser = make_parser()
    output = (
        "<tool_call>Bash"
        "<arg_key>command</arg_key>"
        "<arg_value>ls -la /tmp</arg_value>"
        "</tool_call>"
    )
    result = parser.extract_tool_calls(output, make_request(None))
    assert result.tools_called
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.function.name == "Bash"
    assert json.loads(tc.function.arguments) == {"command": "ls -la /tmp"}


def test_multiple_args_no_newline():
    """Multiple arg pairs without newline separator."""
    parser = make_parser()
    output = (
        "<tool_call>Bash"
        "<arg_key>command</arg_key><arg_value>ls -la /home/alex/mllm</arg_value>"
        "<arg_key>description</arg_key><arg_value>List folder contents</arg_value>"
        "</tool_call>"
    )
    result = parser.extract_tool_calls(output, make_request(None))
    assert result.tools_called
    tc = result.tool_calls[0]
    assert tc.function.name == "Bash"
    args = json.loads(tc.function.arguments)
    assert args["command"] == "ls -la /home/alex/mllm"
    assert args["description"] == "List folder contents"


def test_tool_call_with_preceding_text():
    """Text before tool call becomes content, not part of the call."""
    parser = make_parser()
    output = (
        "I'll list the files for you."
        "<tool_call>Bash"
        "<arg_key>command</arg_key><arg_value>ls /tmp</arg_value>"
        "</tool_call>"
    )
    result = parser.extract_tool_calls(output, make_request(None))
    assert result.tools_called
    assert "I'll list" in (result.content or "")
    assert result.tool_calls[0].function.name == "Bash"


def test_no_tool_call():
    """Plain text response returns no tool calls."""
    parser = make_parser()
    result = parser.extract_tool_calls("Hello, how can I help?", make_request(None))
    assert not result.tools_called
    assert result.tool_calls == []


def test_glm45_format_with_newline_still_works():
    """Parent GLM-4.5 format (with newline) should still parse correctly."""
    parser = make_parser()
    output = (
        "<tool_call>list_files\n"
        "<arg_key>path</arg_key><arg_value>/tmp</arg_value>"
        "</tool_call>"
    )
    result = parser.extract_tool_calls(output, make_request(None))
    assert result.tools_called
    assert result.tool_calls[0].function.name == "list_files"
