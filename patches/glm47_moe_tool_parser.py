# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Patch: replaces vllm/tool_parsers/glm47_moe_tool_parser.py
#
# GLM-4.7 emits tool calls WITHOUT a newline between the function name and the
# first arg tag, unlike GLM-4.5 which uses a newline separator:
#
#   GLM-4.5: <tool_call>func_name\n<arg_key>...</arg_key>...</tool_call>
#   GLM-4.7: <tool_call>func_name<arg_key>...</arg_key>...</tool_call>
#
# The parent class regex r"<tool_call>([^\n]*)\n(.*)</tool_call>" fails because
# it requires the newline. This subclass overrides both regexes.

import regex as re

from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.glm4_moe_tool_parser import Glm4MoeModelToolParser

logger = init_logger(__name__)


class Glm47MoeModelToolParser(Glm4MoeModelToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        # Capture function name as non-whitespace chars before any arg tag or
        # end-of-call, then the remaining arg tags as group 2.
        # Works for both GLM-4.7 (no newline) and GLM-4.5 (newline) formats.
        self.func_detail_regex = re.compile(
            r"<tool_call>([^\s<]+)\s*(.*?)</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )
