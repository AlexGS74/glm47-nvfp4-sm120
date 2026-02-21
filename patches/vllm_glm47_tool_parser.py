"""
Patch vLLM glm47_moe_tool_parser.py to extract tool calls from inside <think> blocks.

GLM-4.7 is an interleaved thinker. With --reasoning-parser glm45 enabled, the
reasoning parser strips <think>...</think> content before the tool parser runs.
If the model places <tool_call> XML inside the think block (common when thinking
mode is on), they are silently discarded.

This patch adds a fallback: if no tool calls are found in the post-think text,
search inside the think block as well.

The chat_template.jinja patch (vllm_glm47_chat_template.py) adds an instruction
telling the model to place tool calls after </think>. This parser patch is the
safety net for cases where the model doesn't follow that instruction.

Tested with: vllm==0.15.1
"""

import sys
import shutil
from pathlib import Path

PATCH_MARKER = "Tool calls found inside <think> block, extracting as fallback"

OLD = '''\
class Glm47MoeModelToolParser(Glm4MoeModelToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self.func_detail_regex = re.compile(
            r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\\\\n|\\s)*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )'''

NEW = '''\
class Glm47MoeModelToolParser(Glm4MoeModelToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self.func_detail_regex = re.compile(
            r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\\\\n|\\s)*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )
        # Matches tool calls that leaked inside a <think>...</think> block
        self.think_block_regex = re.compile(
            r"<think>(.*?)</think>", re.DOTALL
        )

    def extract_tool_calls(self, model_output, request):
        result = super().extract_tool_calls(model_output, request)
        if result.tools_called:
            return result
        # Fallback: search inside <think> block if no tool calls found outside
        think_match = self.think_block_regex.search(model_output)
        if think_match:
            think_content = think_match.group(1)
            if "<tool_call>" in think_content:
                logger.debug(
                    "Tool calls found inside <think> block, extracting as fallback"
                )
                return super().extract_tool_calls(think_content, request)
        return result'''


def find_target_file():
    try:
        import vllm
        p = Path(vllm.__file__).parent / "tool_parsers" / "glm47_moe_tool_parser.py"
        if not p.exists():
            raise FileNotFoundError(f"glm47_moe_tool_parser.py not found at {p}")
        return p
    except ImportError:
        raise ImportError("vllm is not installed / not importable from current Python")


def is_patched(path: Path) -> bool:
    return PATCH_MARKER in path.read_text()


def apply(verbose: bool = True) -> bool:
    target = find_target_file()
    if verbose:
        print(f"Target: {target}")

    if is_patched(target):
        if verbose:
            print("Already patched — nothing to do.")
        return True

    text = target.read_text()
    # Normalize the regex escape sequences for matching
    normalized_old = OLD.replace('\\\\n', '\\n')
    if normalized_old not in text:
        print("ERROR: patch site not found — vLLM version may differ from 0.15.1", file=sys.stderr)
        return False

    backup = target.with_suffix(".py.bak")
    shutil.copy2(target, backup)
    if verbose:
        print(f"Backup: {backup}")

    text = text.replace(normalized_old, NEW.replace('\\\\n', '\\n'), 1)
    target.write_text(text)

    if verbose:
        print("Patch applied successfully.")
    return True


def revert(verbose: bool = True) -> bool:
    target = find_target_file()
    backup = target.with_suffix(".py.bak")
    if not backup.exists():
        print("No backup found — nothing to revert.", file=sys.stderr)
        return False
    shutil.copy2(backup, target)
    if verbose:
        print(f"Reverted {target} from {backup}")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Patch vLLM glm47_moe_tool_parser.py for think-block tool call fallback")
    ap.add_argument("--revert", action="store_true", help="Revert to backup")
    ap.add_argument("--check", action="store_true", help="Check if already patched")
    args = ap.parse_args()

    target = find_target_file()
    if args.check:
        print("Patched" if is_patched(target) else "Not patched")
    elif args.revert:
        sys.exit(0 if revert() else 1)
    else:
        sys.exit(0 if apply() else 1)
