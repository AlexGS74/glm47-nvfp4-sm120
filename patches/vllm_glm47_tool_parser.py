"""
Patch: vllm/tool_parsers/glm47_moe_tool_parser.py

Bug: GLM-4.7 emits tool calls WITHOUT a newline between the function name and
the first arg tag. The parent class (Glm4MoeModelToolParser) regex expects a
newline separator, so all GLM-4.7 tool calls fail to parse.

  Parent regex: r"<tool_call>([^\\n]*)\\n(.*)</tool_call>"  ← requires \\n
  Model output: <tool_call>Bash<arg_key>command</arg_key>...  ← no \\n

Fix: replace glm47_moe_tool_parser.py entirely with a version that overrides
func_detail_regex to use r"<tool_call>([^\\s<]+)\\s*(.*?)</tool_call>".
"""

import sys
import shutil
from pathlib import Path

TARGET_SUBPATH = "vllm/tool_parsers/glm47_moe_tool_parser.py"
SOURCE = Path(__file__).parent / "glm47_moe_tool_parser.py"
PATCH_MARKER = "GLM-4.7 emits tool calls WITHOUT a newline"


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
    backup = target.with_suffix(".py.bak")
    shutil.copy2(target, backup)
    shutil.copy2(SOURCE, target)
    print(f"Patched: {target}  (backup: {backup})")
    return True


def revert() -> bool:
    target = find_target_file()
    backup = target.with_suffix(".py.bak")
    if not backup.exists():
        print(f"No backup found: {backup}")
        return False
    shutil.copy2(backup, target)
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
