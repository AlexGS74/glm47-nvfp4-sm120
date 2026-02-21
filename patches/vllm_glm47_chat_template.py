"""
Patch GLM-4.7-NVFP4 chat_template.jinja to instruct the model to place
tool calls after </think>, not inside the thinking block.

GLM-4.7 is an interleaved thinker. When thinking mode is active, the model
sometimes places <tool_call> XML inside the <think> block. The vLLM reasoning
parser strips that content before the tool parser runs, causing tool calls to
be silently lost.

This patch adds an explicit instruction in the tool section of the chat template
telling the model to emit tool calls after the closing </think> tag.

Tested with: vllm==0.15.1, Salyut1/GLM-4.7-NVFP4 snapshot 531df318
"""

import sys
import shutil
from pathlib import Path

MODEL_SNAPSHOT = "models--Salyut1--GLM-4.7-NVFP4/snapshots/531df318dbe2877b24881f6e15024c7f945e7048"
PATCH_MARKER = "tool calls must appear AFTER the closing </think> tag"

OLD = "For each function call, output the function name and arguments within the following XML format:"
NEW = "For each function call, output the function name and arguments within the following XML format. IMPORTANT: tool calls must appear AFTER the closing </think> tag, never inside the thinking block:"


def find_target_file():
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    candidates = list(hf_cache.glob("**/GLM-4.7*/**/chat_template.jinja"))
    if not candidates:
        raise FileNotFoundError(
            f"chat_template.jinja not found under {hf_cache}. "
            "Set HF_HOME if your cache is elsewhere."
        )
    return candidates[0]


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
    if OLD not in text:
        print("ERROR: patch site not found — template may have changed", file=sys.stderr)
        return False

    backup = target.with_suffix(".jinja.bak")
    shutil.copy2(target, backup)
    if verbose:
        print(f"Backup: {backup}")

    text = text.replace(OLD, NEW, 1)
    target.write_text(text)

    if verbose:
        print("Patch applied successfully.")
    return True


def revert(verbose: bool = True) -> bool:
    target = find_target_file()
    backup = target.with_suffix(".jinja.bak")
    if not backup.exists():
        print("No backup found — nothing to revert.", file=sys.stderr)
        return False
    shutil.copy2(backup, target)
    if verbose:
        print(f"Reverted {target} from {backup}")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Patch GLM-4.7 chat_template.jinja for tool call placement")
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
