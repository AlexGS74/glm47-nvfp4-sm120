#!/usr/bin/env python3
"""
Apply all patches required to serve GLM-4.7-NVFP4 on SM120 (RTX PRO 6000 Blackwell).

Usage:
    python apply_patches.py           # apply all
    python apply_patches.py --check   # check status without modifying
    python apply_patches.py --revert  # revert all (from .bak files)

Tested with: vllm==0.15.1, torch==2.9.1+cu128, flashinfer==0.6.1
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from patches import vllm_glm4_moe, vllm_anthropic_serving, vllm_glm47_tool_parser


PATCHES = [
    ("vLLM glm4_moe.py — skip missing k_scale/v_scale", vllm_glm4_moe),
    ("vLLM anthropic/serving.py — fix tool_calls NoneType crashes", vllm_anthropic_serving),
    ("vLLM glm47_moe_tool_parser.py — fix no-newline tool call format", vllm_glm47_tool_parser),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Report patch status only")
    ap.add_argument("--revert", action="store_true", help="Revert all patches from backups")
    args = ap.parse_args()

    ok = True
    for name, mod in PATCHES:
        print(f"\n{'='*60}")
        print(f"Patch: {name}")
        print('='*60)
        try:
            if args.check:
                target = mod.find_target_file()
                status = "APPLIED" if mod.is_patched(target) else "NOT applied"
                print(f"Status: {status}")
            elif args.revert:
                ok = mod.revert() and ok
            else:
                ok = mod.apply() and ok
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            ok = False

    print()
    if not args.check:
        print("All patches applied successfully." if ok else "Some patches FAILED — see above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
