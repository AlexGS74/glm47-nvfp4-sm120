"""
Patch vLLM glm4_moe.py to skip missing k_scale/v_scale tensors.

The Salyut1/GLM-4.7-NVFP4 checkpoint does not include FP8 KV-cache scale
tensors. vLLM 0.15.1's AutoWeightLoader iterates raw checkpoint keys and
crashes with KeyError when those tensors are absent from params_dict.

Tested with: vllm==0.15.1
"""

import sys
import shutil
from pathlib import Path

PATCH_MARKER = "('k_scale' in name or 'v_scale' in name) and name not in params_dict"

OLD_1 = """\
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)"""

NEW_1 = """\
                if is_pp_missing_parameter(name, self):
                    continue

                # PATCH: skip missing k_scale/v_scale
                if ('k_scale' in name or 'v_scale' in name) and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)"""

OLD_2 = """\
                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)"""

NEW_2 = """\
                    if is_pp_missing_parameter(name, self):
                        continue

                    # PATCH: skip missing k_scale/v_scale
                    if ('k_scale' in name or 'v_scale' in name) and name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)"""


def find_target_file():
    try:
        import vllm
        p = Path(vllm.__file__).parent / "model_executor" / "models" / "glm4_moe.py"
        if not p.exists():
            raise FileNotFoundError(f"glm4_moe.py not found at {p}")
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

    if OLD_1 not in text:
        print("ERROR: patch site 1 not found — vLLM version may differ from 0.15.1", file=sys.stderr)
        return False
    if OLD_2 not in text:
        print("ERROR: patch site 2 not found — vLLM version may differ from 0.15.1", file=sys.stderr)
        return False

    backup = target.with_suffix(".py.bak")
    shutil.copy2(target, backup)
    if verbose:
        print(f"Backup: {backup}")

    text = text.replace(OLD_1, NEW_1, 1)
    text = text.replace(OLD_2, NEW_2, 1)
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
    ap = argparse.ArgumentParser(description="Patch vLLM glm4_moe.py for GLM-4.7-NVFP4")
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
