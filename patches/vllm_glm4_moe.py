"""
Patch vLLM glm4_moe.py to skip missing k_scale/v_scale tensors.

The Salyut1/GLM-4.7-NVFP4 checkpoint does not include FP8 KV-cache scale
tensors. vLLM's AutoWeightLoader iterates raw checkpoint keys and crashes
with KeyError when those tensors are absent from params_dict.

Two patch sites in load_weights():
  Site 1 — expert weight shard loader: always required.
  Site 2 — non-expert weight loader:   required on 0.15.x; on 0.16.x+
            maybe_remap_kv_scale_name() already handles it via None-return,
            so site 2 is patched opportunistically but missing it is OK.

Tested with: vllm==0.15.1, vllm==0.16.1rc1.dev106+ge113a3011
"""

import sys
import shutil
from pathlib import Path

PATCH_MARKER = "('k_scale' in name or 'v_scale' in name) and name not in params_dict"

# ── Site 1: expert weight shard loader ────────────────────────────────────────
# Present in 0.15.x and 0.16.x (same code; 0.16.x adds `break` on the next
# line, which is not part of the match string).

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

# ── Site 2: non-expert weight loader ──────────────────────────────────────────
# 0.15.x: this is the only guard — patch required.
# 0.16.x: maybe_remap_kv_scale_name() runs before this block and returns None
#          for unknown scale names, so the crash cannot be reached. We still
#          patch it if present (belt-and-suspenders), but absence is acceptable
#          when SENTINEL_MAYBE_REMAP is detected in the file.

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

# If this function is present the non-expert path already handles unknown
# scale names safely (returns None → continue).
SENTINEL_MAYBE_REMAP = "maybe_remap_kv_scale_name"


def get_vllm_version() -> str:
    try:
        import vllm
        return vllm.__version__
    except ImportError:
        return "unknown"


def find_target_file() -> Path:
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
    version = get_vllm_version()
    if verbose:
        print(f"Target:  {target}")
        print(f"Version: {version}")

    if is_patched(target):
        if verbose:
            print("Already patched — nothing to do.")
        return True

    text = target.read_text()

    # ── Site 1 (always required) ───────────────────────────────────────────────
    if OLD_1 not in text:
        print(
            f"ERROR: patch site 1 not found in vllm {version}.\n"
            "  Inspect the expert weight shard loader in glm4_moe.py manually.",
            file=sys.stderr,
        )
        return False

    # ── Site 2 (required on 0.15.x; optional on 0.16.x+ with maybe_remap) ────
    site2_present = OLD_2 in text
    has_maybe_remap = SENTINEL_MAYBE_REMAP in text

    if not site2_present:
        if has_maybe_remap:
            if verbose:
                print(
                    "Note: patch site 2 not found, but maybe_remap_kv_scale_name() "
                    "is present — non-expert k_scale/v_scale already handled safely."
                )
        else:
            print(
                f"ERROR: patch site 2 not found in vllm {version} and "
                "maybe_remap_kv_scale_name() is absent.\n"
                "  Inspect the non-expert weight loader in glm4_moe.py manually.",
                file=sys.stderr,
            )
            return False

    backup = target.with_suffix(".py.bak")
    shutil.copy2(target, backup)
    if verbose:
        print(f"Backup:  {backup}")

    text = text.replace(OLD_1, NEW_1, 1)
    if site2_present:
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
        version = get_vllm_version()
        print(f"Version: {version}")
        print("Patched" if is_patched(target) else "Not patched")
    elif args.revert:
        sys.exit(0 if revert() else 1)
    else:
        sys.exit(0 if apply() else 1)
