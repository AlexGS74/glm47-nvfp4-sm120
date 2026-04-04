"""Patch glm4_moe.py to skip missing k_scale/v_scale keys in weight loader.

Usage: python3 glm4_moe_skip_scales.py /opt/vllm/vllm/model_executor/models/glm4_moe.py
"""
import sys

path = sys.argv[1]
with open(path) as f:
    lines = f.readlines()

patch_count = 0
out = []
for line in lines:
    # Insert skip before each `param = params_dict[name` line
    if 'param = params_dict[name' in line and 'k_scale' not in line:
        indent = line[:len(line) - len(line.lstrip())]
        # Extract the variable name (name or name_mapped)
        var = line.strip().split('params_dict[')[1].split(']')[0]
        skip = f"{indent}if ('k_scale' in {var} or 'v_scale' in {var}) and {var} not in params_dict:\n{indent}    continue\n"
        out.append(skip)
        patch_count += 1
    out.append(line)

with open(path, 'w') as f:
    f.writelines(out)

print(f"Patched {patch_count} locations in {path}")
