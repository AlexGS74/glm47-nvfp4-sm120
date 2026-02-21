# SM120 Blackwell FP4 MoE — SGLang Fixes Report

**Date:** 2026-02-20
**Hardware:** 4x NVIDIA RTX PRO 6000 Blackwell (SM120, compute capability 12.0)
**OS:** Ubuntu (new install, driver-only, no CUDA toolkit initially)
**Driver:** 580, nvidia-smi reports CUDA 13.0 (max supported); PyTorch uses cu128 (12.8)
**Model:** `Salyut1/GLM-4.7-NVFP4` (NVFP4 / ModelOpt FP4 quantized)
**SGLang:** dev install at `/home/alex/mllm/sglang`

---

## Background

SM100 is datacenter Blackwell (B200, GB200). SM120 is consumer/workstation Blackwell (RTX PRO 6000, RTX 5090, etc.). All FP4 MoE backends in flashinfer were originally hardcoded for SM100 only. SGLang's auto-selection of `flashinfer_trtllm` is also gated on `is_sm100_supported()`, which returns False for SM120.

`nvidia-smi` reports the maximum CUDA version the driver supports (13.0), while PyTorch is compiled against CUDA 12.8. Both are valid simultaneously — only the PyTorch CUDA version matters for kernel compilation.

---

## Prerequisites

- **nvcc / CUDA toolkit required** for flashinfer JIT kernel compilation. Driver alone is not enough.
  ```bash
  sudo apt install -y cuda-toolkit-12-8
  export PATH=/usr/local/cuda-12.8/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
  ```
- **uv** for running the server in the repo-managed environment.

---

## Serving Script

`scripts/serve_glm47_nvfp4.sh` — run with `TP=4` for a 94 GB model across 4x 96 GB GPUs:

```bash
TP=4 bash scripts/serve_glm47_nvfp4.sh
```

Key flags required for GLM-4.7:
```
--trust-remote-code
--tool-call-parser glm47
--reasoning-parser glm45
--enable-custom-logit-processor
--moe-runner-backend flashinfer_trtllm
```

Use the `/v1/chat/completions` endpoint (not `/generate`) to get properly formatted output.

---

## Errors Encountered and Fixes Applied

### 1. Triton FP4 kernel missing for SM120

**Error (not a crash — silent garbage output):**
```
run_fp4_blockwise_scaled_group_mm_sm120 half no implementation produced
```

**Cause:** Triton FP4 MoE kernel has no SM120 implementation. With `--moe-runner-backend triton` the model runs but produces garbage.

**Fix:** Use `--moe-runner-backend flashinfer_trtllm` instead. This requires all the patches below.

---

### 2. No CUDA architectures found for SM120

**Error:**
```
RuntimeError: No supported CUDA architectures found for major versions [10].
```

**File:** `.venv/lib/python3.12/site-packages/flashinfer/jit/fused_moe.py`, line 237

**Cause:** `gen_trtllm_gen_fused_moe_sm100_module()` only passes `supported_major_versions=[10]` to `get_nvcc_flags_list()`, which skips SM120 (major=12).

**Fix:**
```python
# Before
nvcc_flags = current_compilation_context.get_nvcc_flags_list(
    supported_major_versions=[10]
)

# After
# currently only support Blackwell (sm100 datacenter + sm120 consumer)
nvcc_flags = current_compilation_context.get_nvcc_flags_list(
    supported_major_versions=[10, 12]
)
```

---

### 3. correction_bias dtype mismatch

**Error:**
```
RuntimeError: routing_bias must be bfloat16 or float
```

**File:** `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`

**Cause:** `correction_bias` was cast to `hidden_states.dtype` (float16 when `--dtype half`), but the TRT-LLM kernel requires bfloat16 or float32.

**Fix:**
```python
correction_bias = (
    None
    if topk_config.correction_bias is None
    else topk_config.correction_bias.to(
        torch.bfloat16
        if hidden_states.dtype == torch.float16
        else hidden_states.dtype
    )
)
```

---

### 4. FP4BlockScaleLauncher SM10x hardcode

**Error:**
```
RuntimeError: Check failed: std::get<0>(device_props) == 10 (12 vs. 10) :
This kernel requires 10.x architecture. Current device has SM 120
  File "trtllm_fused_moe_kernel_launcher.cu", line 1222, in FP4BlockScaleLauncher::init
```

**File:** `.venv/lib/python3.12/site-packages/flashinfer/data/csrc/trtllm_fused_moe_kernel_launcher.cu`, line ~1222

**Fix:**
```cpp
// Before
TVM_FFI_ICHECK_EQ(std::get<0>(device_props), 10)
    << "This kernel requires 10.x architecture. Current device has SM "
    << std::get<0>(device_props) << std::get<1>(device_props);

// After
TVM_FFI_ICHECK(std::get<0>(device_props) == 10 || std::get<0>(device_props) == 12)
    << "This kernel requires 10.x or 12.x architecture. Current device has SM "
    << std::get<0>(device_props) << std::get<1>(device_props);
```

After patching `.cu` files, always clear the JIT cache:
```bash
rm -rf ~/.cache/flashinfer/
```

---

### 5. FusedMoeLauncher::init_common SM10x hardcode

**Error:**
```
RuntimeError: Check failed: major == 10 (12 vs. 10) :
MoE kernel requires 10.x architecture. Current device has SM 120
  File "trtllm_fused_moe_kernel_launcher.cu", line 385, in FusedMoeLauncher::init_common
```

**File:** `.venv/lib/python3.12/site-packages/flashinfer/data/csrc/trtllm_fused_moe_kernel_launcher.cu`, line ~385

**Fix:**
```cpp
// Before
TVM_FFI_ICHECK_EQ(major, 10) << "MoE kernel requires 10.x architecture. Current device has SM "
                             << major << minor;

// After
TVM_FFI_ICHECK(major == 10 || major == 12)
    << "MoE kernel requires 10.x or 12.x architecture. Current device has SM "
    << major << minor;
```

After patching, clear the JIT cache:
```bash
rm -rf ~/.cache/flashinfer/
```

---

### 6. SM100 GEMM cubin fails at execution on SM120 — UNRESOLVED UPSTREAM GAP

**Error (after all above patches):**
```
RuntimeError: Error in function 'run' at trtllm_batched_gemm_runner.cu:264:
Error occurred when running GEMM!
(numBatches: 160, GemmMNK: 1 768 5120,
 Kernel: bmm_E2m1_E2m1E2m1_Fp32_bA16_bB16_bC16_t128x8x512u2_s5_et128x8_m128x8x64_
         cga1x1x1_16dp256b_rM_TN_transOut_schedS_biasM_bN_ldgsts_tmaOpt_clmp_
         swiGlu_dynBatch_sm100f)
```

**Root cause:** The kernel name suffix `_sm100f` is the critical detail. This is a *precompiled binary cubin* downloaded from flashinfer's online TRT-LLM gen kernel cache, compiled specifically for SM100 (datacenter Blackwell B200). Binary SM100 code cannot execute on SM120 hardware.

This is fundamentally different from the previous errors (compile-time checks). No amount of patching C++ assertion macros can make an SM100 cubin run on SM120.

**Call chain:**
```
layer.py:1337 trtllm_fp4_block_scale_moe(...)
  → flashinfer/fused_moe/core.py:2552 get_trtllm_moe_sm100_module().trtllm_fp4_block_scale_moe(...)
      → gen_trtllm_gen_fused_moe_sm100_module()  ← downloads SM100 cubins from cloud cache
          → trtllm_batched_gemm_runner.cu:264     ← SM100 binary crashes on SM120
```

**What exists vs. what's missing:**

| Path | SM120 compile support | SM120 runtime support | FP4 |
|------|----------------------|----------------------|-----|
| `flashinfer_trtllm` (TRT-LLM gen cubins) | No — downloads SM100-only cubins | **No** | Yes (SM100 only) |
| `flashinfer_cutlass` (CUTLASS JIT) | Yes — `gen_cutlass_fused_moe_sm120_module` exists with `-DENABLE_FP4` | Yes | **Not routed from SGLang's FP4 layer** |

The CUTLASS JIT path (`gen_cutlass_fused_moe_sm120_module`) does have SM120+FP4 support compiled in. However, SGLang's `ModelOptNvFp4FusedMoEMethod.forward_impl` unconditionally dispatches to `trtllm_fp4_block_scale_moe`, never to a CUTLASS FP4 kernel. There is no CUTLASS-based FP4 MoE dispatch path in SGLang or flashinfer today.

**Status:** Blocked with `flashinfer >= 0.6.0`. See regression note in Upstream Tracking section.

**What a complete fix requires (upstream work):**
1. Either: Add SM120 cubins to flashinfer's TRT-LLM gen online kernel cache
2. Or: Add a CUTLASS-based `fp4_block_scale_moe` function to flashinfer, and wire it into SGLang's `forward_impl` for SM120 detection

---

## Upstream Tracking

### Regression: flashinfer 0.5.x → 0.6.x

SGLang PR #11708 (merged 2025-10-28) explicitly benchmarked NVFP4 on **8x RTX PRO 6000 at ~734 tok/s** using:

```bash
SGLANG_USE_CUTLASS_BACKEND_FOR_FP4_GEMM=1 \
uv --project python run python -m sglang.launch_server \
    --moe-runner-backend flashinfer_cutlass \
    --quantization modelopt_fp4 ...
```

This used the CUTLASS FP4 path via flashinfer 0.5.x (SGLang v0.5.6 pins `flashinfer_python==0.5.3`). The model card for `Salyut1/GLM-4.7-NVFP4` states it was quantized and benchmarked on 8x RTX PRO 6000 — confirmed working at that time.

**flashinfer v0.6.0 (Jan 2026)** introduced the `flashinfer_trtllm` backend using SM100-only precompiled cubins, and the CUTLASS FP4 GEMM path began returning zeros on SM120. The current pin (`flashinfer 0.6.3`) has both problems.

**Tested recovery approach:** Check out SGLang v0.5.6 (flashinfer 0.5.3) and run with `--moe-runner-backend flashinfer_cutlass`. See next section.

---

### Status as of 2026-02-20

**NVFP4 FP4 GEMM is broken on SM120 with flashinfer >= 0.6.0.** Actively tracked:

| Repo | Issue/PR | State | Summary |
|------|----------|-------|---------|
| flashinfer | [#2577](https://github.com/flashinfer-ai/flashinfer/issues/2577) | **open** | NVFP4 mm_fp4 GEMM broken on SM120 — all backends fail: cudnn rejects, cutlass silently returns zeros, trtllm explicitly rejects SM120 |
| sglang | [#18954](https://github.com/sgl-project/sglang/issues/18954) | **open** | NVFP4 models produce NaN outputs on RTX PRO 6000 (SM120) — cutlass backend silently returns zeros → NaN propagation |
| flashinfer | [#2077](https://github.com/flashinfer-ai/flashinfer/issues/2077) | **open** | MoE autotuner prints many failed kernels on SM120 |
| sglang | [#17526](https://github.com/sgl-project/sglang/issues/17526) | open | GLM 4.5/4.6/4.7 Blackwell performance optimization (broader tracking issue) |
| sglang | [#17421](https://github.com/sgl-project/sglang/pull/17421) | merged | Add SM121 (DGX Spark / GB10) support for NVFP4 MoE kernels in sgl-kernel — **SM120 was NOT included** |
| flashinfer | [#2553](https://github.com/flashinfer-ai/flashinfer/pull/2553) | open | Filter out cuDNN runtime-compilation kernels for FP4 GEMM — partial fix for cudnn backend path |
| sglang | [#11737](https://github.com/sgl-project/sglang/pull/11737) | merged (old) | Support CUTLASS FP4 kernel in SM120 — earlier partial fix, superseded by later breakage |

### What each backend does on SM120 today

| Backend | MoE FP4 on SM120 | Notes |
|---------|-----------------|-------|
| `triton` | Silent garbage output | No SM120 FP4 kernel implementation |
| `flashinfer_trtllm` | Crashes at GEMM execution | SM100 cubins downloaded from cloud cache (`_sm100f`), binary incompatible with SM120 |
| `flashinfer_cutlass` | Silent NaN/zeros | CUTLASS FP4 returns all zeros on SM120 (confirmed by flashinfer #2577) |

**There is no working NVFP4 MoE backend for SM120 today.** The patches in this document allow the server to progress further through startup, but the GEMM itself produces incorrect results or crashes.

### What a complete fix requires

1. **In flashinfer:** Fix the CUTLASS FP4 GEMM to return correct results on SM120 (tracked in #2577), OR publish SM120 cubins in the TRT-LLM gen kernel cache
2. **In SGLang/sgl-kernel:** Wire the fixed CUTLASS FP4 path through the MoE layer dispatch for SM120 (extend PR #17421 to cover SM120, not just SM121)
3. **The `flashinfer_trtllm` path** requires SM120 cubins in the online cache — this is the deeper fix that depends on NVIDIA/flashinfer publishing them

---

## Summary of All Patched Files

| File | Change |
|------|--------|
| `flashinfer/jit/fused_moe.py` | `supported_major_versions=[10]` → `[10, 12]` |
| `sglang/srt/layers/moe/fused_moe_triton/layer.py` | `correction_bias` cast to `bfloat16` when input is `float16` |
| `flashinfer/data/csrc/trtllm_fused_moe_kernel_launcher.cu` line ~385 | `ICHECK_EQ(major, 10)` → `ICHECK(major == 10 \|\| major == 12)` |
| `flashinfer/data/csrc/trtllm_fused_moe_kernel_launcher.cu` line ~1222 | `ICHECK_EQ(device_props[0], 10)` → `ICHECK(... == 10 \|\| ... == 12)` |

All `.venv` patches are in the installed flashinfer package and will be lost on reinstall.
The `layer.py` patch is in the SGLang dev install and survives reinstall as long as the repo is not reset.

---

## Updates

### 2026-02-21 - SGLang SM120 NVFP4 Still Broken (NaN Bug Open)

A comprehensive review of the SGLang issue tracker as of 2026-02-21 confirms the following state:

**What has been fixed in SGLang (merged Oct–Jan):**
- PR [#11737](https://github.com/sgl-project/sglang/pull/11737) — CUTLASS FP4 kernel extended to SM120 (`is_blackwell()` check, `nvfp4_scaled_mm_kernels.cu`, `nvfp4_blockwise_moe.cu`)
- PR [#11708](https://github.com/sgl-project/sglang/pull/11708) — `dsv3_fused_a_gemm` kernel bypassed on SM120 (produces zeros); `flashinfer.fp4_quantize` used as workaround
- PR [#14842](https://github.com/sgl-project/sglang/pull/14842) — TRTLLM MHA auto-selection now routes SM120 to FlashInfer instead of crashing
- PR [#16283](https://github.com/sgl-project/sglang/pull/16283) — TRT AllReduce Fusion restricted to SM90/SM100 only
- PR [#18546](https://github.com/sgl-project/sglang/pull/18546) — KV cache scale loading for NVFP4 checkpoints (our `k_scale`/`v_scale` KeyError)

**What is still broken:**
- [Issue #18954](https://github.com/sgl-project/sglang/issues/18954) (opened 2026-02-18, **currently open**) — NVFP4 models produce **NaN outputs** on RTX PRO 6000 (SM120) with `--fp4-gemm-backend flashinfer_cutlass`. This is the same FlashInfer CUTLASS zeros/NaN bug tracked upstream as FlashInfer [#2577](https://github.com/flashinfer-ai/flashinfer/issues/2577). This is exactly our hardware. **SGLang is not usable for NVFP4 on SM120 today.**
- [PR #18314](https://github.com/sgl-project/sglang/pull/18314) (draft, 2026-02-05) — NVFP4 KV cache for SM120/SM100 attention backends, not yet merged.

**Would SGLang be faster than vLLM if it worked?**

Potentially yes. SGLang generally shows better MoE throughput due to:
- DeepGEMM integration and more aggressive MoE kernel tuning
- Better prefix cache implementation
- Overlap scheduling (prefill/decode overlap)

However, vLLM's `VLLM_CUTLASS` backend is working correctly on SM120 today, while SGLang remains broken. Revisit when issue #18954 is resolved.

---

### 2026-02-21 - Upstream KV Cache Scale Fix

SGLang upstream **commit 33c33a7de** ([#18546](https://github.com/sgl-project/sglang/pull/18546)) addresses the KV cache scale loading issue for NVFP4 models.

**What this fixes:**
- The reported `KeyError: 'layers.N.self_attn.qkv_proj.k_scale'` when loading GLM-4.7-NVFP4 checkpoints
- Improved `weight_utils.py` to handle ModelOpt-style scale names that are stored under `k_proj`/`v_proj` but expected under `attn` level

**To apply:**
```bash
cd /home/alex/mllm/sglang
git pull
```

This upstream fix should eliminate the need for manual patching of checkpoint loading. However, the fundamental SM120 FP4 MoE backend issue remains unresolved (see "What a complete fix requires" section above for ongoing blockers).

---

## Version Timeline Analysis

### SGLang Releases

| Version | Release Date | Notes |
|---------|--------------|-------|
| v0.5.6 | 2025-12-02 | Pinned to `flashinfer_python==0.5.3` in requirements |
| v0.5.7 | 2026-01-01 | |
| v0.5.8 | 2026-01-23 | |
| v0.5.8.post1 | 2026-02-05 | |
| Current main | 2026-02-21 | Latest commits, includes upstream KV cache fix |

### flashinfer-python Releases

| Version | Release Date | Notes |
|---------|--------------|-------|
| 0.4.1 | 2025-10-14 | |
| **0.5.3** | 2025-11-20 | **Last known working version** for SM120 FP4 (used by SGLang v0.5.6) |
| 0.5.0 | 2025-11-02 | |
| 0.6.0rc1 | 2025-12-18 | Release candidates for 0.6.0 |
| 0.6.0rc2 | 2025-12-20 | |
| 0.6.0 | 2026-01-08 | **Regression: CUTLASS FP4 starts returning zeros, TRTLLM introduces SM100-only cubins** |
| 0.6.1 | 2026-01-14 | |
| 0.6.2 | 2026-01-23 | |
| 0.6.3 | 2026-02-06 | |
| 0.6.4 | 2026-02-19 | Latest |

### Correlation with Tengyunw Success

The Tengyunw/GLM-4.7-NVFP4 model card was updated on **2025-12-26**, reporting successful inference on 8x RTX 5090s with:

```bash
python3 -m sglang.launch_server --model-path GLM-4.7-NVFP4/ \
    --quantization modelopt_fp4 --tp 8 --attention-backend flashinfer
```

**Timeline context:**
- 2025-12-02: v0.5.6 released with flashinfer 0.5.3
- 2025-11-20 → 2025-12-18: flashinfer 0.5.x (stable) → 0.6.0rc1 (regression window)
- **2025-12-26**: Tengyunw benchmark performed (would use SGLang between v0.5.6 and v0.5.7, before flashinfer 0.6.0 release)
- 2026-01-08: flashinfer 0.6.0 released (regression point)

**Conclusion:** Tengyunw's success was achieved with flashinfer 0.5.x, likely via the CUTLASS FP4 path that later broke in 0.6.0.

---

## Alternative GLM-4.7-NVFP4 Model Cards

### Tengyunw/GLM-4.7-NVFP4

Model card includes SGLang-specific patches and deployment instructions:

**Required patch in `modelopt_quant.py` (~line 1637):**
```python
# Comment out weight scale validation:
#assert (
#    weight_scale.shape[assert_dim] % 16 == 0
#), f"Expected {name}_weight_scale.dim({assert_dim}) to be divisible by 16"
#assert (
#    weight_scale.dtype == torch.float8_e4m3
#), f"{name} Weight Blockscale must be represented as FP8-E4M3"
```

**Deploy command:**
```bash
python3 -m sglang.launch_server --model-path GLM-4.7-NVFP4/ \
    --quantization modelopt_fp4 --tp 8 --attention-backend flashinfer
```

**Key differences from this report:**
- Uses `--attention-backend flashinfer` (this report: crashes, uses `TRITON_ATTN`)
- Patch targets `modelopt_quant.py` assertion (vs. flashinfer `.cu` kernel patches)
- Focuses on weight scale validation format differences

### Model Format Differences

The Tengyunw patch suggests their NVFP4 quantization produces weight scales in a format that doesn't strictly match SGLang's FP8-E4M3 expectations. This could indicate:
- Different quantization calibration datasets/methods
- Slightly different ModelOpt version/modelopt_nvfp4 parameters

Both models (`Salyut1/GLM-4.7-NVFP4` and `Tengyunw/GLM-4.7-NVFP4`) are NVFP4 but may have internal format differences affecting how they're loaded and validated.
