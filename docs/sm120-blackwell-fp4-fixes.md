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

---

## FlashInfer Issue Tracking

The FlashInfer project has numerous open issues and PRs related to SM120/Blackwell FP4 support. Below are the most relevant ones as of 2026-02-21:

### Critical SM120/FP4 Issues

| Issue/PR | Title | State | Date | Relevance |
|----------|-------|-------|------|-----------|
| [#2577](https://github.com/flashinfer-ai/flashinfer/issues/2577) | NVFP4 mm_fp4 GEMM broken on SM120 (RTX PRO 6000 Blackwell) - all backends fail | open | 2026-02-18 | **HIGHEST** - Our exact hardware, CUTLASS zeros, TRTLLM crashes |
| [#2598](https://github.com/flashinfer-ai/flashinfer/pull/2598) | feat: add CuTe DSL flash attention backend for SM120 GPUs | open (PR) | 2026-02-20 | Adds CuTe DSL attention for SM120 |
| [#2589](https://github.com/flashinfer-ai/flashinfer/issues/2589) | FlashInfer API for sparsity updated cubins in nvfp4 moe | open | 2026-02-19 | NVFP4 MoE cubin API |
| [#2561](https://github.com/flashinfer-ai/flashinfer/pull/2561) | feat: SM120 standard attention kernels + CuTe DSL backend | open (PR) | 2026-02-13 | SM120 attention kernels |
| [#2555](https://github.com/flashinfer-ai/flashinfer/issues/2555) | SM120 attention kernels exist but are blocked by wiring issues | open | 2026-02-13 | Wiring bugs prevent SM120 usage |
| [#2553](https://github.com/flashinfer-ai/flashinfer/pull/2553) | filter out runtime kernels for fp4 gemm | open (PR) | 2026-02-13 | FP4 GEMM runtime kernel filtering |
| [#2545](https://github.com/flashinfer-ai/flashinfer/pull/2545) | Draft - Dynamic shape for cuDNN FP4 GEMM to reduce graph build overhead | open (PR) | 2026-02-12 | cuDNN FP4 improvements |
| [#2540](https://github.com/flashinfer-ai/flashinfer/pull/2540) | feat: cute dsl mmfp4 for blackwell | merged | 2026-02-20 | CuTe DSL MMFP4 backend (SM100) |
| [#2520](https://github.com/flashinfer-ai/flashinfer/pull/2520) | Support NVFP4 KV cache decode on SM120 | open (PR) | 2026-02-08 | SM120 NVFP4 KV cache |
| [#2517](https://github.com/flashinfer-ai/flashinfer/pull/2517) | Improve small size performance in cutedsl fp4 | open (PR) | 2026-02-07 | CuTe DSL FP4 optimization |
| [#2496](https://github.com/flashinfer-ai/flashinfer/issues/2496) | [Perf] mxfp4 quantize kernel is slow | open | 2026-02-04 | MXFP4 performance |
| [#2466](https://github.com/flashinfer-ai/flashinfer/issues/2466) | Integrate cuteDSL fp4 dense gemm into flashinfer | open | 2026-02-02 | CuTe DSL FP4 integration |
| [#2443](https://github.com/flashinfer-ai/flashinfer/pull/2443) | Add cute-dsl backends to mxfp[8,4]_quantization | open (PR) | 2026-01-30 | MXFP8/4 quantization integration |
| [#2402](https://github.com/flashinfer-ai/flashinfer/pull/2402) | feat: Add sparsity cubins for sm90 | merged | 2025-12-24 | Sparsity support (added to current) |

### Notable Issues from 2025

| Issue/PR | Title | State | Date | Relevance |
|----------|-------|-------|------|-----------|
| [#2373](https://github.com/flashinfer-ai/flashinfer/issues/2373) | FI trtllm_fp4_block_scale_moe interface core dump and output mismatch | open | 2026-01-19 | NVFP4 MoE core dumps |
| [#2294](https://github.com/flashinfer-ai/flashinfer/issues/2294) | [Feature Request] Support NVFP4 KV Cache Kernel on SM120 through XQA | open | 2026-01-06 | SM120 NVFP4 attention |
| [#2263](https://github.com/flashinfer-ai/flashinfer/pull/2263) | feat: Add FP8/NVFP4 quant fusion for MNNVL Allreduce | open (PR) | 2025-12-24 | FP8/NVFP4 fusion |
| [#2166](https://github.com/flashinfer-ai/flashinfer/issues/2166) | sm120 rtx 6000 | open | 2025-12-03 | Early SM120 RTX 6000 tracking |
| [#2077](https://github.com/flashinfer-ai/flashinfer/issues/2077) | MoE autotune print a lot failed kernel on SM120 | open | 2025-11-11 | SM120 MoE kernels failing |
| [#1741](https://github.com/flashinfer-ai/flashinfer/issues/1741) | mm_fp4 regression (need 0.2s cpu time per run) | open | 2025-09-20 | FP4 performance regression |
| [#1632](https://github.com/flashinfer-ai/flashinfer/issues/1632) | CuteDSL grouped gemm kernel write nan into non-masked position | open | 2025-09-03 | CuteDSL FP4 NaN |

### Summary of FlashInfer Status

**Active Development:**
- Multiple PRs adding SM120 support (#2561, #2598, #2520)
- CuTe DSL MMFP4 backend recently merged (#2540)
- NVFP4 attention and KV cache improvements in progress

**Known Blockers (for us):**
- Issue #2577 is the definitive blocker - all FP4 GEMM backends fail on SM120
- PR #2555 documents that SM120 kernels exist but aren't wired into the runtime
- CUTLASS FP4 returns zeros (NaN propagation) as tracked in SGLang #18954 / flashinfer #2577
- TRTLLM downloads SM100-only cubins that cannot run on SM120

**What needs to happen:**
1. Fix CUTLASS FP4 to return correct values on SM120 (tracked in #2577)
2. Wire existing SM120 kernels into attention/MLA backends (#2555)
3. Publish SM120 cubins for TRTLLM NVFP4 MoE path
4. Integrate and test CuTe DSL MMFP4 for SM120 (currently SM100-only per #2540 description)

**No merged fixes for SM120 NVFP4 + GEMM** as of 2026-02-21. Most SM120 work is in draft or open PR states.

---

## TensorRT-LLM Investigation — 2026-02-22

Investigated TRT-LLM as an alternative serving backend. GLM-4.7 (full MoE) on SM120 with NVFP4 sits at the intersection of three areas each with active blockers. **Not attempted — would fail for reasons outside our control.**

### SM120 / NVFP4 support in TRT-LLM

- **NVFP4 dense GEMM on SM120**: Works since v0.20+ (confirmed working on RTX 5090, ~8300 tok/s reported). SM120 gate was removed in v0.20.0rc3+ after [Issue #5018](https://github.com/NVIDIA/TensorRT-LLM/issues/5018).
- **NVFP4 MoE on SM120**: Active blocker — [Issue #7484](https://github.com/NVIDIA/TensorRT-LLM/issues/7484) `RuntimeError: Only SM100 is supported by FP4 block scale MOE` on RTX 6000 PRO (SM120). Confirmed open/assigned as of 2026-02-22. This is our exact GPU + quant + architecture combo.
- **NVFP4 KV cache on SM120**: Not supported — [Issue #10241](https://github.com/NVIDIA/TensorRT-LLM/issues/10241) opened Jan 2026, added to roadmap but no committed timeline.
- **FP4 CUTLASS GEMM shared-memory on GB10/SM121**: [Issue #11368](https://github.com/NVIDIA/TensorRT-LLM/issues/11368) — SM120 (RTX 5090/PRO 6000) does NOT have this constraint (it affects DGX Spark only). Not a blocker for us.

### GLM-4.7 model support in TRT-LLM

- **GLM-4.7 full model (MoE)**: Open feature request [Issue #10462](https://github.com/NVIDIA/TensorRT-LLM/issues/10462). Not supported.
- **GLM-4.7-Flash** (smaller, non-MoE variant): Added in v1.3.0rc3/rc4 (early Feb 2026) via the AutoDeploy/PyTorch backend path. PR #11150 merged. Still in RC — not in a stable release.
- **GLM-4 (original)**: Listed in the classic TRT-engine support matrix under `examples/chatglm/`, but that path is deprecated in favor of AutoDeploy.
- **NVFP4 MoE checkpoint assertion bug**: [Issue #6569](https://github.com/NVIDIA/TensorRT-LLM/issues/6569) — `AssertionError: w1_weight_scale_2 != w3_weight_scale_2` when loading ModelOpt FP4 MoE checkpoints (reported on GLM-4.5, Qwen3-235B). Workaround PR proposed; resolution unclear. Would likely affect GLM-4.7-NVFP4 as well.

### Anthropic API endpoint

TRT-LLM exposes OpenAI-compat endpoints only (`/v1/chat/completions`, `/v1/completions`). No `/v1/messages` Anthropic endpoint exists and no feature request is open. Would require a proxy layer (LiteLLM or similar) instead of buster-ripper's native Anthropic path — additional complexity with no clear benefit.

### TRT-LLM issue tracker — active blockers to monitor

| Issue | Description | Status |
|-------|-------------|--------|
| [#7484](https://github.com/NVIDIA/TensorRT-LLM/issues/7484) | `Only SM100 supported by FP4 block scale MOE` on SM120 | **Open** — our primary blocker |
| [#10462](https://github.com/NVIDIA/TensorRT-LLM/issues/10462) | GLM-4.7 full model (non-Flash) support request | **Open** |
| [#10241](https://github.com/NVIDIA/TensorRT-LLM/issues/10241) | NVFP4 KV cache not available for SM120 in trtllm-gen | **Open** — roadmapped |
| [#6569](https://github.com/NVIDIA/TensorRT-LLM/issues/6569) | `AssertionError: w1_weight_scale_2 != w3_weight_scale_2` loading ModelOpt FP4 MoE checkpoints | Open — workaround proposed |
| [#11114](https://github.com/NVIDIA/TensorRT-LLM/issues/11114)/[#11115](https://github.com/NVIDIA/TensorRT-LLM/issues/11115) | GLM-4.7-Flash AutoDeploy performance not yet optimized (cuda graph, sharding) | **Open** |

### Verdict

Do not attempt TRT-LLM until at minimum #7484 (FP4 block scale MoE on SM120) and #10462 (GLM-4.7 full model) are both closed. vLLM AWQ is the correct path today.

**Revisit trigger:** Both #7484 and #10462 closed in a stable TRT-LLM release.

---

## Status Update — 2026-02-28

### Summary

No blockers have been resolved. All critical issues remain open. Some attention-related PRs are progressing in flashinfer but none have merged and none address the MoE FP4 GEMM root cause.

### Blocker Status — No Change

| Issue | State | Summary |
|---|---|---|
| flashinfer [#2577](https://github.com/flashinfer-ai/flashinfer/issues/2577) | **Open** | NVFP4 mm_fp4 GEMM broken on SM120 — still the root blocker |
| sglang [#18954](https://github.com/sgl-project/sglang/issues/18954) | **Open** | NVFP4 NaN on SM120 — confirmed FlashInfer CUTLASS backend issue; Triton more reliable than CUTLASS/DeepGemm on SM120 |
| TRT-LLM [#7484](https://github.com/NVIDIA/TensorRT-LLM/issues/7484) | **Open** | FP4 block scale MoE SM100-only — no SM120 fix, still assigned |
| TRT-LLM [#10462](https://github.com/NVIDIA/TensorRT-LLM/issues/10462) | **Open** | GLM-4.7 full model — gibberish output even on B200 hardware; root cause unclear, still investigating |

### Movement — PRs In Progress, None Merged

| PR | State | Notes |
|---|---|---|
| flashinfer [#2598](https://github.com/flashinfer-ai/flashinfer/pull/2598) | Open | CuTe DSL flash attention for SM120 — JIT kernels, validated on SM121, fallback to FA2 |
| flashinfer [#2561](https://github.com/flashinfer-ai/flashinfer/pull/2561) | Open | SM120 standard attention kernels (fmha_v2, BF16/FP16) — validated on DGX Spark, awaiting 7 maintainer approvals |
| flashinfer [#2520](https://github.com/flashinfer-ai/flashinfer/pull/2520) | Open | NVFP4 KV cache decode on SM120 — functional, code review issues (variable scope, copy-paste errors in tests) |
| sglang [#18314](https://github.com/sgl-project/sglang/pull/18314) | Draft | NVFP4 KV cache for SM120 — critical review issues (unconditional Triton forcing, in-place list mutations, code duplication), CI not triggered |

### Notable: Workaround Found for flashinfer #2577

A workaround was identified in #2577: using `.float()` conversion when calculating scales fixes some backend paths. Does **not** fix MoE FP4 specifically. Worth testing for non-MoE FP4 GEMM paths.

### Conclusion

**vLLM AWQ remains the correct path today.** The flashinfer attention PRs (#2561, #2598) are the closest to landing but address attention, not the MoE FP4 GEMM. No change to the serving stack is warranted until flashinfer #2577 closes.

**Revisit trigger:** flashinfer #2577 closed, OR sglang #18954 closed with a confirmed fix.

### Script Updates — 2026-02-28

Adopted the following env vars from a confirmed-working FP8 reference recipe into
`scripts/serve_glm47_nvfp4_vllm.sh` and `scripts/serve_glm47_awq.sh`:

| Variable / Flag | Scripts | Reason |
|---|---|---|
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | both | Safer than `fork` with CUDA contexts; avoids deadlocks |
| `VLLM_SLEEP_WHEN_IDLE=1` | both | GPU drops to P8 between requests — free power saving |
| `CUDA_DEVICE_ORDER=PCI_BUS_ID` | both | GPU order matches nvidia-smi; avoids surprises with TP |
| `--compilation-config '{"level":3,"cudagraph_mode":"full"}'` | both | Higher throughput via CUDA graph; overridable via `COMPILATION_CONFIG` |
| `VLLM_MARLIN_USE_ATOMIC_ADD=1` | AWQ only | AWQ uses Marlin kernels; atomic add fixes correctness on Blackwell |
