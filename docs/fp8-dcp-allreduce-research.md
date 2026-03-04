# FP8 Compressed AllReduce Research — SM120 / PCIe

Research into FP8 compressed communication for tensor parallelism on 4x RTX PRO 6000 (SM120, PCIe 5.0, ASUS WRX90E-SAGE).
Date: March 3, 2026

---

## TL;DR

No FP8 compressed allreduce is available today in SGLang or vLLM on NVIDIA GPUs.
NCCL 2.27+ supports it natively on SM90+, but neither framework exposes a flag to use it.
PCIe systems would benefit the most (2x on L40 per Flash Communication paper), but
the infrastructure isn't wired up yet.

---

## Current State by Framework

### SGLang

| Feature | Status | SM120 |
|---|---|---|
| FP8 compressed allreduce | On Q1 2026 roadmap ([#12780](https://github.com/sgl-project/sglang/issues/12780)), not shipped | N/A |
| FlashInfer allreduce fusion | Fuses allreduce+RMSNorm+quant | **Broken** — only SM90/SM100 ([#15650](https://github.com/sgl-project/sglang/issues/15650)) |
| GEMM+AllReduce overlap | FP16 done ([#9058](https://github.com/sgl-project/sglang/pull/9058)), FP8 planned | Likely SM90/SM100 only |
| `--disable-flashinfer-cutlass-moe-fp4-allgather` | FP4 before allgather in EP | EP only, not TP allreduce |
| `--enable-torch-symm-mem` | Symmetric memory for NCCL | SM90+ required |

### vLLM

| Feature | Status | SM120 |
|---|---|---|
| Custom allreduce | **Disabled** on >2 PCIe GPUs ([#13719](https://github.com/vllm-project/vllm/issues/13719)) | Falls back to NCCL |
| FP8 compressed allreduce | Not available on NVIDIA | N/A |
| AMD QuickReduce (FP8/INT4) | Integrated, ROCm only | **Not applicable** |

### NCCL

- NCCL 2.27+ supports native FP8 reductions (`ncclFloat8e4m3`, `ncclFloat8e5m2`)
- Minimum SM90 (Hopper) — SM120 (Blackwell) is supported
- FP32 accumulators on PCIe, FP16 on NVLink Switch (NVLS)
- CUDA 12.8 required for symmetric kernel FP8 on Blackwell
- **Neither SGLang nor vLLM invoke this for TP allreduce**

---

## Hardware Requirements

FP8 compressed allreduce is a software optimization — it does NOT require a PCIe switch
or Turin box. However, those improve baseline bandwidth:

| Topology | GPU-to-GPU path | Benefit from FP8 compress |
|---|---|---|
| ASUS Sage (WRX90E) | GPU → CPU root complex → GPU | **High** — PCIe is the bottleneck |
| Turin dual-socket | GPU → PCIe switch → GPU | Medium — better baseline |
| NVLink (H100/B200) | Direct GPU-to-GPU | Low — NVLink isn't the bottleneck |

The Flash Communication paper confirms: PCIe systems benefit most from compressed allreduce.

---

## Academic Research

### Flash Communication (arXiv 2412.04964)

"Reducing Tensor Parallelization Bottleneck for Fast LLM Inference"

- **L40 (PCIe):** Up to **2.06x TTFT reduction** with INT4 compressed allreduce
- **A100 SXM (NVLink):** 1.19x — much less because NVLink isn't the bottleneck
- INT4 allreduce kernel: **3.18x latency reduction** vs ring allreduce for volumes >64MB
- Uses asymmetric quantization: INT4 (ReduceSum) + INT8 (AllGather)
- **Not integrated into SGLang or vLLM**

### FlashCommunication V2 (arXiv 2508.03760)

Extends to arbitrary bit widths. Also standalone, not integrated.

---

## GLM-4.7 Blackwell Optimization (SGLang)

[Issue #17526](https://github.com/sgl-project/sglang/issues/17526) tracks GLM-4.7 Blackwell work:

- `flashinfer_trtllm` MoE backend: 10% gain — **SM100 only, not SM120**
- FP8 KV buffer kernel fusion: 3% gain
- CuteDSL FP4 GEMM: 4% gain
- No FP8 compressed allreduce mentioned

### Festr's Plan (from RTX6kPRO Discord, March 3 2026)

1. Port SGLang's 2 dedicated SM120 MLA kernels (FlashInfer BF16 + XQA FP8) into vLLM
2. Wire GLM-4.7 to use them in non-DSA mode (vLLM blocks this for SM120 currently)
3. Fix NVFP4 MoE GEMM crash (works in SGLang, wiring issue in vLLM)
4. Normal FA2 for prefill
5. MTP should work once above is done

This is the most impactful near-term path — dedicated MLA kernels explain the
60 tok/s (vLLM/Triton) vs 100 tok/s (SGLang/FlashInfer) gap.

---

## Recommendations

1. **Near-term:** Wait for Festr's MLA kernel port to vLLM, or use SGLang for GLM-4.7
2. **Medium-term:** Watch SGLang Q1 2026 roadmap for FP8 allreduce shipping
3. **Long-term:** Flash Communication integration would give ~2x on our PCIe topology
4. **Not useful:** AMD QuickReduce (wrong vendor), flashinfer allreduce fusion (SM120 broken)

---

## Sources

- [SGLang 2026 Q1 Roadmap — #12780](https://github.com/sgl-project/sglang/issues/12780)
- [SGLang GLM Blackwell Optimization — #17526](https://github.com/sgl-project/sglang/issues/17526)
- [SGLang AllReduce Overlap — #8728](https://github.com/sgl-project/sglang/issues/8728)
- [SGLang FlashInfer SM120 Bug — #15650](https://github.com/sgl-project/sglang/issues/15650)
- [vLLM Custom AllReduce PCIe — #13719](https://github.com/vllm-project/vllm/issues/13719)
- [Flash Communication — arXiv 2412.04964](https://arxiv.org/abs/2412.04964)
- [FlashCommunication V2 — arXiv 2508.03760](https://arxiv.org/abs/2508.03760)
- [NCCL 2.27 FP8 Support](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/)
- [QuickReduce (AMD ROCm)](https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce/README.html)
