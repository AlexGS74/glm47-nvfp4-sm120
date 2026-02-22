# GLM-4.7 AWQ Benchmark — vLLM 0.15.1, SM120 (4× RTX PRO 6000 Blackwell)

**Quant:** QuantTrio/GLM-4.7-AWQ (`awq_marlin` kernel)
**Date:** 2026-02-22
**Config:** TP=4, DTYPE=half, ATTENTION=FLASHINFER, max_model_len=200000, max_num_batched_tokens=32768
**Benchmark:** `vllm bench serve`, random dataset, input=512 tok, output=256 tok, 32 prompts per run
**Tool:** direct to vLLM port 30000 (bypass buster-ripper), OpenAI `/v1/chat/completions`

---

## No MTP (SPEC_TOKENS=0)

| Concurrency | System tok/s | Per-req tok/s | TTFT median | TTFT P99 |
|-------------|-------------|---------------|-------------|----------|
| 1  | 68  | 68  | 41 ms  | 54 ms  |
| 2  | 110 | 55  | 72 ms  | 95 ms  |
| 4  | 187 | 47  | 102 ms | 107 ms |
| 8  | 280 | 35  | 115 ms | 162 ms |
| 16 | 438 | 27  | 155 ms | 163 ms |

System tok/s = concurrency × (1000 / TPOT_ms)

## MTP SPEC_TOKENS=1

_TODO — rerun with SPEC_TOKENS=1_

---

## Comparison vs NVFP4 (no MTP)

NVFP4 benchmarks were run at `max_num_batched_tokens=16384` with `enable_thinking=false`.

| Concurrency | NVFP4 sys tok/s | AWQ sys tok/s | AWQ vs NVFP4 |
|-------------|-----------------|----------------|--------------|
| 1  | 48  | 68  | **+42%** |
| 2  | 62  | 110 | **+77%** |
| 4  | 118 | 187 | **+58%** |
| 8  | 203 | 280 | **+38%** |

AWQ is consistently faster. Likely reasons:
- `awq_marlin` fused dequant+matmul is very efficient on SM120
- NVFP4 benchmark was at half the batch token budget (16384 vs 32768)
- NVFP4 uses TRITON_ATTN; AWQ uses FLASHINFER (faster attention on SM120 for non-FP4 quants)

**Theoretical ceiling:** 22 GB weights/GPU × 4 GPUs at 1.28 TB/s → ~58 tok/s at C=1 for memory-bound decode.
AWQ at 68 tok/s exceeds this estimate — `awq_marlin` kernel likely achieves compute-bound decode at C=1
(Marlin packs weights efficiently enough to overlap compute and memory).

## MTP results (SPEC_TOKENS=1)

76.3% acceptance rate observed in production (Mean acceptance length: 1.76).
**Known issue:** tool call streaming parser corrupted with SPEC_TOKENS > 1.

| Concurrency | Sys tok/s | TPOT median ms | TTFT median ms | TTFT P99 ms |
|-------------|-----------|----------------|----------------|-------------|
| 1  | 57  | 17.61 | 50 ms  | 67 ms  |
| 2  | 101 | 19.83 | 96 ms  | 121 ms |
| 4  | 170 | 23.49 | 113 ms | 135 ms |
| 8  | 293 | 27.30 | 137 ms | 169 ms |
| 16 | 448 | 35.74 | 178 ms | 258 ms |

## No MTP vs MTP=1 comparison

| Concurrency | No MTP tok/s | MTP=1 tok/s | Delta |
|-------------|-------------|-------------|-------|
| 1  | 68  | 57  | **-16%** |
| 2  | 110 | 101 | **-8%**  |
| 4  | 187 | 170 | **-9%**  |
| 8  | 280 | 293 | +5%      |
| 16 | 438 | 448 | +2%      |

**Conclusion:** MTP is slower at the concurrency levels Claude Code uses (1–2 agents).
76% acceptance rate sounds promising but draft token generation overhead dominates at low
concurrency. MTP only breaks even at C=8+. Default set to `SPEC_TOKENS=0`.

---

## Notes

- `awq_marlin` auto-detected and enabled (vLLM log: "Detected that the model can run with awq_marlin")
- MTP breaks tool calls at SPEC_TOKENS > 1; not worth fixing since MTP is slower anyway
- SPEC_TOKENS=0: all features work, throughput already better than NVFP4
- Thinking renders natively in Claude Code with `--served-model-name claude-opus-4-5-20251001`
- AWQ faster than NVFP4 due to software maturity — vLLM SM120 NVFP4 path uses TRITON_ATTN
  and CUTLASS workarounds; AWQ uses FlashInfer + optimised Marlin kernel
