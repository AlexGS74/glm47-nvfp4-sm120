# GLM-5 NVFP4 Recipes
*#glm-5 | RTX6kPRO Discord | Feb-Mar 2026*

See also: [GLM-5 NVFP4 Thread Summary](./glm5-nvfp4-thread-summary.md)

---

## Recipe 1: Initial Working Config (Feb 16-17, 2026)

**Model:** lukealonso/GLM-5-NVFP4 (no MTP)  ~50 tok/s at 0 context

```bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=PHB
export NCCL_ALLOC_P2P_NET_LL_BUFFERS=1
export NCCL_MIN_NCHANNELS=8
export OMP_NUM_THREADS=8
export SAFETENSORS_FAST_GPU=1

python3 -m sglang.launch_server \
  --model lukealonso/GLM-5-NVFP4 \
  --served-model-name glm-5 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --trust-remote-code \
  --tp 8 \
  --mem-fraction-static 0.95 \
  --max-running-requests 8 \
  --kv-cache-dtype bf16 \
  --quantization modelopt_fp4 \
  --attention-backend flashinfer \
  --moe-runner-backend flashinfer_cutlass \
  --disable-custom-all-reduce \
  --enable-flashinfer-allreduce-fusion \
  --host 0.0.0.0 \
  --port 5000 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}'
```

NOTE: Use bf16 KV cache only. fp8_e4m3 produces garbled output.
NCCL P2P hangs: add iommu=pt (amd_iommu=pt on AMD) to kernel command line.

---

## Recipe 2: Pinned Docker Setup (Feb 28, 2026)

**Model:** lukealonso/GLM-5-NVFP4 (no MTP)  0ctx=44tok/s | 15k=30tok/s

```bash
docker pull lmsysorg/sglang:dev-cu13
docker run -it --rm --cpuset-cpus "0-63" \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt:/mnt/ --ipc=host --shm-size=8g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus all --network host \
  lmsysorg/sglang:dev-cu13 bash
pip install --upgrade transformers
```

```bash
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS \
NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 NCCL_MIN_NCHANNELS=8 OMP_NUM_THREADS=8 \
SAFETENSORS_FAST_GPU=1 python3 -m sglang.launch_server \
  --model-path lukealonso/GLM-5-NVFP4 --tp 8 --trust-remote-code \
  --attention-backend flashinfer --moe-runner-backend flashinfer_cutlass \
  --kv-cache-dtype bf16 --tool-call-parser glm47 --reasoning-parser glm45 \
  --quantization modelopt_fp4 --disable-custom-all-reduce \
  --enable-flashinfer-allreduce-fusion --mem-fraction-static 0.9 \
  --cuda-graph-max-bs 8 --host 0.0.0.0 --port 5000 \
  --served-model-name glm-5 --max-running-requests 8 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}'
```
---

## Recipe 3: MTP + EAGLE ~ 100 tok/sec (Mar 1, 2026) RECOMMENDED

**Model:** festr2/GLM-5-NVFP4-MTP  ~100 tok/s | 60-80 tok/s long ctx | 39-57 tok/s at 200k

```bash
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS \
NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 NCCL_MIN_NCHANNELS=8 OMP_NUM_THREADS=8 \
SAFETENSORS_FAST_GPU=1 python3 -m sglang.launch_server \
  --model-path /mnt/GLM-5-NVFP4-MTP-FP8 --tp 8 --trust-remote-code \
  --attention-backend flashinfer --moe-runner-backend flashinfer_cutlass \
  --kv-cache-dtype bf16 --tool-call-parser glm47 --reasoning-parser glm45 \
  --quantization modelopt_fp4 --disable-custom-all-reduce \
  --enable-flashinfer-allreduce-fusion --mem-fraction-static 0.9 \
  --cuda-graph-max-bs 8 --host 0.0.0.0 --port 5000 \
  --served-model-name glm-5 --max-running-requests 8 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

Why EAGLE works: MTP head is bf16 same as DSv3.2. SGLang runs GLM-5 as DSv3.1 model.
FA2 used in prefill ignoring MLA (backwards compatible via lightning indexer).
sm120 stuck on FA2 sm89, missing TMEM/tcgen05 and FA4 support.

---

## Recipe 4: High-Concurrency Config (Mar 4, 2026 Latest)

Changes: --max-running-requests 64, --cuda-graph-max-bs 32

```bash
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS \
NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 NCCL_MIN_NCHANNELS=8 OMP_NUM_THREADS=8 \
SAFETENSORS_FAST_GPU=1 python3 -m sglang.launch_server \
  --model-path /mnt/GLM-5-NVFP4-MTP --tp 8 --trust-remote-code \
  --attention-backend flashinfer --moe-runner-backend flashinfer_cutlass \
  --kv-cache-dtype bf16 --tool-call-parser glm47 --reasoning-parser glm45 \
  --quantization modelopt_fp4 --disable-custom-all-reduce \
  --enable-flashinfer-allreduce-fusion --mem-fraction-static 0.9 \
  --cuda-graph-max-bs 32 --host 0.0.0.0 --port 5000 \
  --served-model-name glm-5 --max-running-requests 64 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

---

## Important Notes

Model selection:
- lukealonso/GLM-5-NVFP4: no MTP weights, cannot use speculative decoding
- festr2/GLM-5-NVFP4-MTP: Festr quant with MTP weights, required for EAGLE

MTP weights warning (vLLM PR #35548 by MatthewBonan):
Serving lukealonso with MTP loads garbage (torch.empty()) into MTP layers.
Zero acceptance rates. Always use festr2/GLM-5-NVFP4-MTP for EAGLE.

vLLM status (Mar 2026): Does NOT work with GLM-5 on SM120.
FlashInfer MLA requires qk_nope_head_dim=128 but GLM-5 has 192. Use SGLang only.

Hardware: Min 8x RTX PRO 6000. Power: 400W/card in prefill, peaks 600-640W.

Long context (150k+): Grimulkan DCA patches: 6 tok/s -> 32 tok/s at 150k.

---

## Performance Summary

| Config | Context | Speed |
|---|---|---|
| Recipe 1/2 (no MTP) | 0 | ~50 tok/s |
| Recipe 1/2 (no MTP) | 15k | ~30 tok/s |
| Recipe 3/4 (MTP+EAGLE) | 0 | ~100 tok/s |
| Recipe 3/4 (MTP+EAGLE) | 100k | ~2x baseline |
| Recipe 3/4 (MTP+EAGLE) | 200k | 39-57 tok/s |
| Recipe 3/4 + DCA | 150k | 32 tok/s |
| Recipe 3/4 (MTP+EAGLE) | long | 60-80 tok/s |

MMLU-Pro: NVFP4 = 87.3% vs benchmark 87.7% (gap: -0.4%)
Author: Festr, March 2026
