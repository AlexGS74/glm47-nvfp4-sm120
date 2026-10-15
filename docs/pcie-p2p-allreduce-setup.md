# PCIe P2P & AllReduce Setup — RTX PRO 6000 Blackwell

How to enable forced P2P and PCIe allreduce on RTX/workstation GPUs.
Data center GPUs (A100, H100, B200) have P2P enabled by default and don't need this.

Date: April 5, 2026 — updated October 15, 2026 after live re-verification on the
4x RTX PRO 6000 box (see "Verified state" at the bottom).
Source: [vllm-project/vllm#39040](https://github.com/vllm-project/vllm/pull/39040)

---

## Problem

RTX and workstation GPUs have P2P disabled by default in the NVIDIA driver.
Without it, PCIe allreduce falls back to slower NCCL paths, and SGLang's
`--enable-pcie-oneshot-allreduce` crashes with:

```
RuntimeError: invalid argument (pcie_allreduce.cu:321)
```

## Setup

### 1. Create modprobe config

The known-good filename on this box is `nvidia-p2p-override.conf` (chosen to
match what `llm_decode_bench`'s P2P check suggests); any `.conf` name works.

```bash
sudo tee /etc/modprobe.d/nvidia-p2p-override.conf << 'EOF'
# Force PCIe P2P on RTX/workstation GPUs for allreduce support.
# Required for: SGLang --enable-pcie-oneshot-allreduce,
#               vLLM VLLM_ENABLE_PCIE_ALLREDUCE=1
# Source: https://github.com/vllm-project/vllm/pull/39040
options nvidia NVreg_RegistryDwords="ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"
EOF
```

If a config already exists (e.g. with `NVreg_UseKernelSuspendNotifiers`), merge
the options onto one line or use a separate file — modprobe reads all `.conf`
files. An earlier `nvidia-p2p.conf` variant here was superseded and renamed to
`nvidia-p2p.conf.disabled`.

### 2. Reload driver or reboot

Option A — reload (kills all GPU processes):
```bash
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
```

Option B — reboot:
```bash
sudo reboot
```

### 3. Verify

```bash
# Check registry params are set
cat /proc/driver/nvidia/params | grep RegistryDwords
# Should show: ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;
#              GrdmaPciTopoCheckOverride=1;EnableResizableBar=1

grep EnableResizableBar /proc/driver/nvidia/params   # expect: 1

# Check GPU topology
nvidia-smi topo -m
# NODE = direct-attach PCIe (expected for RTX PRO 6000 on WRX90E-SAGE).
# NOTE: topo keeps showing NODE even with the override active — that string
# is physical topology and never changes. The *behavior* changes: the driver
# forces direct GRDMA peer access across the root boundary.
```

## Kernel parameters (same box, measured relevant to serving perf)

| Param | Value | Why |
|---|---|---|
| `iommu=pt` | kernel cmdline | enables GPU P2P — measured **+40% throughput** (58→81 tok/s NVFP4 decode) |
| `pcie_aspm=off` | kernel cmdline | PCIe power management off — stability/latency for GPU links |
| CPU governor | `performance` | was `powersave` by default |
| swappiness | `0` | avoid paging model/KV pages out |

## Using PCIe AllReduce

### SGLang

Add to docker run or launch command:
```
--enable-pcie-oneshot-allreduce
--enable-pcie-oneshot-allreduce-fusion
```

Note: Do NOT combine with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` —
this breaks CUDA IPC memory handle exchange used by custom allreduce.

Caveat observed on this box: oneshot allreduce crashed with
`invalid argument (pcie_allreduce.cu:321)` on driver 595.45 + cu130 during IPC
graph buffer benchmark — retry after driver update or on cu132 builds.

### vLLM

```bash
export VLLM_ENABLE_PCIE_ALLREDUCE=1
```

## Registry Keys Explained

| Key | Value | Purpose |
|-----|-------|--------|
| `ForceP2P` | `0x11` | Force-enable PCIe peer-to-peer transfers |
| `RMForceP2PType` | `1` | Use BAR1 P2P (works without IOMMU changes) |
| `RMPcieP2PType` | `2` | PCIe P2P transfer type |
| `GrdmaPciTopoCheckOverride` | `1` | Skip PCI topology check that blocks P2P on consumer GPUs |
| `EnableResizableBar` | `1` | Resizable BAR (also needs BIOS support) — VRAM mapping performance |

## Verified state (October 15, 2026)

On the 4x RTX PRO 6000 Max-Q box this recipe runs on:

- `/etc/modprobe.d/nvidia-p2p-override.conf` active; old `nvidia-p2p.conf.disabled`
- `/proc/driver/nvidia/params`: `EnableResizableBar: 1`, full 5-key RegistryDwords live
- cmdline: `iommu=pt pcie_aspm=off`
- First verified working 2026-06-24: the override unblocked b12x oneshot allreduce
  (previously rejected tensors with `fully_connected=False` on this PCIe topology,
  forcing NCCL host-memory fallback). Two earlier "hardware-bound" diagnoses were
  wrong — the empirical P2P test overruled them.

## Topology Reference

4x RTX PRO 6000 on ASUS WRX90E-SAGE (single NUMA node, PCIe Gen5):

```
     GPU0  GPU1  GPU2  GPU3
GPU0   X   NODE  NODE  NODE
GPU1  NODE   X   NODE  NODE
GPU2  NODE  NODE   X   NODE
GPU3  NODE  NODE  NODE   X
```

All GPUs connect through the PCIe host bridge within the same NUMA node.
No PCIe switch — each GPU on its own root port.
