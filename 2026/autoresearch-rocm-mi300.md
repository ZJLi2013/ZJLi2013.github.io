# Running Karpathy's Autoresearch on AMD MI300 with ROCm 6.4.3

*2026-03-09 | David Z.J. Lee*

Karpathy recently released [autoresearch](https://github.com/karpathy/autoresearch) -- a minimal autonomous LLM training framework that lets an AI agent iterate on model architecture, hyperparameters, and training code overnight. The idea: give the agent a 5-minute training budget per experiment, let it modify `train.py`, run, check `val_bpb`, keep or discard, repeat.

The catch? It only supports NVIDIA GPUs out of the box (H100, Flash Attention 3 via the `kernels` package, CUDA-specific `torch.compile`). This post documents porting it to a single **AMD Instinct MI300** using `rocm/pytorch:6.4.3`.

---

## Environment

| Item | Value |
|------|-------|
| GPU | AMD Instinct MI300 (gfx942) |
| Docker Image | `rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0` |
| ROCm | 6.4.43484 |
| PyTorch | 2.6.0 (ROCm build) |

## What Needed to Change

Three things broke when moving from CUDA to ROCm. All changes are in `train.py` only -- `prepare.py` works unmodified.

### 1. Flash Attention 3 -> PyTorch SDPA

The original code pulls FA3 from the `kernels` pip package (NVIDIA-only). Replaced with `F.scaled_dot_product_attention`, which on ROCm automatically dispatches to **AOTriton** -- AMD's optimized triton-based attention kernel:

```python
IS_ROCM = hasattr(torch.version, 'hip') and torch.version.hip is not None

if IS_ROCM:
    q = q.transpose(1, 2)  # B,T,H,D -> B,H,T,D for SDPA
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    y = y.transpose(1, 2).contiguous().view(B, T, -1)
else:
    y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
    y = y.contiguous().view(B, T, -1)
```

**Trade-off:** SDPA doesn't support the sliding `window_size` parameter, so the SSSL (short-short-short-long) window pattern degrades to full causal attention on all layers. This changes training dynamics slightly.

### 2. `torch.compile` Disabled

PyTorch 2.6.0 on ROCm has a strict dtype check in `lerp_()` that breaks inside compiled graphs -- bfloat16 tensors mixed with float32 scalar weights. This is fixed in PyTorch 2.9+. For now:

```python
_maybe_compile = torch.compile(dynamic=False, fullgraph=True) if not IS_ROCM else lambda fn: fn
```

This disables compilation for both the MuonAdamW optimizer fused kernels and the model itself.

### 3. `lerp_` dtype casting

Even in eager mode, PyTorch 2.6.0 on ROCm requires explicit dtype matching for `lerp_`:

```python
exp_avg.lerp_(grad, (1 - beta1_t).to(dtype=dtype))
```

## Results

The default config: 50.3M parameter GPT, 8 layers, 512 dim, 5-minute wall-clock training budget.

### Final Metrics

| Metric | Value |
|--------|-------|
| **val_bpb** | **1.520835** |
| training time | 302.2 s |
| total time (incl. startup) | 399.3 s |
| peak VRAM | ~97 GB / 192 GB |
| MFU | 3.18% |
| throughput | ~161K tok/sec |
| total tokens | 54.5M |
| steps | 104 |

### Loss Curve

| Step | Progress | Loss (EMA) | LR mult |
|------|----------|------------|---------|
| 0 | 0% | 9.01 | 1.00 |
| 10 | 0% | 6.80 | 1.00 |
| 20 | 10% | 5.91 | 1.00 |
| 40 | 31% | 5.32 | 1.00 |
| 57 | 50% | 4.98 | 1.00 |
| 70 | 64% | 4.74 | 0.72 |
| 90 | 86% | 4.45 | 0.29 |
| 103 | 100% | 4.32 | 0.01 |

Loss dropped from 9.0 to 4.3 monotonically. Cosine warm-down kicked in at 50%.

## Why MFU is Low (and What to Do About It)

**3.18% MFU** is well below what MI300 can do. The main bottleneck is running in eager mode without `torch.compile`. On CUDA with PyTorch 2.9.1, compilation fuses ops and eliminates kernel launch overhead -- easily a 3-10x improvement.

The node already has `rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1` pulled. Re-running with that image should:
- Re-enable `torch.compile`
- Fix the `lerp_` dtype issue natively
- Potentially bring MFU to 15-30%+

Other low-hanging fruit:
- **Increase `DEVICE_BATCH_SIZE`** -- only using ~97 GB of 192 GB HBM3
- **Restore sliding-window attention** -- once ROCm flash-attn supports window_size
- **Multi-GPU** -- the node has 2+ MI300 GPUs

## Reproducibility

```bash
# On an AMD MI300 node with Docker + ROCm drivers:
git clone https://github.com/karpathy/autoresearch.git
cd autoresearch

# Apply ROCm patches to train.py (see above), then:
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --shm-size=64g --ipc=host \
  -v $(pwd):/workspace/autoresearch \
  -v ~/.cache/autoresearch:/root/.cache/autoresearch \
  -e HIP_VISIBLE_DEVICES=0 \
  -w /workspace/autoresearch \
  rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0 \
  bash -c 'pip install matplotlib pyarrow rustbpe tiktoken && \
           python3 prepare.py --num-shards 10 && \
           python3 train.py'
```

## Takeaway

Porting autoresearch to ROCm was straightforward -- three targeted changes in `train.py`, no changes to the data pipeline or model architecture. The PyTorch SDPA + AOTriton path "just works" as a drop-in for Flash Attention on AMD. The main performance gap comes from the older PyTorch 2.6.0 lacking robust `torch.compile` support on ROCm, which is solved in 2.9.1.

Autoresearch is a neat idea. The next step: hook up the autonomous agent loop on MI300 and let it run overnight.

---

*[Back to index](../README.md)*
