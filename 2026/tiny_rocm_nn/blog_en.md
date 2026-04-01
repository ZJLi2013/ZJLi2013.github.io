**EN** | [CN](./blog_cn.md)

# Porting tiny-cuda-nn to AMD ROCm: A Deep Dive into GPU Architecture Differences

## Why This Matters

In 2022, Thomas Müller at NVIDIA Research released [Instant NeRF](https://nvlabs.github.io/instant-ngp/), compressing Neural Radiance Field (NeRF) training from hours to seconds. Behind this breakthrough was a framework called [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) — by fusing an entire multi-layer perceptron into a single GPU kernel (Fully Fused MLP), it achieved 5-20x faster training than PyTorch for small networks.

The impact of tiny-cuda-nn extends far beyond NeRF. It has become foundational infrastructure for the 3D reconstruction and neural rendering ecosystem:

- **[Instant NGP](https://github.com/NVlabs/instant-ngp)** (SIGGRAPH 2022): Real-time neural graphics primitives supporting NeRF, SDF, and image representations
- **[Nerfstudio](https://github.com/nerfstudio-project/nerfstudio)**: The most popular NeRF development framework, with its high-performance backend relying on tiny-cuda-nn
- **[NerfAcc](https://github.com/KAIR-BAIR/nerfacc)**: General-purpose NeRF acceleration toolbox
- **[nvdiffrec](https://github.com/NVlabs/nvdiffrec)** (CVPR 2022 Oral): Extracting triangle meshes, materials, and lighting from images
- **3D Gaussian Splatting** and its variants: Many implementations use tiny-cuda-nn's hash encoding for accelerated scene representation

In generative AI, as 3D generation (text-to-3D, image-to-3D) gains momentum, efficient neural field training has become a critical bottleneck. The multiresolution hash encoding and fully fused MLP provided by tiny-cuda-nn are core components enabling real-time performance in these applications.

**Yet this entire ecosystem is locked to NVIDIA GPUs.**

tiny-cuda-nn is deeply coupled to NVIDIA's proprietary stack — CUTLASS (templated GEMM library), CUDA WMMA (tensor core interface), cuBLAS — none of which run on AMD hardware. This means all the 3D reconstruction and neural rendering projects listed above cannot run on high-performance datacenter GPUs like the AMD Instinct MI300X.

As AMD accelerates its presence in the AI training and inference market (MI300X is deployed across multiple supercomputers and cloud platforms), breaking this vendor lock-in has practical value for:
- **Researchers**: Hardware choice freedom, reducing dependence on a single vendor
- **Cloud providers**: Full 3D/NeRF workload support on AMD GPU instances
- **The AMD ecosystem**: Validating ROCm's capabilities and identifying gaps in high-performance GPU compute scenarios

This post documents the complete porting process of [tiny-rocm-nn](https://github.com/ZJLi2013/tiny-rocm-nn) — from API translation to the discovery and fix of two subtle GPU architecture bugs, along with surprising findings from performance analysis.

## The Porting Path

The port was carried out in two stages:

```
tiny-cuda-nn (CUTLASS + CUDA WMMA)
  → cublas branch (replaced CUTLASS with cuBLAS)
    → hipblas branch (hipBLAS + rocWMMA)  ← final
```

The first stage replaced CUTLASS (NVIDIA's templated GEMM library) with cuBLAS API calls, reducing porting complexity. The second stage translated cuBLAS → hipBLAS and CUDA WMMA → rocWMMA to complete the AMD adaptation.

The API-level translation was relatively straightforward: `cublasGemmEx` → `hipblasGemmEx`, `nvcuda::wmma` → `rocwmma`, `__shfl_sync` → `__shfl`. The real challenges lay much deeper.

## The Symptom: Training Crashes After 3000 Steps

After the port compiled and forward inference appeared correct, running the standard image-learning benchmark revealed:

| Step | CUDA (reference) | ROCm (ported) |
|------|-----------------|---------------|
| 0 | 9.45 | 9.45 |
| 1000 | ~0.05 (converged) | 0.112 (not converged) |
| 2000 | ~0.03 | 0.538 (diverging!) |
| 3000 | ~0.02 | 0.910 |
| 3100 | — | **NaN** (crashed) |

The loss dropped normally for the first 1000 steps, then reversed and climbed until exploding to NaN around step 3100. This "good then bad" pattern is particularly insidious — it suggests not a simple computation error, but a **slowly accumulating numerical corruption**.

## Bug #1: The Implicit Assumption of Wave64

### NVIDIA Warps vs AMD Wavefronts

The fundamental execution unit on NVIDIA GPUs is a 32-thread warp. AMD GPUs use 64-thread wavefronts (wave64). This difference appears simple but permeates every corner of low-level kernel code.

The fused MLP kernel in tiny-cuda-nn uses `int4` vectorized memory copies to transfer data between global and shared memory. The key pattern:

```cpp
// Each iteration, a warp (32 threads) loads int4 vectors covering 16 rows
// N_ITERS iterations cover 16 * N_ITERS rows
for (int i = 0; i < N_ITERS; ++i) {
    int row = cyclic_index(lane_id, i, WIDTH);
    shmem[row] = global[row];  // int4 vectorized copy
}
```

The `N_ITERS` calculation implicitly assumed a 32-thread warp. With a 64-thread wave:
- Each iteration covers **32 rows** (not 16)
- But `N_ITERS` wasn't adjusted accordingly
- Result: the upper half of the wave (threads 32-63) reads and writes **out-of-bounds addresses**

### Pinpointing the Issue

By writing fused-vs-unfused comparison tests, we found:
- Errors were **concentrated at batch indices 128-255**
- Maximum error occurred at `col=142`
- This matched exactly the wave's out-of-bounds write region

### The Fix

Replace hardcoded strides with wave-size-aware computation:

```cpp
constexpr uint32_t ROWS_PER_COPY_STEP = WAVE_SIZE / 2;  // 64/2=32
constexpr uint32_t COPY_N_ITERS = (16 * N_ITERS) / ROWS_PER_COPY_STEP;
```

The fix was applied to 6 locations in `fully_fused_mlp.cpp`. After fixing, all forward-pass tests passed (0 mismatches).

**But training still diverged.**

## Bug #2: rocWMMA Fragment Layout Incompatibility

### From Correct Forward to Broken Backward

The forward pass was fully correct, confirming Bug #1's fix worked. But training still hit NaN at ~3000 steps, meaning the **backward pass** had issues.

To rule out width-specific bugs, we trained with WIDTH=64 (the same config as the passing tests) — it also diverged at step 2200. The problem was systematic.

### Deep Comparison: CUDA vs rocWMMA Backward Kernels

The critical backward-pass operation compares matrix multiplication results (accumulator fragment) with forward activation values (matrix_a fragment) element-wise to implement ReLU backward:

```cpp
// CUDA WMMA: this code is correct
fragment<accumulator, 16,16,16, __half> acc;    // MMA result
fragment<matrix_a, 16,16,16, __half> fwd_act;   // forward activations
// element-wise: if fwd_act.x[t] > 0 then keep acc.x[t], else zero
warp_activation_backward(acc, fwd_act);
```

In CUDA WMMA, `accumulator.x[t]` and `matrix_a.x[t]` map to the **same matrix position**. This is an (undocumented) guarantee of NVIDIA's WMMA implementation.

**In rocWMMA, this guarantee does not hold.**

AMD's MFMA (Matrix Fused Multiply-Add) instruction uses **different register layouts** for input and output matrices. Element `t` of an `accumulator` fragment and element `t` of a `matrix_a` fragment correspond to **different row-column positions** in the matrix.

This means `warp_activation_backward` applied the ReLU gate to **wrong positions** — instead of correctly gating gradients, it randomly preserved or zeroed them. The corruption was gradual: each backward step introduced small gradient errors that accumulated until weights exploded around step 3000.

### The Fix: From Registers to Shared Memory

The solution was to abandon fragment-level element-wise operations and use shared memory as an intermediary:

```
MMA result (fragment)
  → store_matrix_sync → shared memory (deterministic row/col layout)
  → apply ReLU backward element-wise in shared memory
  → continue subsequent computation
```

Shared memory layout is explicit (`row * stride + col`), independent of any fragment internal mapping. The fix was applied to two locations: hidden-layer backward (`threadblock_layer`) and last-layer backward (`kernel_mlp_fused_backward`).

### Results After Fix

| Step | Before Fix | After Fix |
|------|-----------|-----------|
| 0 | 9.456 | 9.456 |
| 1000 | 0.112 → diverging | **0.053** |
| 3000 | NaN | **0.026** |
| 5000 | — | **0.020** |
| 10000 | — | **0.017** |

Loss decreased monotonically, no NaN through 10,000 steps. Training fully converged.

## Performance Analysis

Benchmark results on AMD Instinct MI300X (ROCm 6.4.3):

| Configuration | Training (ms/step) | Throughput |
|:---|:---:|:---:|
| **tiny-rocm-nn** | 7.55 | 34.7M samples/sec |
| PyTorch FP32 | 9.47 | 27.7M samples/sec |
| PyTorch FP16 (AMP) | 6.05 | 43.3M samples/sec |
| PyTorch FP16 + torch.compile | 5.89 | 44.5M samples/sec |

tiny-rocm-nn is **1.25x faster** than PyTorch FP32, but slower than PyTorch AMP.

### The hipBLAS FP16 Compute Experiment

A natural optimization idea was switching hipBLAS compute type from FP32 to FP16, since the original tiny-cuda-nn uses CUTLASS with FP16 accumulation. We tested this:

- **Accuracy**: FP16 compute produced identical loss convergence (step 10000 loss: 0.018 vs 0.017)
- **Performance**: FP16 compute was **2.2x slower** (21.1 ms/step vs 9.65 ms/step)

The reason: AMD MFMA hardware **natively accumulates in FP32**. Setting hipBLAS compute type to `HIPBLAS_R_16F` forced a non-MFMA fallback path. This differs from CUTLASS's FP16 accumulation — CUTLASS controls accumulation precision at the software tile level, while hipBLAS compute type directly affects hardware path selection.

### Root Cause of the Performance Gap

The performance gap between tiny-rocm-nn and PyTorch AMP is not about numerical precision — it's **architectural**:

- **tiny-cuda-nn (NVIDIA)**: Uses CUTLASS to fuse GEMM + activation into a single kernel (epilogue fusion), eliminating extra kernel launches and memory round-trips
- **tiny-rocm-nn (AMD)**: Uses hipBLAS for GEMM (external library call) with activation in a separate kernel, adding launch overhead and memory traffic

## Lessons Learned

### 1. Wave64 Is Not Just "Multiply by 2"

AMD's 64-thread wavefronts differ from NVIDIA's 32-thread warps in ways that go far beyond thread count. Any code that computes indices based on warp/wave size — particularly vectorized memory copies — needs careful review. These bugs only trigger at specific batch ranges and may evade standard tests.

### 2. Fragment Layout Is an Implementation Detail, Not an Interface Guarantee

In CUDA WMMA, `accumulator` and `matrix_a` fragments happen to share the same element layout, but this is not part of the API specification. Relying on this undocumented behavior leads to silent failures on rocWMMA (and potentially future CUDA architectures). The safe approach is to route any cross-fragment-type element-wise operations through shared memory.

### 3. "Good Then Bad" Loss Curves Signal Gradient Corruption

If training looks normal for the first 1000 steps then diverges, it usually means the backward pass has a small but systematic numerical error. Correct forward pass does not imply correct backward pass — they may exercise entirely different code paths and memory layouts.

### 4. Performance Tuning Requires Understanding Hardware Paths

hipBLAS with `HIPBLAS_R_16F` compute type is paradoxically slower on AMD because it bypasses MFMA. The same API parameter can trigger drastically different execution paths on different hardware. Performance optimization must be grounded in understanding the target hardware, not API-level analogies.

## Future Optimization Directions

1. **composable_kernel replacing hipBLAS**: AMD's [composable_kernel](https://github.com/ROCm/composable_kernel) library provides CUTLASS-equivalent epilogue fusion capabilities, enabling GEMM + activation to be fused into a single kernel
2. **hipBLASLt**: Offers more flexible algorithm selection for the small-matrix GEMMs common in MLP training
3. **Fused kernel tuning**: Optimize rocWMMA kernel shared memory access patterns, occupancy, and wave scheduling
4. **Stream-level parallelism**: Explore overlapping GEMM, activation, and optimizer kernels across multiple streams

## Conclusion

Porting tiny-cuda-nn to ROCm is far more than a mechanical API translation. Both core bugs stemmed from fundamental differences between NVIDIA and AMD GPU architectures — warp/wavefront size and tensor core register layouts. These differences are entirely invisible at the API level and can only be understood and fixed by diving into the GPU execution model.

The significance of this port is that it demonstrates **even GPU kernels deeply bound to NVIDIA tensor cores can run correctly on AMD MFMA hardware**. rocWMMA and hipBLAS provide sufficient capabilities for the port, though some implicit assumptions (fragment layout compatibility) require explicit handling.

For the 3D reconstruction and neural rendering community, tiny-rocm-nn opens the possibility of running Instant NGP, Nerfstudio, and similar projects on AMD GPUs. The current version is fully validated for functional correctness; performance optimization — particularly epilogue fusion via composable_kernel — is the focus of the next phase.

The project is open-source on [GitHub](https://github.com/ZJLi2013/tiny-rocm-nn). Contributions and feedback are welcome.
