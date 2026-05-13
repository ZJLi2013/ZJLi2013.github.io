**EN** | [CN](./readme-cn.md)

# Porting spconv to ROCm: How FlyDSL Replaced 20,000 Lines of CUDA Codegen

## Why This Matters

[spconv](https://github.com/traveller59/spconv) is the de facto standard library for 3D sparse convolution. Nearly every major point cloud model -- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [CenterPoint](https://github.com/tianweiy/CenterPoint) -- depends on it for autonomous driving perception, 3D object detection, and scene understanding.

**Yet spconv has never supported AMD GPUs.**

The root cause isn't spconv itself. spconv delegates all heavy computation to [cumm](https://github.com/FindDefinition/cumm), a custom CUDA GEMM library. cumm uses [pccm](https://github.com/FindDefinition/pccm) -- a Python-based C++ code generator -- to emit architecture-specific CUDA kernels targeting NVIDIA tensor cores (Volta HMMA, Turing WMMA, Ampere MMA). This entire compilation pipeline is inextricably bound to `nvcc`, CUDA shared memory semantics, and NVIDIA-specific matrix instructions.

```
spconv -> cumm (20K+ lines Python codegen -> C++ -> nvcc -> CUDA kernel)
               |-- pccm (Python C++ Code Manager)
```

A naive HIPIFY approach was ruled out early: cumm isn't a C++ library with CUDA API calls -- it's a *compiler backend* that generates C++ strings from Python. The pccm framework, tensorview runtime, and NVRTC JIT compilation chain would all need fundamental rewrites.

Instead, we took a different path: replace the entire codegen stack with [FlyDSL](https://github.com/ROCm/FlyDSL), a Python kernel DSL for AMD GPUs. The result is **spconv running on MI300X with 28/28 tests passing**, and the GEMM backend reduced from 20,000 lines of codegen to ~130 lines of Python.

**Repos**: [cumm-rocm](https://github.com/ZJLi2013/cumm-rocm) | [spconv-rocm](https://github.com/ZJLi2013/spconv_rocm)

---

## Key Takeaway 1: FlyDSL as a Drop-in for the CUDA/CUTLASS Ecosystem

### The Abstraction Alignment

cumm was designed to mirror CUTLASS -- the canonical high-performance GEMM template library on NVIDIA. It implements the same three-level tiling hierarchy:

```
Grid:   tile_shape = (M_tile, N_tile, K_tile)     -- per threadblock
Warp:   warp_tile_shape = (M_warp, N_warp, K_warp) -- per warp
Thread: tensorop = (M_op, N_op, K_op)              -- hardware MMA instruction
```

FlyDSL maps to this hierarchy one-to-one, but targeting AMD MFMA instructions:

| cumm / CUTLASS (NVIDIA) | FlyDSL (AMD) | What it controls |
|:---|:---|:---|
| `tile_shape = (M, N, K)` | `fx.flat_divide(A, (BLOCK_M, BLOCK_K))` | Grid-level tiling |
| `warp_tile_shape` + `thread_map` | `fx.make_tiled_mma(atom, thr_layout)` | Wave/warp mapping |
| `tensorop = (16, 8, 8)` (Turing MMA) | `fx.MFMA(16, 16, 32, fx.Float16)` | Hardware matrix instruction |
| pccm `__shared__` + staging | `fx.get_dyn_shared()` + XOR swizzle + LDS pipeline | Shared memory management |
| `gen_shuffle_params()` static compile | `@flyc.jit` + `autotune.py` | Kernel compilation + tuning |
| NVRTC dynamic compile | MLIR -> ROCDL -> GPU binary JIT | Runtime compilation |

This structural alignment means the migration isn't a rewrite -- it's a *re-targeting*. The tiling logic, dataflow patterns, and performance mental model carry over directly.

### From 20K Lines to 130 Lines

The original cumm `gemm/` directory contained over 20,000 lines of Python code that generated C++ CUDA kernels -- tile iterators, warp-level MMA wrappers, shared memory pipelines, epilogue accumulators, and architecture-specific dispatch for Simt, Volta, Turing, and Ampere.

The ROCm replacement, `FlyDSLGemmTuner`, is ~130 lines:

```python
# cumm-rocm: FlyDSLGemmTuner (simplified)
# Input: matmul(C[M,N], A[M,K], B[N,K], beta)

# 1. Get FlyDSL's recommended tile config for this shape
cfg = get_default_kwargs(m, n, k)

# 2. Auto-shrink tiles to fit gfx942 LDS (64KB)
while smem_estimate(cfg) > 65536:
    shrink(cfg)  # TILE_M -> TILE_N -> TILE_K

# 3. One function call replaces the entire codegen pipeline
hgemm_splitk_(C, A, B, hgemm_kwargs=cfg)
```

Behind this simplicity, FlyDSL's `hgemm_splitk` kernel implements the full MFMA GEMM pipeline: multi-stage LDS buffering, XOR bank-conflict-free swizzle, split-K for tall-skinny matrices, and explicit instruction scheduling (`sched_mfma`, `sched_vmem`, `sched_dsrd`). These are the same techniques cumm implements via codegen -- but expressed as ~500 lines of readable Python instead of generated C++ strings.

### Why This Works: The "Compiler Layer" Is No Longer Needed

cumm's complexity was the answer to a 2019 question: *"How do you write high-performance GEMM on NVIDIA GPUs without CUTLASS's C++ template complexity?"* The answer was pccm -- a Python meta-programming framework that generates C++ code.

In 2026, JIT kernel DSLs (FlyDSL, Triton) have made the code generation layer unnecessary. The kernel is the Python function. There's no intermediate C++ stage, no nvcc invocation, no separate compilation step:

```
2019:  Python (algorithm) -> pccm (codegen) -> C++ -> nvcc -> GPU
2026:  Python (algorithm) -> FlyDSL (JIT) -> MLIR -> GPU
```

The analogy: going from hand-written assembly to a compiler. The intermediate representation disappears, but the same hardware capabilities are accessible.

### Agent-Friendly Development

A practical benefit worth highlighting: FlyDSL kernels are ordinary `.py` files. JIT compilation means edit-run cycles are instant. Errors produce Python tracebacks, not pages of C++ template errors. This makes kernel development significantly more accessible to AI code agents -- the implicit GEMM kernel described later was iteratively developed and debugged through an experiment-driven workflow where each version was tested remotely on MI300X hardware.

---

## Key Takeaway 2: spconv Now Runs on ROCm

### The Full Stack

The end-to-end port required migrating two libraries and writing new HIP kernels:

```
Original:  Python -> pccm (codegen) -> C++ -> nvcc -> GPU     (152 files, 33K lines)
ROCm:      Python -> torch.mm/FlyDSL + HIP kernel -> GPU     (~1.5K lines)
```

| Component | Original spconv | ROCm port |
|:---|:---|:---|
| GEMM | pccm C++ codegen (~20K lines) | FlyDSL `hgemm_splitk` / `torch.mm` |
| Indice pairs (neighbor lookup) | CUDA hash table via pccm | Self-contained HIP kernel (688 lines) |
| Implicit GEMM (fused conv) | pccm CUDA codegen | FlyDSL MFMA fused kernel |
| Python binding | pccm -> nvcc -> .so | `torch.utils.cpp_extension.load()` JIT |

**Code stats**: 152 files changed, 759 insertions, 32,771 deletions.

### Test Coverage

28 tests pass on MI300X (gfx942), covering:

- **Forward**: SubMConv3d, SparseConv3d (stride=2), 1x1 conv, SparseSequential, ToDense
- **Backward**: gradient propagation through SubM, strided conv, Sequential
- **Numerical correctness**: sparse conv vs dense conv comparison, bias verification
- **Pooling**: SparseMaxPool3d, GlobalMaxPool, GlobalAvgPool
- **Edge cases**: 1D/2D conv, large batch (4), single point, dilation, indice_key reuse
- **Transpose**: SparseConvTranspose3d (5->10 upsample)
- **Encoder-Decoder**: typical 3D detection UNet (SubM->BN->ReLU->Strided)

### The Indice Pairs Problem

The most interesting engineering challenge wasn't the GEMM -- it was the *indice pairs* computation. In sparse convolution, before any GEMM happens, you need to find which input voxels contribute to which output voxels. This neighbor-lookup-to-rulebook step is the foundational operation.

The original spconv uses pccm-generated CUDA hash table kernels with thrust sort -- highly optimized but deeply coupled to the pccm framework. Our initial pure-Python CPU implementation (using `dict`) was 200-500ms for 50K voxels. The PyTorch-native GPU version (`sort` + `searchsorted`) brought it down to ~15ms.

The final solution was a self-contained HIP kernel:

```
spconv/csrc_hip/
  hash_table.h              (95 lines)  -- Murmur3 + open-addressing GPU hash table
  indice_pairs_kernel.hip   (470 lines) -- SubM 2D-grid + Regular conv 3-stage pipeline
  indice_pairs_api.cpp      (123 lines) -- PyTorch C++ extension binding
```

| Implementation | SubM N=50K, 3x3x3 | vs original CUDA |
|:---|:---:|:---:|
| CPU Python dict | ~200-500 ms | -- |
| GPU PyTorch native | 15.2 ms | ~7x slower |
| **GPU HIP kernel** | **0.6 ms** | **~3x faster** |
| Original CUDA hash table | ~2 ms | 1x |

The HIP kernel outperforms the original CUDA implementation because it avoids the pccm/tensorview overhead and uses a cleaner hash table design with hipCUB for sort/unique.

---

## Key Takeaway 3: Iterative Kernel Optimization via FlyDSL

Beyond replacing cumm's GEMM, we used FlyDSL to build a *new* kernel that doesn't exist in the original spconv: a fused implicit GEMM that merges the 27-position gather-GEMM-scatter loop into a single kernel launch.

This kernel went through 8 iterative versions, each tested on MI300X:

| Version | Architecture | Key Change | N=5K, 16->32 | N=20K, 64->128 |
|:---|:---|:---|:---:|:---:|
| V4 | per-thread scalar | LUT-based gather | 350 us | 5338 us |
| V5 | per-thread scalar | LDS accumulator | 202 us | 10954 us (LDS full) |
| V6 | per-thread scalar | Output column tiling | 203 us | 3702 us |
| V7 | per-thread scalar | Register accumulator | 195 us | 2254 us |
| V8a | wave/block tiled | Tile ownership | 121 us | 1515 us |
| V8c | MFMA fragment | `mfma_f32_16x16x4` | 99 us | 1665 us |

The progression from V4 to V8a shows a 2.9x improvement for small channels and 3.5x for large channels -- achieved entirely through algorithmic restructuring in Python, without touching any C++ code.

The final integrated results on a 4-layer 3D detection backbone:

| Path | Backbone time | vs baseline |
|:---|:---:|:---:|
| Native (at::mm loop) | 11.2 ms | 1.0x |
| **Implicit GEMM (FlyDSL)** | **8.87 ms** | **1.26x** |

---

## Performance: Where We Are and What's Left

### Current Performance (MI300X gfx942)

**Single-layer forward breakdown:**

| Config | indice_pairs | indice_conv | Full forward |
|:---|:---:|:---:|:---:|
| SubM 3x3x3, C=32, N=5K | 0.50 ms | 0.37 ms (implicit) | 0.82 ms |
| SubM 3x3x3, C=32->64, N=20K | 0.52 ms | 0.55 ms (implicit) | 1.02 ms |
| Conv 3x3x3 s=2, C=64->128, N=20K | 0.93 ms | 1.57 ms (native) | 2.44 ms |

**4-layer backbone (typical 3D detection):**

```
SubM(16->32) -> stride2(32->64) -> SubM(64->64) -> stride2(64->128)
B=2, N=20K/batch, spatial=[200, 200, 10]
```

| Metric | Time |
|:---|:---:|
| Forward | 8.87 ms |
| Forward + Backward | ~27 ms |

For 3D detection inference, ~9ms forward with implicit GEMM approaches real-time requirements (~110 FPS).

### Performance Optimization Journey

The full optimization path from the initial pure-Python prototype:

| Optimization | Component | Impact |
|:---|:---|:---|
| HIP kernel for indice pairs | indice_pairs | **27-41x** vs PyTorch native |
| C++ for-loop sink | indice_conv loop | **1.5-2x** (eliminated 27x Python->GPU sync) |
| FlyDSL implicit GEMM | indice_conv GEMM | **2.5-3.7x** for SubM (fused gather+GEMM+scatter) |
| Backbone total | end-to-end | Initial ~40ms -> **8.87ms** (~4.5x) |

### Known Limitations

**1. Large-channel implicit GEMM regression.** For Conv layers with large channels (e.g., 64->128), the implicit GEMM path is ~6% slower than the native `at::mm` loop. The current `crossk_pf_xor_bk32` kernel's preprocessing overhead outweighs the fusion benefit when individual GEMMs are large enough. The runtime dispatcher handles this by falling back to native path when `c_in * c_out` exceeds the crossover threshold.

**2. Inference only for implicit GEMM.** The fused kernel currently lacks autograd support. Training automatically falls back to the native gather-GEMM-scatter path, which fully supports backward. Adding `torch.autograd.Function` wrappers is planned.

**3. No GEMM autotune yet.** The `FlyDSLGemmTuner` uses rule-based tile selection (FlyDSL's recommended config + LDS capacity shrinking). Runtime autotuning -- compiling 3-5 tile variants and benchmarking per shape, similar to Triton's `@autotune` -- could yield 10-30% additional gains.

**4. fp32 only for implicit GEMM.** The fused kernel currently operates in f32. Extending to fp16/bf16 would reduce memory traffic and leverage MFMA's higher throughput modes.

**5. No CUDA vs ROCm direct comparison yet.** We have not benchmarked the original CUDA spconv on equivalent NVIDIA hardware. The numbers above are ROCm-only; cross-platform performance comparison is future work.

---

## Architecture Reflection: Why 40x Less Code?

The original spconv + cumm is ~33K lines. The ROCm port is ~1.5K lines. This 20x reduction isn't because we cut corners -- it's because **90% of the original code exists to solve "how to run GEMM on a GPU," not "how sparse convolution works."**

| Original component | Lines | ROCm replacement | Why it's gone |
|:---|:---:|:---|:---|
| pccm codegen (Python -> C++ strings) | ~20K | FlyDSL `hgemm_splitk_()` | JIT DSL replaces code generation |
| Multi-arch kernels (Simt/Volta/Turing/Ampere) | ~5K | Single `hgemm_splitk` | AMD MFMA unified across gfx942/950/1250 |
| tensorview (C++ tensor wrapper) | ~3K | `torch.Tensor` | ROCm PyTorch handles all tensor ops |
| nvcc/NVRTC build system | ~2K | None | FlyDSL JIT, no compile step |
| C++ extension bindings | ~1K | `torch.utils.cpp_extension.load()` | JIT compilation for HIP kernels |

The insight: in 2019, building a high-performance GEMM required a custom compilation pipeline. In 2026, JIT kernel DSLs absorb that complexity. What remains is the *algorithm* -- sparse indexing, gather/scatter patterns, fusion strategies -- which is inherently concise.

---

## What's Next

| Priority | Optimization | Expected gain |
|:---|:---|:---|
| P1 | GEMM runtime autotune (Triton-style) | 10-30% |
| P1 | fp16/bf16 implicit GEMM | 1.5-2x memory bandwidth reduction |
| P2 | Implicit GEMM backward (training support) | Enables full training on ROCm |
| P2 | Grouped GEMM (batch 27 small GEMMs) | ~1.5x for native path |
| P3 | Mask/LUT fusion into `get_indice_pairs` | Eliminate 20-50% implicit GEMM preprocessing |
| P3 | MFMA multistage double buffer (V9 kernel) | Compute-bound regime for large channels |

---

## Conclusion

This port demonstrates two things:

**First, FlyDSL is a practical replacement for the CUDA/CUTLASS ecosystem.** The abstraction alignment between FlyDSL and CUTLASS-style GEMM design -- tiling hierarchy, warp/wave mapping, shared memory pipelines, hardware matrix instructions -- means that expertise in CUDA kernel development transfers directly. The 20K-to-130-line reduction isn't about losing capabilities; it's about JIT compilation absorbing the code generation machinery. FlyDSL kernels can be iteratively developed and tuned in Python with immediate feedback -- a qualitative improvement over the codegen-compile-test cycle.

**Second, spconv on ROCm is real.** 28 tests pass. The 4-layer backbone runs in 8.87ms on MI300X. The implicit GEMM kernel, built from scratch in FlyDSL, delivers 2.5-3.7x speedup over the native gather-GEMM-scatter path for SubM convolutions. Performance gaps remain (large-channel regression, no autotune, inference-only fusion), but the functional foundation is complete -- and the optimization path is clear.

For the point cloud ecosystem, this means OpenPCDet, MMDetection3D, CenterPoint, and similar frameworks can begin targeting AMD GPUs. The port is open-source:

- **cumm-rocm**: [github.com/ZJLi2013/cumm-rocm](https://github.com/ZJLi2013/cumm-rocm) (branch: `rocm`)
- **spconv-rocm**: [github.com/ZJLi2013/spconv_rocm](https://github.com/ZJLi2013/spconv_rocm) (branch: `rocm`)
- **FlyDSL**: [github.com/ROCm/FlyDSL](https://github.com/ROCm/FlyDSL)
