
[EN](./blog_en.md) | **CN**

# tiny-rocm-nn: tiny-cuda-nn 的 ROCm 版本

## 为什么

2022 年，NVIDIA 研究院的 Thomas Müller 发布了 [Instant NeRF](https://nvlabs.github.io/instant-ngp/)，将神经辐射场（NeRF）的训练时间从数小时压缩到数秒。这一突破的背后，是一个名为 [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) 的底层框架——它通过将多层感知器完全融合到单个 GPU kernel 中（Fully Fused MLP），实现了比 PyTorch 快 5-20 倍的小型网络训练速度。

tiny-cuda-nn 的影响远超 NeRF 本身。它已经成为 3D 重建与神经渲染领域的核心基础设施：

- **[Instant NGP](https://github.com/NVlabs/instant-ngp)**（SIGGRAPH 2022）：实时神经图形原语，支持 NeRF、SDF、图像等多种表示
- **[Nerfstudio](https://github.com/nerfstudio-project/nerfstudio)**：最流行的 NeRF 开发框架，其高性能后端依赖 tiny-cuda-nn
- **[NerfAcc](https://github.com/KAIR-BAIR/nerfacc)**：通用 NeRF 加速工具箱
- **[nvdiffrec](https://github.com/NVlabs/nvdiffrec)**（CVPR 2022 Oral）：从图像提取三角网格、材质和光照
- **3D Gaussian Splatting** 及其变体：许多实现使用 tiny-cuda-nn 的哈希编码加速场景表示

在生成式 AI 领域，随着 3D 生成（text-to-3D、image-to-3D）成为热点，高效的神经场表示训练成为关键瓶颈。tiny-cuda-nn 提供的多分辨率哈希编码和全融合 MLP 是这些应用追求实时性的核心组件。

**然而，这整个生态完全锁定在 NVIDIA GPU 上。**

tiny-cuda-nn 深度绑定 NVIDIA 专有技术栈——CUTLASS（模板化 GEMM 库）、CUDA WMMA（张量核心接口）、cuBLAS——没有一个组件能在 AMD 硬件上运行。这意味着上述所有 3D 重建和神经渲染项目，在 AMD Instinct MI300X 这样的高性能数据中心 GPU 上无法使用。

随着 AMD 在 AI 推理和训练市场的布局加速（MI300X 已在多个超算和云平台部署），打破这一锁定对于：
- **研究者**：提供硬件选择自由度，降低对单一供应商的依赖
- **云服务商**：在 AMD GPU 实例上提供完整的 3D/NeRF 工作负载支持
- **AMD 生态**：验证 ROCm 在高性能 GPU 计算场景下的能力和局限

都具有实际价值。

本文记录了 [tiny-rocm-nn](https://github.com/ZJLi2013/tiny-rocm-nn) 的完整移植过程——从 API 替换到两个隐蔽的 GPU 架构差异 bug 的发现和修复，以及性能分析中的意外发现。

## 移植路径

移植分两步进行：

```
tiny-cuda-nn (CUTLASS + CUDA WMMA)
  → cublas 分支 (cuBLAS 替换 CUTLASS)
    → hipblas 分支 (hipBLAS + rocWMMA)  ← 最终版本
```

第一步将 CUTLASS（NVIDIA 的模板化 GEMM 库）替换为 cuBLAS API 调用，降低移植复杂度。第二步将 cuBLAS → hipBLAS、CUDA WMMA → rocWMMA，完成 AMD 适配。

API 层面的替换相对直接：`cublasGemmEx` → `hipblasGemmEx`，`nvcuda::wmma` → `rocwmma`，`__shfl_sync` → `__shfl`。真正的挑战隐藏在更深处。

## 症状：训练在 3000 步后崩溃

移植完成后，编译通过，前向推理看起来正常。但运行标准的图像学习 benchmark 时：

| Step | CUDA (参考) | ROCm (移植后) |
|------|-----------|--------------|
| 0 | 9.45 | 9.45 |
| 1000 | ~0.05 (收敛) | 0.112 (未收敛) |
| 2000 | ~0.03 | 0.538 (发散!) |
| 3000 | ~0.02 | 0.910 |
| 3100 | — | **NaN** (崩溃) |

Loss 在前 1000 步正常下降，然后突然反转上升，最终在 3100 步左右爆炸为 NaN。这种"先好后坏"的模式极其阴险——它暗示不是简单的计算错误，而是某种**缓慢积累的数值腐蚀**。

## Bug #1：Wave64 的隐式假设

### NVIDIA Warp vs AMD Wavefront

NVIDIA GPU 的基本执行单元是 32 线程的 warp。AMD GPU 则使用 64 线程的 wavefront（wave64）。这个差异看似简单，实则渗透到了底层 kernel 的每一个角落。

tiny-cuda-nn 的融合 MLP kernel 使用 `int4` 向量化内存拷贝来在全局内存和共享内存之间搬运数据。关键代码模式：

```cpp
// 每次迭代，一个 warp(32线程) 用 int4 加载覆盖 16 行
// N_ITERS 次迭代覆盖 16 * N_ITERS 行
for (int i = 0; i < N_ITERS; ++i) {
    int row = cyclic_index(lane_id, i, WIDTH);
    shmem[row] = global[row];  // int4 向量化拷贝
}
```

这里 `N_ITERS` 的计算隐式假设了 32 线程的 warp。当 wave 变成 64 线程时：
- 每次迭代覆盖 **32 行**（而非 16 行）
- 但迭代次数 `N_ITERS` 没有相应调整
- 结果：后半的 wave（线程 32-63）读写了**越界地址**

### 精确定位

通过编写 fused vs unfused 对比测试，我们发现：
- 错误**集中在 batch index 128-255**
- 最大误差出现在 `col=142`
- 这恰好是 wave 越界写入的区域

### 修复

将硬编码的步长替换为 wave-size 感知的计算：

```cpp
constexpr uint32_t ROWS_PER_COPY_STEP = WAVE_SIZE / 2;  // 64/2=32
constexpr uint32_t COPY_N_ITERS = (16 * N_ITERS) / ROWS_PER_COPY_STEP;
```

修复覆盖 `fully_fused_mlp.cpp` 中的 6 个位置。修复后，所有前向传播测试通过（0 个不匹配）。

**但训练仍然发散。**

## Bug #2：rocWMMA Fragment 布局不兼容

### 从前向正确到反向崩溃

前向传播完全正确，说明 Bug #1 的修复有效。但训练仍在 ~3000 步后 NaN，意味着**反向传播**存在问题。

为排除宽度特定的 bug，我们在 WIDTH=64（与通过测试相同的配置）下训练——同样在 2200 步发散。问题是系统性的。

### 深入对比 CUDA vs rocWMMA 的反向 kernel

反向传播的关键操作是将矩阵乘法结果（累加器 fragment）与前向激活值（matrix_a fragment）逐元素比较，实现 ReLU 反向：

```cpp
// CUDA WMMA: 这段代码是正确的
fragment<accumulator, 16,16,16, __half> acc;    // MMA 结果
fragment<matrix_a, 16,16,16, __half> fwd_act;   // 前向激活值
// 逐元素: if fwd_act.x[t] > 0 then keep acc.x[t], else zero
warp_activation_backward(acc, fwd_act);
```

在 CUDA WMMA 中，`accumulator.x[t]` 和 `matrix_a.x[t]` 映射到矩阵的**同一个位置**。这是 NVIDIA WMMA 实现的一个（未文档化的）保证。

**在 rocWMMA 中，这个保证不成立。**

AMD 的 MFMA（Matrix Fused Multiply-Add）指令对输入矩阵和输出矩阵使用**不同的寄存器布局**。`accumulator` fragment 的第 `t` 个元素和 `matrix_a` fragment 的第 `t` 个元素对应矩阵中**不同的行列位置**。

这意味着 `warp_activation_backward` 将 ReLU 门控应用到了**错误的位置**——不是把 gradient 正确地门控，而是随机地保留或丢弃。这种腐蚀是渐进的：每一步反向传播都引入微小的梯度错误，累积到 ~3000 步后导致权重爆炸。

### 修复：从寄存器到共享内存

解决方案是放弃 fragment 级别的逐元素操作，改用共享内存作为中间层：

```
MMA结果 (fragment)
  → store_matrix_sync → 共享内存 (行列布局确定)
  → 在共享内存中逐元素做 ReLU backward
  → 继续后续计算
```

共享内存的布局是显式的（`row * stride + col`），不依赖任何 fragment 内部映射。修复应用到两个位置：隐藏层反向（`threadblock_layer`）和最后一层反向（`kernel_mlp_fused_backward`）。

### 修复后的结果

| Step | 修复前 | 修复后 |
|------|-------|-------|
| 0 | 9.456 | 9.456 |
| 1000 | 0.112 → 发散 | **0.053** |
| 3000 | NaN | **0.026** |
| 5000 | — | **0.020** |
| 10000 | — | **0.017** |

Loss 单调下降，10000 步无 NaN，训练完全收敛。

## 性能分析

在 AMD Instinct MI300X（ROCm 6.4.3）上的 benchmark 结果：

| 配置 | 训练速度 (ms/step) | 吞吐量 |
|------|-------------------|--------|
| **tiny-rocm-nn** | 7.55 | 34.7M samples/sec |
| PyTorch FP32 | 9.47 | 27.7M samples/sec |
| PyTorch FP16 (AMP) | 6.05 | 43.3M samples/sec |
| PyTorch FP16 + compile | 5.89 | 44.5M samples/sec |

tiny-rocm-nn 比 PyTorch FP32 快 **1.25x**，但慢于 PyTorch AMP。

### hipBLAS FP16 Compute 实验

一个自然的优化想法是将 hipBLAS 的 compute type 从 FP32 切换到 FP16，因为原版 tiny-cuda-nn 用的 CUTLASS 就是 FP16 累加。我们做了实验：

- **精度**：FP16 compute 的 loss 收敛趋势与 FP32 完全一致（step 10000 loss: 0.018 vs 0.017）
- **性能**：FP16 compute 反而**慢了 2.2 倍**（21.1 ms/step vs 9.65 ms/step）

原因是 AMD MFMA 硬件**原生使用 FP32 做累加**。将 hipBLAS compute type 设为 `HIPBLAS_R_16F` 强制走了非 MFMA 的回退路径。这与 CUTLASS 的 FP16 累加不同——CUTLASS 是在软件层面控制 tile 级累加精度，而 hipBLAS 的 compute type 直接影响硬件路径选择。

### 性能差距的根源

tiny-rocm-nn 与 PyTorch AMP 的性能差距不在数值精度，而在**架构层面**：

- **tiny-cuda-nn**：使用 CUTLASS 将 GEMM + 激活函数融合到单个 kernel（epilogue fusion），无额外 kernel launch 和内存往返
- **tiny-rocm-nn**：使用 hipBLAS 做 GEMM（外部库调用），激活函数在独立 kernel 中，多了 launch 开销和 memory round-trip

## 经验总结

### 1. Wave64 不只是"乘以 2"

AMD 的 64 线程 wavefront 与 NVIDIA 的 32 线程 warp 差异远不止线程数。任何基于 warp/wave 大小做索引计算的代码（特别是向量化内存拷贝）都需要仔细审查。这类 bug 只在特定 batch 范围触发，普通测试可能覆盖不到。

### 2. Fragment 布局是实现细节，不是接口保证

CUDA WMMA 中 `accumulator` 和 `matrix_a` fragment 的元素布局恰好相同，但这不是 API 规范的一部分。依赖这种未文档化的行为，在 rocWMMA（或未来的 CUDA 架构）上会静默失败。安全的做法是通过共享内存中转任何跨 fragment 类型的逐元素操作。

### 3. "先好后坏"的 loss 曲线是梯度腐蚀的信号

如果训练前 1000 步正常但随后发散，通常意味着反向传播中存在微小但系统性的数值错误。前向传播正确不代表反向传播也正确——它们可能走完全不同的代码路径和内存布局。

### 4. 性能优化需要理解硬件路径

hipBLAS 的 `HIPBLAS_R_16F` compute type 在 AMD 上反而更慢，因为它绕过了 MFMA。同样的 API 参数在不同硬件上可能触发截然不同的执行路径。性能调优必须基于对目标硬件的理解，而非 API 层面的类比。

## 后续优化方向

1. **composable_kernel 替换 hipBLAS**：AMD 的 [composable_kernel](https://github.com/ROCm/composable_kernel) 库提供类似 CUTLASS 的 epilogue fusion 能力，可将 GEMM + 激活融合到单个 kernel
2. **hipBLASLt**：针对小矩阵 GEMM 提供更灵活的算法选择
3. **Fused kernel 调优**：优化 rocWMMA kernel 的共享内存访问模式、occupancy 和 wave 调度
4. **多流并行**：探索 GEMM、激活和优化器 kernel 的流级重叠

## 结语

将 tiny-cuda-nn 移植到 ROCm 不仅是 API 的机械替换。两个核心 bug 都源于 NVIDIA 和 AMD GPU 架构的根本差异——warp/wavefront 大小和张量核心的寄存器布局。这些差异在 API 层面完全不可见，只有深入到 GPU 执行模型才能理解和修复。

这次移植的意义在于：它证明了**即使是深度绑定 NVIDIA 张量核心的 GPU kernel，也可以在 AMD MFMA 硬件上正确运行**。rocWMMA 和 hipBLAS 提供了足够的能力完成移植，尽管一些隐式假设（fragment 布局兼容性）需要显式处理。

对于 3D 重建和神经渲染社区，tiny-rocm-nn 打开了在 AMD GPU 上运行 Instant NGP、Nerfstudio 等项目的可能性。当前版本在功能正确性上已完全验证，性能优化（特别是通过 composable_kernel 实现 epilogue fusion）是下一阶段的工作重点。

项目代码开源在 [GitHub](https://github.com/ZJLi2013/tiny-rocm-nn)，欢迎贡献和反馈。
