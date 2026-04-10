# rocm3d: 一张替换表 + Code Agent = CUDA→ROCm 自动迁移

> **TL;DR** — 把 CUDA→ROCm 迁移的经验浓缩成一份 **Cursor Agent Skill**（ROCm 库替换表），剩下的交给 Code Agent 自动完成。30+ repo 覆盖 3D 生成/重建、视频世界模型、VLA 具身智能，大部分 out-of-box 运行在 AMD MI300X 上。项目已开源：[rocm3d](https://github.com/ZJLi2013/rocm3d)

---

## 从 autorun 到 skill：一次大幅简化

在上一版 `rocm3d-autorun` 中，我构建了一套三模块流水线：Skill 生成脚本 → docker_agent 远端执行 → LLM 分析日志自动 patch。架构看起来很完整，但实际跑下来发现：

- **docker_agent 的能力被 Code Agent 覆盖了。** Cursor、Claude Code 等 Agent 本身就能 SSH 到远端节点执行命令、读日志、修脚本、重试 — 而且上下文更丰富，不需要单独的 JSON patch 协议。
- **LLM log analyzer 的 few-shot 积累，不如直接写进 Skill。** 与其让 LLM 每次从日志中重新推理 root cause，不如把已知的 error→fix 映射直接编码为规则。
- **真正有价值且不可替代的是那张替换表。** 哪个 CUDA 库对应哪个 ROCm 安装方式、哪个版本有 wheel、哪个需要源码编译 — 这些信息散落在各个 GitHub issue 和 AMD 文档中，整合成一份 Skill 后，任何 Code Agent 都能直接消费。

所以，**rocm3d** 现在只保留一个核心文件：

```
.cursor/skills/rocm-lib-compat/
  SKILL.md       # ROCm 库替换表 + 版本策略 + 已知坑位
```

外围的 docker_agent、LLM analyzer、结构化 JSON 输出全部移除。脚本生成、远端执行、日志分析、错误修复 — 这些工作交给 Cursor / Claude Code 等通用 Code Agent 完成。

---

## 核心：ROCm 库替换表

迁移一个 ML repo 到 ROCm，核心难点是依赖替换。SKILL.md 维护了一张实战验证过的替换表：

| CUDA 库 | ROCm 方案 | ROCm 版本 |
|---------|----------|----------|
| flash-attn | `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pip install flash-attn` (Triton) | 6.x / 7.x |
| flash-attn | `pip install aiter` — AITER CK 后端，**快 ~25%** | **7.x** |
| xformers | `pip install xformers --index-url https://download.pytorch.org/whl/rocm6.4` | 6.4 |
| gsplat | `pip install gsplat --index-url=https://pypi.amd.com/simple` | 6.4 / 7.0 |
| pytorch3d | 预编译 ROCm wheel（源码编译产出 CPU-only，已踩坑） | 6.4 |
| bitsandbytes | `pip install bitsandbytes` (≥v0.45.3 原生支持 ROCm) | 6.4+ |
| nvdiffrast | [ROCm fork](https://github.com/ZJLi2013/nvdiffrast/tree/rocm) (RDNA + CDNA3) | 6.4+ |
| flex_gemm | [ROCm fork](https://github.com/ZJLi2013/FlexGEMM/tree/rocm) (Triton backend) | 6.4+ |
| cumesh | [ROCm fork](https://github.com/ZJLi2013/CuMesh/tree/rocm) (hipified) | 6.4+ |
| tinycudann | [tiny-rocm-nn](https://github.com/ZJLi2013/tiny-rocm-nn) | 6.4+ |

配合版本策略决策规则：
- Repo 用了 xformers / gsplat / pytorch3d → **ROCm 6.4**（wheel 生态最完整）
- Repo 只用 flash-attn 且追求极致性能 → **ROCm 7.x**（AITER CK）
- 纯 PyTorch → **任意 ROCm 版本**，SDPA 自动走 AOTriton

---

## 工作流：人给方向，Agent 执行

实际使用流程极简：

```
1. 在 Cursor 中说：
   "使用 rocm-lib-compat skill，给 https://github.com/<owner>/<repo> 生成 ROCm install 脚本"

2. Agent 读取 repo 的 README + requirements.txt，
   按替换表生成 install.sh

3. Agent SSH 到远端 GPU 节点，在 Docker 中执行

4. 失败 → Agent 读日志、修脚本、重试
   （Code Agent 本身的能力，不需要额外框架）

5. 成功 → 跑推理/训练/eval，收集结果
```

这个流程中没有任何自定义的 Python 框架，只有一份 SKILL.md 和通用的 Code Agent 能力。

---

## 验证结果

基于 AMD MI300X + ROCm 6.4，已验证 30+ repo 横跨四个领域：

### 3D 生成与重建（12 个已验证）

| Repo | 领域 | 关键 ROCm 库 |
|------|------|-------------|
| [Tencent/Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) | Image-to-3D + PBR | — (AOTriton FA) |
| [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | Image-to-3D (O-Voxel, 4B) | flash-attn, flex_gemm, cumesh, nvdiffrast ([ROCm fork](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm)) |
| [wgsxm/PartCrafter](https://github.com/wgsxm/PartCrafter) | 部件感知 3D 生成 | pytorch3d |
| [apple/ml-sharp](https://github.com/apple/ml-sharp) | 3D 重建 | gsplat |
| [naver/dust3r](https://github.com/naver/dust3r) | 稠密立体重建 | croco |
| [facebookresearch/fast3r](https://github.com/facebookresearch/fast3r) | 快速 3D 重建 | croco |
| [nv-tlabs/Difix3D](https://github.com/nv-tlabs/Difix3D) | 3D 扩散修复 | xformers |
| [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | 视觉定位 | — |
| [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | 单目深度 + 3DGS | xformers, gsplat |
| [expenses/gaussian-splatting](https://github.com/expenses/gaussian-splatting) | 3DGS | diff-gaussian-rasterization |
| [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | 地图重建 | — |
| [openai/shap-e](https://github.com/openai/shap-e) | 文本/图像转 3D | — |

### 视频生成 / 世界模型（4 个已验证）

| Repo | 领域 | 亮点 |
|------|------|------|
| [SkyworkAI/Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) | 视频世界模型 | AITER CK 后端，PR ready |
| [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) | 学习型世界模型 | 推理 + 8-GPU 训练 |
| [H-EmbodVis/HyDRA](https://github.com/H-EmbodVis/HyDRA) | 混合记忆视频世界模型 | FA2 Triton vs SDPA 对比 |
| [ABU121111/DreamWorld](https://github.com/ABU121111/DreamWorld) | 视频生成 (Wan2.1) | 2 videos, ~39min |

### VLA / 具身智能（2 个已验证）

| Repo | 领域 | 亮点 |
|------|------|------|
| [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM) | World Action Model | LIBERO eval 5/5 success, out-of-box |
| [starVLA/starVLA](https://github.com/starVLA/starVLA) | VLA 框架 (Qwen3-VL) | 8-GPU 训练 20K steps + LIBERO 3-suite eval avg **97.8%**, out-of-box |

### 标杆项目：TRELLIS.2 端到端 ROCm 适配

[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) 是目前依赖最复杂的验证案例 — 同时需要 flash-attn、FlexGEMM、CuMesh、nvdiffrast、o-voxel 五个 CUDA 原生库。为此创建了完整的 [ROCm fork](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm)，`setup.sh` 自动检测平台并安装对应的 ROCm 兼容依赖，实现了 `git clone + setup.sh` 一键部署。

### 标杆项目：starVLA 训练 + LIBERO Eval

[starVLA](https://github.com/starVLA/starVLA) 是最新的 VLA 框架，在 AMD MI300X 上实现了完整的 8-GPU 分布式训练（DeepSpeed ZeRO-2）+ LIBERO 机器人仿真评估。20K steps 的 checkpoint 在三个 LIBERO suite 上取得了 spatial 96.7% / object 100% / goal 96.7%（avg 97.8%）的成功率，与 NVIDIA 30K steps 基线持平。整个过程 zero code change — 纯 PyTorch + AOTriton SDPA out-of-box。

---

## 关键发现

跑完这些 repo 后，几个值得分享的观察：

**1. 大部分 ML repo 已经 out-of-box。** 纯 PyTorch 的 repo（无 flash-attn / xformers / gsplat）通常零改动就能跑。PyTorch 2.6+ 的 SDPA 自动走 AOTriton，性能接近 flash-attn。

**2. ROCm 6.4 的 wheel 生态比想象中完整。** xformers、gsplat、pytorch3d 都有预编译 wheel，flash-attn 走 Triton 路径 37 秒装完（不需要编译 2199 个 .hip 文件）。

**3. 真正的 blocker 集中在少数深度绑定 CUDA 的库。** tinycudann、spconv、nvdiffrast — 这些需要源码级 hipify 或维护 ROCm fork。但它们数量有限，逐个击破后可惠及整个下游生态。

**4. Skill > Framework。** 一开始我以为需要构建复杂的 agent 流水线（docker_agent + LLM analyzer + patch protocol），实践证明一份好的 Skill 文档 + 通用 Code Agent 就够了。Skill 是可组合、可演化的知识载体；Framework 是僵化的、需要维护的代码。

---

## 与上一版的对比

| 方面 | v1 (rocm3d-autorun) | v2 (rocm3d) |
|------|-------------------|-------------|
| 架构 | Skill + docker_agent + LLM analyzer | **只有 Skill** |
| 代码量 | ~2000 行 Python | **~0 行**（纯 Markdown） |
| 依赖 | docker SDK, LLM API client | **无** |
| 执行方式 | 自建 CLI + JSON patch 协议 | **Cursor / Claude Code 原生能力** |
| 经验沉淀 | 两条通道（SKILL.md + fewshot.md） | **一条通道**（SKILL.md） |
| 覆盖范围 | 6 repo（仅 install） | **30+ repo**（install + 推理 + 训练 + eval） |
| 可维护性 | 需要维护 Python 代码 | **只需更新替换表** |

---

## 快速上手

```bash
# 1. 克隆项目
git clone https://github.com/ZJLi2013/rocm3d.git

# 2. 在 Cursor 中，项目根目录下触发 skill：
#    "使用 rocm-lib-compat skill，给 https://github.com/<owner>/<repo> 生成 ROCm install 脚本"

# 3. Agent 会自动：
#    - 读取目标 repo 的依赖
#    - 按替换表生成 install 命令
#    - SSH 到远端 GPU 节点执行
#    - 失败时自动修复重试
```

不需要安装任何依赖。Skill 是纯文本，任何支持 Agent Skill 的工具都能消费。

---

## 后续方向

1. **补齐 ROCm fork 生态**：tinycudann (tiny-rocm-nn)、spconv — 打通后可覆盖 NeRF/3DGS 全栈
2. **ROCm 7.x 全面验证**：AITER CK 在 flash-attn 密集型 workload 上快 25%，但 xformers/gsplat wheel 尚未跟进
3. **更多 VLA / 世界模型**：具身智能和视频世界模型是 ROCm 生态的新增长点
4. **上游贡献**：将验证过的 ROCm 兼容性以 PR / Issue 形式回馈原始 repo

---

*项目地址：[https://github.com/ZJLi2013/rocm3d](https://github.com/ZJLi2013/rocm3d)*

*系列文章：*
- *[Unlocking 3D Generative AI on AMD GPUs](Enable_3D-GenAI-on-AMD.md) — 手动迁移 10 个 3D 重建/生成模型*
- *[Autoresearch on AMD MI300](autoresearch-rocm-mi300.md) — Karpathy's autoresearch 迁移实录*
