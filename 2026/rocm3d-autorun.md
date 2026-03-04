# rocm3d-autorun: 用 AI Agent 自动化 3D 模型的 ROCm 迁移

> **TL;DR** — 用一个 AI Agent 驱动的流水线，把 3D 生成/重建主流模型从 CUDA 生态自动迁移到 ROCm 平台：脚本生成 → Docker 执行 → LLM 自动修复 → 经验沉淀，6 个 repo 全部安装成功。项目已开源：[rocm3d-autorun](https://github.com/ZJLi2013/rocm3d-autorun)

---

## 背景

在上一篇文章 [Unlocking 3D Generative AI on AMD GPUs](2026/Enable_3D-GenAI-on-AMD.md) 中，我展示了如何在 AMD GPU 上手动跑通主流 3D 重建和生成模型。

手动迁移的问题很明显：

- 每个 repo 的依赖环境各不相同，install 脚本需要逐一调试
- 错误信息往往隐藏在几百行日志中，定位 root cause 耗时
- 踩过的坑下次遇到新 repo 时又得重复排查
- 迁移目标越来越多（3DGS 加速库、World Model、VLA），手动方式无法规模化

**rocm3d-autorun** 就是为了解决这个问题而构建的：一个把"手动迁移+调试"过程自动化的 AI Agent 流水线。

---

## 整体架构

```
用户输入：repo URL
         ↓
┌──────────────────────────────────────────┐
│  Skill: rocm-install-script-generator   │
│  消费者：Cursor / Claude Code Agent      │
│                                          │
│  读取 repo README + requirements，       │
│  按 Block A~H 规则组装 install.sh         │
└──────────────────────────────────────────┘
         ↓ git push → 远端 GPU 节点 git pull
┌──────────────────────────────────────────┐
│  docker_agent（纯命令行，不依赖 IDE）     │
│                                          │
│  1. docker run install.sh               │
│  2. 失败 → LLM 分析日志 → 生成 patch     │
│  3. 应用 patch → 重试（最多 3 次）       │
│  4. 输出结构化 JSON 结果                 │
└──────────────────────────────────────────┘
         ↓ 经验沉淀
┌──────────────────────────────────────────┐
│  两条经验通道                             │
│  · SKILL.md：通用 ROCm 兼容规则更新      │
│  · analyzer_fewshot.md：error→patch 示例 │
└──────────────────────────────────────────┘
```

三个核心模块相互独立，也可以单独使用：

| 模块 | 作用 | 运行环境 |
|------|------|---------|
| `rocm-install-script-generator` skill | 生成 install/run 脚本 | 本地 Cursor / Claude Code |
| `docker_agent` | 执行验证 + 自动 patch | 远端 GPU 节点（只需 Python + Docker） |
| `llm_log_analyzer` | 日志分析 + patch 生成 | 随 docker_agent 自动调用 |

---

## Skill：冷启动生成脚本

Skill 是给 AI Agent 读的操作手册（`docs/skills/rocm-install-script-generator/SKILL.md`），定义了一套标准化的脚本生成规范，分为 Block A~H：

| Block | 作用 |
|-------|------|
| A | 基础环境（ROCm PyTorch 安装） |
| B | ROCm 版本适配（xFormers、gsplat 等） |
| C | 过滤 requirements.txt 中的冲突包 |
| D | `--no-build-isolation` 场景处理 |
| E | git+URL 依赖过滤（避免覆盖已安装包） |
| F | CUDA kernel hipify（待扩展） |
| G | run 阶段脚本生成 |
| H | 经验约束（已知 fail pattern 的防御性规则） |

在 Cursor 中对话触发后，Agent 读取 repo 的 README 和 `requirements.txt`，按 Block 规则组装出可在 ROCm 环境运行的 `install.sh`，输出到 `samples/auto_gen/<repo>_install.sh`。

---

## docker_agent：自动闭环执行

生成的脚本推送到远端 GPU 节点后，`docker_agent` 负责完整的执行+修复闭环：

```bash
PYTHONPATH=./src python -m docker_agent \
  --repo_url https://github.com/<owner>/<repo> \
  --base-image rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0 \
  --install-script samples/auto_gen/<repo>_install.sh \
  --auto-patch-on-fail \
  --max-auto-patch-retries 3 \
  -o samples/auto_gen/test_output/<repo>.json
```

失败时的自动修复流程：

```
install 失败
    ↓
LLM 分析日志（stdout tail + stderr tail + script_text）
    ↓
生成 JSON patch plan：
  {
    "root_cause": { "evidence": [...], "why": "..." },
    "execution_plan": {
      "action": "patch_script",
      "patches": [
        { "op": "replace_line", "match": "旧行", "content": "新行" }
      ]
    }
  }
    ↓
应用 patch 到 install.sh → 重试 install
    ↓（最多 3 次，或 action=need_human 时停止）
```

输出的结构化 JSON 包含完整执行日志、patch 记录和最终状态，方便离线分析和 CI 集成。

---

## LLM Provider：轻量多 Provider 支持

项目内置一个零依赖（纯 Python stdlib）的 LLM 客户端，支持：

```python
# OpenAI
LLM_PROVIDER=openai
LLM_API_KEY=sk-...

# Anthropic
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-...
LLM_MODEL=claude-3-5-haiku-20241022

# 任意 OpenAI-compatible（AMD Gateway、vLLM、Ollama）
LLM_PROVIDER=openai_compat
LLM_BASE_URL=https://llm-api.example.com/v1
LLM_API_KEY=<key>
```

没有 LangChain，没有 vendor SDK，只需 `requirements.txt` 中的 `docker>=7.0.0`。LLM 不可达时，自动回退到 `need_human` 状态，不影响执行链路。

---

## 当前验证结果

基于 MI300X GPU 节点，ROCm 6.4 环境，验证了以下 6 个 3D 生成/重建 repo：

| Repo | 领域 | Install 状态 | 备注 |
|------|------|------------|------|
| [mvinverse](https://github.com/dvlab-research/mvinverse) | 多视角 3D 重建 | ✅ AUTO_SUCCESS | |
| [Anything-3D](https://github.com/Anything-of-anything/Anything-3D) | 通用 3D 分割 | ✅ AUTO_SUCCESS | |
| [any4d](https://github.com/snap-research/any4d) | 4D 动态场景 | ✅ AUTO_SUCCESS | |
| [DimensionX](https://github.com/wenqsun/DimensionX) | 视频→3D | ✅ AUTO_SUCCESS | |
| [FLARE](https://github.com/ant-research/FLARE) | 大规模重建 | ✅ LLM_FIXED | auto-patch 修复后成功 |
| [ReCamMaster](https://github.com/KwaiVGI/ReCamMaster) | 相机轨迹控制 | ✅ AUTO_SUCCESS | |

6/6 安装成功，其中 FLARE 经历了 1 轮 LLM auto-patch 后成功（`--no-build-isolation` 场景）。

---

## 经验沉淀机制

这是整个系统中我认为最有价值的部分。

每次新 repo 踩坑后，经验分两条通道沉淀：

**通道 1：SKILL.md 更新（通用规则）**

当发现可以推广到所有 repo 的新规则时，更新 `docs/skills/rocm-install-script-generator/SKILL.md`，让下一次冷启动生成的脚本直接规避这个问题。

典型例子：发现 `git+https://...@<commit>` 依赖会覆盖已安装的 ROCm 版本 torch 后，在 Block E 中添加过滤规则。

**通道 2：analyzer_fewshot.md 追加（具体 patch 示例）**

当发现可复用的 `error signal → patch` 模式时，追加一个示例到 `src/docker_agent/prompts/analyzer_fewshot.md`，让 LLM 在未来遇到同类错误时直接模仿。

格式要求：给出错误特征 + 完整的 patch JSON（含 op/match/content），LLM 直接模仿，不需要再推理。

---

## 与手动迁移的对比

| 方面 | 手动迁移 | rocm3d-autorun |
|------|---------|---------------|
| 脚本生成 | 逐行手写，参考文档 | Agent 自动生成，30 秒内 |
| 错误定位 | 人工读几百行日志 | LLM 自动提取 error snippet |
| 修复迭代 | 手动改脚本重跑 | auto-patch + retry，无人值守 |
| 经验复用 | 靠记忆或注释 | 结构化沉淀到 SKILL.md + fewshot |
| 规模化 | 线性增长的人力 | 新 repo 只需触发 agent，基础设施复用 |

---

## 快速上手

```bash
# 1. 克隆项目
git clone https://github.com/ZJLi2013/rocm3d-autorun.git
cd rocm3d-autorun

# 2. 安装依赖（远端 GPU 节点）
pip install -r requirements.txt

# 3. 配置 LLM provider
cp .env.example .env
# 编辑 .env，填写 LLM_PROVIDER / LLM_API_KEY / LLM_BASE_URL

# 4. 本地用 Cursor 生成 install 脚本（对话触发 skill）
# "给 https://github.com/<owner>/<repo> 生成 ROCm install 脚本"

# 5. git push 后，远端节点执行
PYTHONPATH=./src python -m docker_agent \
  --repo_url https://github.com/<owner>/<repo> \
  --base-image rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0 \
  --install-script samples/auto_gen/<repo>_install.sh \
  --auto-patch-on-fail \
  -o samples/auto_gen/test_output/<repo>.json
```

---

## 后续方向

当前 6 个 repo 的 install 阶段全部完成，下一步按优先级：

1. **run 阶段打通（近期）**：在已安装容器内跑推理，暴露 ROCm 真实 runtime 障碍（算子兼容、`torch.cuda.*` 调用等），补充 runtime 类 fewshot 经验
2. **Gaussian Splatting 加速库（中期）**：`diff-gaussian-rasterization`、`simple-knn` 等核心库的 HIP 适配，一旦打通可惠及整个 3DGS 生态
3. **World Model / VLA（中期）**：等 3DGS 基础库稳定后进入
4. **Skill 体系扩展**：增加 `python-api-sample-generator` skill，自动生成推理示例代码

---

*项目地址：[https://github.com/ZJLi2013/rocm3d-autorun](https://github.com/ZJLi2013/rocm3d-autorun)*
