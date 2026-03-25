# Franka Genesis 端到端合成数据管线实战

背景是基于一个最简单的cube pick-up 实验，端到端走通**数据采集 -> 策略训练 -> 仿真验证**的流程。当前，之前有两周的基于 Lerobot so-101 的试错，详细参考 [Lerobot101 x Genesis 仿真实战](../lerobot_genesis_2/readme.md)，当时的基本结论是so-101 作为5DOF 机械臂，在仿真理想环境下，非常容易造成过约束而造成 IK Solver 跳变，对于任何微小扰动都可能发散的问题，所以，这里选择了 7DOF 的 Franka 机械臂，基本上 IK Solver 一上来就能100% 成功抓取，更重要的是对小扰动具有鲁棒可恢复。

---

## 一、背景

目标是构建一条**合成数据管线**：在仿真器中自动生成机器人抓取数据 → 训练策略模型 → 闭环评估 。

技术栈：
- **仿真器**：Genesis（GPU 加速物理仿真）
- **机器人**：Franka Emika Panda（7-DOF，工业级精度）
- **任务**：Pick-Cube（从桌面抓取红色方块并抬起）
- **策略架构**：MLP BC → ACT（Action Chunking Transformer）→ SmolVLA（Vision-Language-Action）
- **评估指标**：HOME-START 成功率（机器人从固定起始位置出发，策略完全自主控制）

整个项目的核心问题是：**用 IK（逆运动学）规划器自动生成的轨迹数据，训练出可部署的抓取策略**

---

## 二、实验历程

### 2.1 起点：数据完美，模型也收敛了，但成功率 0%

第一轮实验的结果看起来非常好：

| 指标 | 结果 |
|------|------|
| IK 数据采集成功率 | 200/200 = **100%** |
| BC 训练 loss | **0.0003**（极低） |
| HOME-START 成功率 | **0/10 = 0%** |

数据 100% 成功、loss 极低收敛、但部署成功率 0%。这是一个非常典型的 Imitation Learning 陷阱：**loss 低 ≠ 策略正确**。

当前引入了一个诊断指标 **E1 N-step**：每 N 步注入一次 ground truth (GT) action，测量策略在两次 GT 之间的"自由飞行"能力：

| E1 间隔 | 成功率 |
|---------|--------|
| N=1（每步 GT） | 100% |
| N=5 | 90% |
| N=10 | 60% |
| N=20 | 0% |

策略在短距离（5步）内表现尚可，但一旦自由运行超过 20 步就完全失败。这就是经典的 **Covariate Shift**：策略的微小预测误差逐步累积，导致机器人进入从未见过的状态，然后彻底发散。

### 2.2 排除假设的「马拉松」

接下来的两周，当前系统性地测试了 7 个假设，试图找到 0% 成功率的原因。每个假设都经过严格的实验验证：

#### ❌ 假设 1：训练时加噪声能改善泛化

直觉：给输入加高斯噪声，让模型见过更多状态分布。

```
结果：E1 N=5 从 90% 暴跌到 0%
```

**教训**：噪声只扩展了输入分布，但没有提供正确的 recovery label。模型看到了"偏离后的状态"，但对应的 action label 仍然是"正常轨迹上的动作"——这是答非所问。

#### ❌ 假设 2：ACT 的 Action Chunking 能解决误差累积

直觉：一次预测多步动作（chunk），减少单步累积误差。

```
结果：HOME-START 仍然 0%，E1 N=5 从 90% 降到 0%
原因：28.8M 参数只训练了 2000 步（loss~0.8），严重欠拟合
```

**教训**：架构方向可能正确，但训练不足会让一切假设失去意义。ACT 需要远多于 2000 步的训练。

#### ❌ 假设 3：DART 数据增强能让策略学会纠偏

DART（Disturbances for Augmenting Robot Trajectories）是一种经典的 IL 数据增强方法：在数据采集时注入扰动，让专家示范如何从偏离状态恢复。

当前实现了两种 DART 变体：
- **Jump DART**：一次性跳跃式扰动
- **Drift DART**：连续微扰

```
Jump DART:  σ=0 ratio 11.8x（vs baseline 12.5x），几乎没有改善
Drift DART: σ=0 ratio 12.89x，甚至略有恶化
```

后来发现，**实现的根本不是真正的 DART**。原始 DART 论文的核心前提是：supervisor 必须是**闭环策略**——能对任意状态给出正确的 action。而当前 IK planner 是**开环的**：在 episode 开头一次性规划整条轨迹，扰动后 action label 仍然是预规划的插值点，不是从当前状态重新求解的 IK。

```python
# 当前实现（开环，错误的 DART）
traj = plan_pick_trajectory(cx, cy)   # 开头一次性规划
for target in traj:
    perturb(state)                     # 扰动状态
    execute(target)                    # 但 target 仍是预规划的！
    record(state, target)              # action label 不匹配当前 state

# 正确的 DART（闭环）
for each step:
    current = observe()
    target = solve_ik(current, goal)   # 从当前状态重新求解
    perturb_and_execute(target)
    record(current, target)            # action label 始终匹配当前 state
```

**教训**：实现算法时必须回到原始论文核实前提条件。名字相同但缺少关键组件的实现，可能完全无效。

#### ❌ 假设 4：加入历史状态能提供速度信息

直觉：拼接前两步的 state 作为输入，让模型利用运动历史推断当前阶段。

```
结果：σ=0 ratio 从 12.5x 恶化到 20.3x
```

**教训**：Temporal blindness 不是瓶颈。历史信息在分布外（OOD）状态下反而让模型更脆弱——它过拟合了训练数据的时序模式。

#### ❌ 假设 5：闭环 IK 重规划能解决 action label 质量问题

在发现开环 DART 的问题后，当前实现了真正的闭环 IK：每步从当前构型重新求解 IK。

```
结果：σ=0 ratio 从 12.5x 恶化到 20.80x
HOME-START 仍然 0%
```

分析发现 σ=0.0005 的扰动幅度太保守（累积偏移 ~0.004 rad），远小于部署时的真实误差（~0.009 rad/step）。闭环 IK 确实改善了 label 质量，但扰动不够大时，策略仍然没见过真实 drift 幅度的状态。

#### 🐛 发现隐藏 Bug：State-Action 时序错位

在排查过程中，当前发现**所有数据采集脚本**都有一个时序 bug：

```python
# 错误（所有脚本）：先执行，再观察
execute(action)          # 1. 执行 action
sim_step()               # 2. 仿真一步
state = observe()        # 3. 读 state（已经是执行后了！）
record(state, action)    # 记录的是 (s_{t+1}, a_t)

# 正确：先观察，再执行
state = observe()        # 1. 先读 state (s_t)
execute(action)          # 2. 再执行
sim_step()               # 3. 仿真
record(state, action)    # 记录的是 (s_t, a_t) ✓
```

由于 PD 控制器跟踪很紧，执行完 action 后 `s_{t+1} ≈ a_t`，模型实际上学到了**恒等映射 f(x) ≈ x**——即"原地不动"。

修复后 policy 行为有改善（HOME 处 displacement 从 0.109 降到 0.049 rad），但 HOME-START **仍然 0%**。

**教训**：时序错位是 bug 但不是 root cause。修复 bug 是必要的，但不充分。

### 2.3 顿悟时刻：Goal Ambiguity

在排除了上述所有假设后，当前终于意识到一个根本性的问题。

数据采集时，cube 的位置在 `x ∈ [0.4, 0.7], y ∈ [-0.2, 0.2]` 范围内随机化，但 observation 只有 9D 关节角度，**不包含 cube 位置**。

这意味着：

```
相同的 HOME state + cube=(0.40, 0.10) → action A（arm 向左前伸）
相同的 HOME state + cube=(0.70,-0.20) → action B（arm 向右前伸）
相同的 HOME state + cube=(0.55, 0.00) → action C（arm 向正前伸）
```

MSE 优化的结果：`π(HOME) = mean(A, B, C, ...) = 指向一个不存在的"平均 cube 位置"`

**策略第一步就走错了，然后 covariate shift 滚雪球。**

这个问题用一个公式概括：

```
当前学的：π(s) → a        ← 不是单值函数！（同一 s 对应多个不同 a）
应该学的：π(s, goal) → a  ← goal = cube position，给定 goal 后映射唯一确定
```

### 2.4 一行命令验证 Root Cause

为了验证这个假设，当前做了最简单的实验：**固定 cube 位置**（零代码改动，只改命令行参数）。

```bash
# 唯一改动：--cube-x-min 0.55 --cube-x-max 0.55 --cube-y-min 0 --cube-y-max 0
```

| 指标 | cube 随机化 | **cube 固定** |
|------|-----------|-------------|
| HOME-START | **0%** | **100%** ✅ |
| lift | -0.0003m | **0.1554m** |

**从 0% 直接跳到 100%。** 这是整个实验系列中首次出现 HOME-START 成功。

观测、模型、训练参数完全不变，唯一区别是消除了 goal ambiguity。这也回溯性地解释了**为什么之前所有方法全部无效**：

| 之前的尝试 | 为什么无效 |
|-----------|----------|
| Noise augmentation | 扩展了 state 分布但 label 仍是 goal-agnostic |
| DART | 扰动数据仍然缺 cube position，仍是 multi-modal |
| 闭环 IK | 改善了 label 质量，但 observation 仍缺 goal |
| History | 历史 state 提供了速度但没提供 goal |
| 时序修复 | 修正了 (s,a) 配对语义，但 s 仍不含 goal |

---

## 三、从 0% 到 60%：架构升级之路

确认 root cause 后，解决方案很明确：**把 goal 信息加入 observation**。

### 3.1 MLP BC + Goal Conditioning：天花板 ~30%

将 observation 从 9D 扩展到 11D（9 joints + cube_x + cube_y），但 MLP 的泛化能力很快碰到天花板：

| 实验 | unseen cube 成功率 | 问题 |
|------|-----------------|------|
| 10 episodes | 40%（假信号*） | eval seed = training seed |
| 100 episodes | 0% | MLP 过拟合，"查表"而非"插值" |
| 100ep + noise σ=0.01 | **30%** | MLP 架构极限 |

> *一个重要教训：eval seed 和 training seed 相同时，eval 的 cube 位置 = 训练时的位置，实际测的是"训练集成功率"。此后当前强制 eval seed ≠ training seed。

**教训**：MLP 在连续 goal 空间无法有效泛化。100 episodes 的数据量反而让 MLP 学会了"记住每个训练位置的精确动作"，rollout 时微小偏差即触发 cascading failure。

### 3.2 ACT：Transformer 的泛化优势

切换到 ACT（Action Chunking Transformer, ~45M params），同样 100 episodes 数据：

**结果：6/10 = 60% 成功率**

相比 MLP BC 的 30%，ACT 的优势来自：
1. **Transformer 架构**：在连续空间有更好的插值能力
2. **Action chunking**：一次预测多步动作，提供时间一致性
3. **VAE 正则化**：防止过拟合

### 3.3 SmolVLA：冻结视觉的高效训练

SmolVLA（450M params, HuggingFace LeRobot 官方模型）仅训练 action expert（~50M），冻结预训练的 SigLIP 视觉编码器和 SmolLM2 语言模型：

**结果：6/10 = 60% 成功率，训练仅需 6 分钟**

| 维度 | ACT | SmolVLA |
|------|-----|---------|
| 成功率 | 60% | 60% |
| 训练时间 | 42 min | **6 min** |
| VRAM | ~6 GB | ~2.2 GB |
| 参数量 | ~45M (全部训练) | 450M (仅训练 50M) |

更有趣的是，两者在**不同 cube 位置上互补**：ACT 独占成功 2 个位置，SmolVLA 独占成功 2 个位置。理论组合可达 80%。

---

## 四、当前的局限

### 4.1 推理依赖 Oracle Cube 位置

必须诚实地说明：**当前的 60% 成功率是在"作弊"条件下取得的。**

ACT 和 SmolVLA 的输入中包含 `cube_x, cube_y`——这两个值直接来自仿真器的 `cube.get_pos()`，是 ground truth 世界坐标。真实部署时不可能直接获取。

```
当前 observation = [9D joints, cube_x, cube_y]  ← cube_xy 来自仿真器 GT
                                                   真实部署不可用
```

这意味着当前验证的是 "**给定目标位置的 oracle 信息，action head 能否正确执行抓取**"，而非 "**模型能否从视觉中自主感知目标并执行**"。

### 4.2 为什么说这仍然是有价值的

虽然用了 oracle 信息，这个实验阶段验证了几个关键结论：

1. **Goal Ambiguity 是 root cause**：没有 goal 信息时任何策略都是 0%
2. **合成数据管线可行**：IK 规划器 + Genesis 仿真器生成的数据，能训出可工作的策略
3. **Transformer 架构对泛化至关重要**：MLP 天花板 30%，ACT/SmolVLA 达到 60%
4. **SmolVLA 预训练视觉编码器的训练效率极高**：6 分钟达到 ACT 水平

---

## 五、提炼：踩坑总结与最佳实践

### 🔴 踩坑清单

| # | 踩坑 | 代价 | 教训 |
|---|------|------|------|
| 1 | **Loss 低 ≠ 策略正确** | 浪费了大量时间在"为什么 loss 这么低还不 work" | MSE 收敛到条件均值 E[a\|s] 是数学正确的，但如果映射是多模态的，条件均值就是一个不存在的"平均动作" |
| 2 | **时序 bug 隐藏在所有数据脚本中** | 所有实验都在错误数据上跑 | 永远先写诊断脚本验证 (s,a) 配对的语义正确性 |
| 3 | **Eval seed = Training seed** | 40% 的假信号 | 强制 eval seed ≠ training seed，测 unseen 数据 |
| 4 | **没回原始论文检查 DART 前提** | 两轮 DART 实验白做 | 实现算法前必须确认前提条件是否满足 |
| 5 | **Noise augmentation 不提供 recovery label** | E1 N=5 从 90% 暴跌到 0% | 给输入加噪 ≠ 教模型纠偏，需要配套正确的 action label |
| 6 | **MLP 在大数据量上"查表"** | 100ep 反而不如 10ep | 简单架构 + 更多数据 = 过拟合，需要合适的架构复杂度 |
| 7 | **忽略 observability** | 整个前半段实验 | 设计 IL 系统的第一步是确认 π(obs) → a 是否为单值函数 |

### 🟢 最佳实践

1. **Phase 0：先验证 π(obs)→a 是单值函数**——列出所有随机化变量，确认 observation 包含足够信息区分它们；用固定参数做消融确认
2. **诊断优先于修复**——先建 E1 N-step、D1 σ=0 ratio、时序诊断等工具，再决定改什么
3. **逐步加复杂度**——固定 cube → oracle goal → vision inference，每步确认前一步 work 再进入下一步
4. **合成数据三个 Gap**——开环→闭环（每步 IK）、off-trajectory 覆盖率（1%→100%）、joint-space→task-space 插值
5. **评估严谨性**——eval seed ≠ training seed、HOME-START 全自主为金标准、记录每 episode 具体结果

---

## 六、VLA 后训练的 Next Steps

当前停在 "oracle cube position + action head fine-tune" 阶段。走向可部署 VLA 的路线图：

1. **去掉 Oracle，视觉承担 Goal Inference**（最关键）——从 `π(joints, cube_xy, images)` 变为 `π(joints, images)`。SmolVLA 的 SigLIP 预训练编码器已有 zero-shot 物体识别能力，是首选路线；预计成功率从 60% 降至 ~40%，可能需要 LoRA fine-tune 或更多数据（200+ ep）
2. **提升视觉真实感**——当前纯色平面渲染与真实世界差距大。路径：Rasterizer (L1) → RayTracer (L2) → **Blender + Gaussian Splatting (L3)**。关键洞察：RL 训练不需要逼真视觉（state-based），但 VLA 训练数据必须逼真
3. **RL Expert 作为 Recovery Oracle**——纯 IK 数据缺乏 recovery supervision。Genesis 官方两阶段框架（PPO teacher → DAgger BC student）是最成熟方案，也是 DiffRL→VLA、SimpleVLA-RL、RL-Co 等顶会工作的核心思路
4. **VLA 在线 RL Fine-tune**（终极形态）——参考 SimpleVLA-RL (ICLR 2026)：VLA 可直接用 RL 微调，outcome reward（二值成功/失败）优于 dense reward
5. **数据多样性 > 精度**——DLR 论文结论：多环境（厨房/工厂/办公室）× 多策略模式 > 单环境大量精确数据

---

## 七、全局实验总览

| 实验 | 架构 | 数据 | unseen 成功率 | 训练时间 | 关键结论 |
|-----|------|------|-------------|---------|---------|
| Baseline (F1-F3) | MLP BC 71K | 200ep, 9D, 无 goal | **0%** | 1min | Goal ambiguity → 必然失败 |
| V3a 固定 cube | MLP BC 71K | 10ep, 固定 cube | **100%** | 1min | 确认 root cause |
| V3b→V4b MLP+goal | MLP BC 71K | 100ep, 11D | **30%** | 15min | MLP 泛化天花板 |
| **V4c ACT** | **ACT ~45M** | **100ep, 11D+vision** | **60%** | **42min** | Transformer 泛化 |
| **V5 SmolVLA** | **SmolVLA 450M** | **100ep, 11D+vision** | **60%** | **6min** | 预训练视觉高效 |

---


*本文基于 [lerobot_from_zero_to_expert](https://github.com/ZJLi2013/lerobot_from_zero_to_expert) 项目的实验记录整理。项目使用 Genesis 仿真器 + LeRobot 框架 + Franka Panda 机械臂，完整实验代码已开源。*


