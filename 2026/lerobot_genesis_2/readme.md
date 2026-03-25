# Lerobot SO-101 × Genesis 仿真实战（二）—— 闭环验证溃败记

## 摘要

[上一篇](../lerobot_genesis_1/lerobot_genesis_practice.md)花了一周在 Genesis 里把 SO-101 的 cube 抓取跑通。这一周沿着"合成数据 → 策略训练 → 闭环验证"的管线继续推进，结果——**全部失败**。BC、ACT、SmolVLA 三种策略训练 loss 都正常收敛，GT action 回放 100% 成功，但闭环评估一律 0%。最终定位到一条因果链：5-DOF 零冗余 → IK basin 跳变 → 多模态 action → covariate shift 在 5 步内发散。这不是学习问题，是系统自由度结构问题。本文是一份完整的踩坑记录。

---

## 背景

上周的成果是：在 Genesis 中完成了 SO-101 的 grasp_center 标定、可行域扫描，并在单个 cube 位置上成功抓取。自然的下一步就是：

1. **数据生成**：在可行域内随机化 cube 位置，批量生成抓取轨迹
2. **策略训练**：用合成数据训练 BC / ACT / SmolVLA
3. **闭环验证**：在仿真中做 policy rollout，统计成功率

思路很清晰，每一步看起来都不难。但每个环节都有坑，而且坑与坑之间互相耦合。

---

## 一、合成数据：50% 的天花板

### 1.1 数据生成策略

基于上一篇确认的可行域，先在高置信区域做 cube 位置随机采样：

```
feasible core（≥4/5 z-slices 可行）：x ∈ [0.13, 0.19], y ∈ [-0.09, 0.06]
```

对每个采样位置，用 IK solver 规划完整的抓取轨迹（HOME → pre-grasp → descent → close → lift），记录为 LeRobot 格式的 episode。

### 1.2 成功率卡在 50%

G老师（GPT）跑了大量实验，尝试了各种策略来提升 IK 轨迹规划的泛化成功率：

| 轮次 | 策略 | 成功率 | 观察 |
|---|---|---:|---|
| A | 广域采样 + 默认 seed | ~50% | 多个 failure 在轨迹早期即进入"爪子朝上"的错误构型 |
| B | 收紧到 feasible core | ~50% | 说明不只是边界问题 |
| D | 强 hard gate（夹爪朝下 + 位置误差） | 0% | 约束过强，可行候选被全部拒绝 |
| F | 镜像 seed + 短 rollout + 自适应 gate | 10% | 模块叠加后排序失真，效果退化 |
| G | per-key top-2 → global top-3 → 延迟剪枝 | **50%** | 简化耦合后恢复稳定区间 |
| H1 | 在 G 上仅加诊断标注 | 50% | 选择机制可用（good_branch_exists=90%） |
| H2 | 在 G 上改 structured seeds | 40% | 覆盖能力被破坏 |

从 seed 扩展、cost 弱约束到 branch filter，尝试了一圈后，成功率始终稳定在 **50% 左右**。

### 1.3 根因：IK 种子 basin 不一致

G老师最终给出的结论很精炼：

> **50% 瓶颈的根因：同一个 IK 种子在不同工作空间位置下收敛到不同的 basin。**

`home_rad` 是一个固定的关节角向量。当 cube 位置在 feasible core 内变化时，从同一个 `home_rad` 出发的 IK 迭代会收敛到不同的 basin（elbow up/down, wrist flip）——这是 damped least-squares IK 的固有特性，不是代码 bug。

Phase G 的"延迟决策 + 多分支"策略本质是事后补救：先生成多个候选，再选最好的。H1 诊断证实选择机制本身可用（good_branch_exists_rate=0.9），但仍有 50% 失败。这说明：**"好分支"在规划阶段存在且被选中，但在执行阶段仍不稳定。**

结论：在 5-DOF 零冗余运动学 + `quat=None` 的 IK basin 不确定性下，scripted grasp 在 XY 泛化上存在硬天花板。**继续在管线里追求 >90% 成功率是错误的投入方向。**

### 1.4 实际执行

接受 50% 的现实后，数据生成策略变为：

- 先生成 200 ep，按 `gen_summary.json` 过滤 success/failure
- 仅保留 success episodes 作为主训练集（约 100 ep）
- 若 success 少于 120，继续补采样

最终采集了 300 ep，其中 147 success。

---

## 二、策略训练：Loss 正常收敛 ≠ 策略可用

### 2.1 三种策略对比

| | MLP BC | ACT | SmolVLA |
|---|---|---|---|
| 架构 | 2 层 MLP | Transformer Enc-Dec + ResNet | VLM (SigLIP + SmolLM2) + Action Expert |
| 输入 | state (6D) | state + images | state + images + language |
| 参数量 | ~70K | ~45M | ~450M |
| 预训练 | 无 | 无 | `lerobot/smolvla_base` |
| 显存 (4090) | < 1 GB | ~6 GB | ~14 GB |

### 2.2 训练配置（以 SmolVLA 为例）

- `n_steps=6000`, `batch_size=8`, `lr=1e-4`
- `pretrained=lerobot/smolvla_base`
- `freeze_vision_encoder=True`, `train_expert_only=True`
- 可训练参数 ~100M / 总 450M

### 2.3 训练结果

三种策略训练 loss 均正常收敛（BC: 0.001, ACT: 0.043, SmolVLA: 2.36）。单从训练角度看，基于 LeRobot 框架的训练本身是稳定的。

**但后续闭环验证证明：loss 收敛和策略可用是两回事。**

---

## 三、闭环验证：全线溃败

### 3.1 Step 0：Oracle Replay（管线完整性验证）

先排除管线本身的问题。直接从 LeRobot 数据集读取 GT action（绝对关节角度），逐帧喂给 Genesis PD 控制器，不经过任何模型推理：

```
LeRobot Dataset → action[t] (absolute deg) → control_dofs_position() → scene.step()
```

测试了两种模式：

| 模式 | 成功率 | 说明 |
|---|---:|---|
| `direct`（纯回放） | **100%** (5/5) | 验证 Genesis 物理 + 控制 + 判定逻辑 |
| `pipeline`（加 clamp + smooth ±8°/步） | **100%** (5/5) | 验证安全限幅不误杀正确轨迹 |

两种模式结果完全一致 → **管线验证通过，问题在模型侧。**

### 3.2 Step 1：闭环 Policy Rollout

每个 step 经过完整的 感知 → 推理 → 执行 → 物理更新 循环：

```
Genesis 渲染图像 + 读关节状态
  → SmolVLA/BC/ACT 推理
    → 后处理（反归一化 + clamp + smooth）
      → control_dofs_position()
        → scene.step()
          → cube_z 判定
```

**结果**：

| 策略 / checkpoint | 成功率 |
|---|---:|
| SmolVLA base（官方基座） | 0/10 = **0%** |
| SmolVLA fine-tuned（300ep 训练） | 0/10 = **0%** |
| MLP BC（全量 / 单 basin / noise aug） | **0%** |
| ACT | **0%** |

所有策略，所有配置，闭环一律 0%。cube 纹丝不动（avg_lift ≈ -0.0003m）。

### 3.3 根因定位

面对"replay 100% 但闭环 0%"的巨大反差，逐步排查：

**排查 1：模型是否学到了？**

将训练数据的真值图片喂给 fine-tuned SmolVLA，检查预测 action：

| 关节 | fine-tuned 预测误差 | base 预测 |
|---|---:|---|
| shoulder_pan | ~1° | ≈ 0°（未学习） |
| shoulder_lift | ~4° | ≈ 0° |
| elbow_flex | ~4° | ≈ 0° |
| wrist_flex | ~5° | ≈ 0° |

Fine-tuned 模型在训练图片上关键关节误差 1-5°，**不是没学到**。但 `freeze_vision_encoder=True` 意味着视觉特征完全冻结在 base 预训练分布上 → 闭环渲染的新图片 → 视觉特征变了 → action expert 退化到 base prior ≈ 0°。

**排查 2：退到最简单的 BC，是否可行？**

回退到 state-only MLP BC（无视觉），用 GT action injection 实验量化 covariate shift：

| N（每 N 步注入一次 GT action） | 自由飞行步数 | 成功率 |
|---:|---:|---:|
| 1（每步都用 GT） | 0 | **70%** |
| 5 | 4 | **0%** |

**Policy 仅能自由飞行 < 5 步（< 0.17s @ 30fps）就完全失效。**

---

## 四、为什么会这样？—— Covariate Shift 与数据本质缺陷

### 4.1 问题的本质

BC 训练的核心假设是：测试时的输入分布与训练时相同。但在闭环中，模型的输出决定下一步的输入——误差通过递归反馈自我放大：

```
Expert rollout (数据采集):
  s0 → a0=IK(s0) → s1 → a1=IK(s1) → ...      ← action 始终正确

Policy rollout (闭环评估):
  s0 → a0=π(s0) → s1' → a1=π(s1') → ...      ← s1' ≠ s1，因为 a0 有误差
                   ↑                              s1' ∉ 训练分布 → 误差更大
                   这里开始发散
```

BC 的累积误差随轨迹长度 $T$ 呈 $O(T^2)$ 增长（Ross & Bagnell, 2010）。

### 4.2 IK Expert 数据的致命缺陷

当前数据生成流程：

```python
for target_deg in ik_trajectory:
    so101.control_dofs_position(target_deg)    # IK 精确目标
    scene.step()
    state_deg = so101.get_dofs_position()       # 实际状态
    dataset.add_frame({
        "observation.state": state_deg,          # 全部在理想轨迹上
        "action": target_deg,                    # IK 精确目标
    })
```

**所有 `observation.state` 都在理想轨迹上**，数据中不存在"偏了怎么修正"的样本。训练出的 policy 等价于"只会在正确轨道上行驶，不会修方向盘"。

对比人类 teleoperation 数据：

| 属性 | IK expert 数据（本项目） | 人类 teleop 数据 |
|---|---|---|
| 状态分布宽度 | 极窄（仅理想轨迹） | 较宽（有噪声、抖动） |
| 是否包含 recovery | 否 | 是（人类会不断修正） |
| 等价理论框架 | 纯 BC | 隐式 DAgger |

### 4.3 IK 多解性：雪上加霜

SO-101 的 5-DOF 零冗余让 covariate shift 问题雪上加霜。同一个 cube 位置，IK solver 可能收敛到完全不同的 basin，差异集中在 wrist_roll（118 ep 的 t=15 统计：std=113.3°，范围 -157° ~ +163°）。

模型输入（6D 关节角）中，其他 5 维在不同 basin 间相似，**无法区分当前属于哪个 basin** → 同一个 observation 对应多个合法 action → MSE loss 强制回归条件均值 → 单个 episode 的误差可达 37°。

**这不是 overfitting，不是泛化问题——是 MSE 在 partial observability 下的必然结果。**

### 4.4 为什么别人的 BC 看上去能用？

初学者（包括我）的常见错觉："cube pick 是最基础的 manipulation 任务，BC 应该能轻松解决。"

**答案藏在数据里**：

1. **人类 teleop 数据天然包含 recovery**。人在采集时本身就在做闭环控制（偏了 → 看到 → 修回来），数据中天然含有 `(偏离状态, 修正动作)` 对。

2. **隐藏的任务简化**。常见教程里的 trick：

   | trick | 效果 | 本项目 |
   |---|---|---|
   | 低控制频率（10Hz） | 减少 error compounding 步数 | ✗（30Hz，120 步/ep） |
   | delta action | 系统更稳定 | ✗（绝对位置控制） |
   | 高容忍度任务（pushing） | 不需精确对齐 | ✗（抓取需 mm 级精度） |

3. **隐式的数据增强**。多次采集、失败重录、多操作者风格差异——本质上在扩展 state 覆盖范围。

---

## 五、尝试修补

### 5.1 Noise Augmentation

最简单的改进：训练时对 state 加高斯噪声 `state += N(0, σ)`，不改数据采集，只改训练脚本。

| 配置 | N=1 成功率 | N=5 成功率 |
|---|---:|---:|
| 原始 BC | 70% | **0%** |
| + noise σ=3° | **90%** | **10%** |

**首次打破 N=5 的零壁垒**，但本质局限明确：扩展了 state 分布，没改变 action 语义（仍用 clean expert label）。是止痛药，不是治病方案。

### 5.2 DAgger v0

初步尝试 DAgger：policy rollout → IK relabel → fine-tune。闭环仍 0%，但 ep4 出现首个有意义的 lift（0.034m, sustain=6/8 frames）。关键发现：expert label 必须经 `smooth_action` 平滑（Δ≤8°/step），否则与原始 lerp 插值的 expert data 产生矛盾（raw label loss: 0.22 → smoothed: 0.05）。

当前瓶颈：单轮 10 ep rollout 数据量不足，且 on-policy labels 全部停留在 descent 阶段。


---

## 六、总结

经过一周在 SO-101 上的系统实验，确认了一条完整的因果链：

```
5-DOF = 任务 DOF → IK 唯一解 → wrist_roll basin jump → 多模态 action
  → covariate shift 放大 → N=5 步即发散 → 闭环 0%
```

核心证据：

| 实验 | 发现 | 本质原因 |
|---|---|---|
| wrist_roll 诊断 | 误差 0.9° ~ 37.4°（basin 依赖） | 5-DOF IK 唯一解 |
| N-step GT injection | N=1: 70%, N=5: 0% | 自由飞行 < 5 步即发散 |
| noise augmentation | N=5: 0% → 10%（有上限） | state 扩展有效但 action 语义不变 |
| DAgger v0 | 0%（ep4 接近成功） | recovery label 受限于唯一 IK 解 |

**这不是学习问题，是系统自由度结构问题。80% 的时间花在了对抗物理约束。**

### 下一步：切换 Franka

| 维度 | SO-101 (5-DOF) | Franka (7-DOF) |
|---|---|---|
| 冗余度 | 0 | +1（null-space） |
| IK 解空间 | 唯一解（basin jump） | 连续一维自运动流形 |
| 轨迹连续性 | wrist_roll 可跳变 37°+ | 连续平滑 |
| Recovery 可行性 | 本质不存在 | 天然存在（null-space） |
| DAgger 可行性 | label 受限 | IK 从任意状态有连续解 |

---

### 参考文献

- Ross, S. & Bagnell, J.A. (2010). *Efficient Reductions for Imitation Learning.* AISTATS.
- Ross, S., Gordon, G. & Bagnell, J.A. (2011). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* AISTATS. (DAgger)
