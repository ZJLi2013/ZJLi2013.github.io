# Franka Genesis End-to-End Synthetic Data Pipeline in Practice

The background of this project is based on a simple cube pick-up experiment, aiming to build an end-to-end pipeline covering **data collection → policy training → simulation evaluation**. Prior to this, there were two weeks of trial-and-error with LeRobot SO-101, detailed in [LeRobot101 × Genesis Simulation Practice](../lerobot_genesis_2/readme.md). The key takeaway was that the SO-101, being a 5-DOF arm, easily becomes over-constrained in an idealized simulation environment, causing IK solver jumps and divergence under any small perturbation. Therefore, the 7-DOF Franka arm was chosen instead — its IK solver achieves 100% successful grasps right out of the box, and more importantly, it is robust and recoverable under small perturbations.

---

## 1. Background

The goal is to build a **synthetic data pipeline**: automatically generate robot grasping data in simulation → train a policy model → closed-loop evaluation.

Tech stack:
- **Simulator**: Genesis (GPU-accelerated physics simulation)
- **Robot**: Franka Emika Panda (7-DOF, industrial-grade precision)
- **Task**: Pick-Cube (grasp a red cube from the table and lift it)
- **Policy architectures**: MLP BC → ACT (Action Chunking Transformer) → SmolVLA (Vision-Language-Action)
- **Evaluation metric**: HOME-START success rate (robot starts from a fixed home position, policy has full autonomous control)

The core question of the entire project is: **Can automatically generated trajectory data from an IK (inverse kinematics) planner be used to train a deployable grasping policy?**

---

## 2. Experiment Journey

### 2.1 Starting Point: Perfect Data, Converged Model, but 0% Success Rate

The first round of experiments looked great on paper:

| Metric | Result |
|--------|--------|
| IK data collection success rate | 200/200 = **100%** |
| BC training loss | **0.0003** (extremely low) |
| HOME-START success rate | **0/10 = 0%** |

100% data collection success, extremely low converged loss, yet 0% deployment success. This is a classic Imitation Learning trap: **low loss ≠ correct policy**.

A diagnostic metric **E1 N-step** was introduced: inject one ground truth (GT) action every N steps, and measure the policy's "free flight" capability between GT injections:

| E1 Interval | Success Rate |
|-------------|-------------|
| N=1 (GT every step) | 100% |
| N=5 | 90% |
| N=10 | 60% |
| N=20 | 0% |

The policy performs reasonably within short horizons (5 steps), but completely fails once it runs freely for more than 20 steps. This is the classic **Covariate Shift**: small prediction errors accumulate step by step, driving the robot into states never seen during training, leading to total divergence.

### 2.2 The "Marathon" of Ruling Out Hypotheses

Over the next two weeks, 7 hypotheses were systematically tested to find the cause of the 0% success rate. Each hypothesis was rigorously validated through experiments:

#### ❌ Hypothesis 1: Adding Noise During Training Improves Generalization

Intuition: Add Gaussian noise to inputs so the model sees a broader state distribution.

```
Result: E1 N=5 plummeted from 90% to 0%
```

**Lesson**: Noise only expanded the input distribution but did not provide correct recovery labels. The model saw "deviated states" but the corresponding action labels were still "actions from the nominal trajectory" — a mismatch between question and answer.

#### ❌ Hypothesis 2: ACT's Action Chunking Can Solve Error Accumulation

Intuition: Predict multiple steps at once (chunking) to reduce single-step error accumulation.

```
Result: HOME-START still 0%, E1 N=5 dropped from 90% to 0%
Reason: 28.8M parameters trained for only 2000 steps (loss~0.8), severely underfitted
```

**Lesson**: The architectural direction may be correct, but insufficient training renders all hypotheses meaningless. ACT requires far more than 2000 training steps.

#### ❌ Hypothesis 3: DART Data Augmentation Can Teach the Policy to Recover

DART (Disturbances for Augmenting Robot Trajectories) is a classic IL data augmentation method: inject perturbations during data collection so the expert demonstrates how to recover from deviated states.

Two DART variants were implemented:
- **Jump DART**: One-shot jump perturbation
- **Drift DART**: Continuous micro-perturbations

```
Jump DART:  σ=0 ratio 11.8x (vs baseline 12.5x), almost no improvement
Drift DART: σ=0 ratio 12.89x, even slightly worse
```

It was later discovered that **what was implemented was not true DART at all**. The key premise of the original DART paper is that the supervisor must be a **closed-loop policy** — capable of providing correct actions for any state. However, the IK planner used was **open-loop**: it plans the entire trajectory once at the beginning of the episode. After perturbation, the action labels were still pre-planned interpolation targets, not re-solved IK from the current state.

```python
# Current implementation (open-loop, incorrect DART)
traj = plan_pick_trajectory(cx, cy)   # Plan entire trajectory upfront
for target in traj:
    perturb(state)                     # Perturb state
    execute(target)                    # But target is still pre-planned!
    record(state, target)              # Action label doesn't match current state

# Correct DART (closed-loop)
for each step:
    current = observe()
    target = solve_ik(current, goal)   # Re-solve IK from current state
    perturb_and_execute(target)
    record(current, target)            # Action label always matches current state
```

**Lesson**: When implementing an algorithm, always go back to the original paper to verify the prerequisites. An implementation with the same name but missing key components may be completely ineffective.

#### ❌ Hypothesis 4: Adding State History Provides Velocity Information

Intuition: Concatenate the previous two steps' states as input, allowing the model to use motion history to infer the current phase.

```
Result: σ=0 ratio worsened from 12.5x to 20.3x
```

**Lesson**: Temporal blindness is not the bottleneck. History information actually makes the model more fragile under out-of-distribution (OOD) states — it overfits to the temporal patterns in the training data.

#### ❌ Hypothesis 5: Closed-Loop IK Re-planning Can Fix Action Label Quality

After discovering the open-loop DART issue, true closed-loop IK was implemented: re-solve IK from the current configuration at every step.

```
Result: σ=0 ratio worsened from 12.5x to 20.80x
HOME-START still 0%
```

Analysis revealed that the perturbation magnitude of σ=0.0005 was too conservative (cumulative offset ~0.004 rad), far smaller than the actual error during deployment (~0.009 rad/step). Closed-loop IK did improve label quality, but when perturbations are not large enough, the policy still hasn't seen states at the real drift magnitude.

#### 🐛 Hidden Bug Found: State-Action Temporal Misalignment

During debugging, a timing bug was discovered in **all data collection scripts**:

```python
# Wrong (all scripts): execute first, then observe
execute(action)          # 1. Execute action
sim_step()               # 2. Step simulation
state = observe()        # 3. Read state (already post-execution!)
record(state, action)    # Records (s_{t+1}, a_t)

# Correct: observe first, then execute
state = observe()        # 1. Read state (s_t) first
execute(action)          # 2. Then execute
sim_step()               # 3. Step simulation
record(state, action)    # Records (s_t, a_t) ✓
```

Because the PD controller tracks tightly, after executing the action `s_{t+1} ≈ a_t`, the model effectively learned an **identity mapping f(x) ≈ x** — i.e., "stay in place."

After the fix, policy behavior improved (HOME displacement dropped from 0.109 to 0.049 rad), but HOME-START **remained 0%**.

**Lesson**: The temporal misalignment was a bug but not the root cause. Fixing the bug was necessary but not sufficient.

### 2.3 The Eureka Moment: Goal Ambiguity

After ruling out all the above hypotheses, the fundamental issue was finally realized.

During data collection, the cube position was randomized within `x ∈ [0.4, 0.7], y ∈ [-0.2, 0.2]`, but the observation consisted only of 9D joint angles — **cube position was not included**.

This means:

```
Same HOME state + cube=(0.40, 0.10) → action A (arm reaches forward-left)
Same HOME state + cube=(0.70,-0.20) → action B (arm reaches forward-right)
Same HOME state + cube=(0.55, 0.00) → action C (arm reaches straight ahead)
```

The MSE optimization result: `π(HOME) = mean(A, B, C, ...) = points toward a non-existent "average cube position"`

**The policy takes a wrong first step, and then covariate shift snowballs.**

This problem can be summarized in one formula:

```
What was learned:  π(s) → a        ← Not a single-valued function! (same s maps to multiple different a's)
What should be:    π(s, goal) → a  ← goal = cube position; given the goal, the mapping is uniquely determined
```

### 2.4 One Command to Verify the Root Cause

To verify this hypothesis, the simplest possible experiment was conducted: **fix the cube position** (zero code changes, only command-line arguments changed).

```bash
# Only change: --cube-x-min 0.55 --cube-x-max 0.55 --cube-y-min 0 --cube-y-max 0
```

| Metric | Cube Randomized | **Cube Fixed** |
|--------|----------------|---------------|
| HOME-START | **0%** | **100%** ✅ |
| lift | -0.0003m | **0.1554m** |

**From 0% straight to 100%.** This was the first HOME-START success in the entire experiment series.

Observation, model, and training parameters were all unchanged — the only difference was eliminating goal ambiguity. This also retrospectively explains **why all previous methods failed**:

| Previous Attempt | Why It Failed |
|-----------------|---------------|
| Noise augmentation | Expanded state distribution but labels were still goal-agnostic |
| DART | Perturbed data still lacked cube position, remained multi-modal |
| Closed-loop IK | Improved label quality, but observation still lacked goal |
| History | Historical states provided velocity but not goal |
| Timing fix | Corrected (s,a) pairing semantics, but s still didn't contain goal |

---

## 3. From 0% to 60%: The Architecture Upgrade Path

With the root cause confirmed, the solution was clear: **add goal information to the observation**.

### 3.1 MLP BC + Goal Conditioning: Ceiling ~30%

Observation was expanded from 9D to 11D (9 joints + cube_x + cube_y), but the MLP's generalization quickly hit a ceiling:

| Experiment | Unseen Cube Success Rate | Issue |
|-----------|------------------------|-------|
| 10 episodes | 40% (false signal*) | eval seed = training seed |
| 100 episodes | 0% | MLP overfits, "memorizes" rather than "interpolates" |
| 100ep + noise σ=0.01 | **30%** | MLP architecture limit |

> *An important lesson: when eval seed equals training seed, the eval cube positions are the same as training positions, effectively measuring "training set success rate." Henceforth, eval seed ≠ training seed was strictly enforced.

**Lesson**: MLP cannot effectively generalize in continuous goal spaces. With 100 episodes, the MLP learned to "memorize the exact actions for each training position," and any slight deviation during rollout triggers cascading failure.

### 3.2 ACT: The Generalization Advantage of Transformers

Switching to ACT (Action Chunking Transformer, ~45M params) with the same 100 episodes of data:

**Result: 6/10 = 60% success rate**

Compared to MLP BC's 30%, ACT's advantages come from:
1. **Transformer architecture**: Better interpolation capability in continuous spaces
2. **Action chunking**: Predicting multiple steps at once provides temporal consistency
3. **VAE regularization**: Prevents overfitting

### 3.3 SmolVLA: Efficient Training with Frozen Vision

SmolVLA (450M params, HuggingFace LeRobot official model) trains only the action expert (~50M), freezing the pre-trained SigLIP vision encoder and SmolLM2 language model:

**Result: 6/10 = 60% success rate, training takes only 6 minutes**

| Dimension | ACT | SmolVLA |
|-----------|-----|---------|
| Success rate | 60% | 60% |
| Training time | 42 min | **6 min** |
| VRAM | ~6 GB | ~2.2 GB |
| Parameters | ~45M (all trained) | 450M (only 50M trained) |

More interestingly, the two models are **complementary on different cube positions**: ACT exclusively succeeds on 2 positions, SmolVLA exclusively succeeds on 2 positions. A theoretical combination could reach 80%.

---

## 4. Current Limitations

### 4.1 Inference Relies on Oracle Cube Position

To be honest: **the current 60% success rate is achieved under "cheating" conditions.**

The inputs to ACT and SmolVLA include `cube_x, cube_y` — these values come directly from the simulator's `cube.get_pos()`, which provides ground truth world coordinates. In real deployment, these would not be directly available.

```
Current observation = [9D joints, cube_x, cube_y]  ← cube_xy from simulator GT
                                                      Not available in real deployment
```

This means the current evaluation validates "**given oracle information about the target position, can the action head correctly execute the grasp**", rather than "**can the model autonomously perceive the target from vision and execute**."

### 4.2 Why This Is Still Valuable

Although oracle information is used, this experimental phase validated several key conclusions:

1. **Goal Ambiguity is the root cause**: Without goal information, any policy achieves 0%
2. **The synthetic data pipeline is viable**: Data generated by the IK planner + Genesis simulator can train working policies
3. **Transformer architecture is critical for generalization**: MLP ceiling at 30%, ACT/SmolVLA reach 60%
4. **SmolVLA's pre-trained vision encoder enables highly efficient training**: 6 minutes to match ACT's performance

---

## 5. Distilled Insights: Pitfall Summary & Best Practices

### 🔴 Pitfall Checklist

| # | Pitfall | Cost | Lesson |
|---|---------|------|--------|
| 1 | **Low loss ≠ correct policy** | Wasted significant time on "why does the loss converge but the policy doesn't work" | MSE converging to the conditional mean E[a\|s] is mathematically correct, but if the mapping is multi-modal, the conditional mean is a non-existent "average action" |
| 2 | **Timing bug hidden in all data scripts** | All experiments ran on incorrect data | Always write diagnostic scripts first to verify the semantic correctness of (s,a) pairings |
| 3 | **Eval seed = Training seed** | 40% false signal | Strictly enforce eval seed ≠ training seed, test on unseen data |
| 4 | **Didn't check DART prerequisites in the original paper** | Two rounds of DART experiments wasted | Before implementing any algorithm, verify that its prerequisites are met |
| 5 | **Noise augmentation doesn't provide recovery labels** | E1 N=5 plummeted from 90% to 0% | Adding noise to inputs ≠ teaching the model to recover; correct action labels must accompany the perturbed states |
| 6 | **MLP "memorizes" with more data** | 100ep actually worse than 10ep | Simple architecture + more data = overfitting; need appropriate architectural complexity |
| 7 | **Ignoring observability** | Entire first half of experiments | The first step in designing an IL system is confirming that π(obs) → a is a single-valued function |

### 🟢 Best Practices

1. **Phase 0: Verify that π(obs)→a is a single-valued function first** — List all randomized variables, confirm the observation contains sufficient information to distinguish them; use fixed-parameter ablations to confirm
2. **Diagnose before fixing** — Build diagnostic tools (E1 N-step, D1 σ=0 ratio, timing diagnostics, etc.) first, then decide what to change
3. **Add complexity incrementally** — Fixed cube → oracle goal → vision inference; confirm each step works before moving to the next
4. **Three gaps in synthetic data** — Open-loop → closed-loop (per-step IK), off-trajectory coverage (1% → 100%), joint-space → task-space interpolation
5. **Evaluation rigor** — eval seed ≠ training seed, HOME-START full autonomy as the gold standard, record specific results for each episode

---

## 6. Next Steps for VLA Post-Training

Currently at the "oracle cube position + action head fine-tune" stage. Roadmap toward a deployable VLA:

1. **Remove Oracle, Let Vision Handle Goal Inference** (most critical) — Transition from `π(joints, cube_xy, images)` to `π(joints, images)`. SmolVLA's SigLIP pre-trained encoder already has zero-shot object recognition capability and is the preferred path; expected success rate drop from 60% to ~40%, may require LoRA fine-tuning or more data (200+ episodes)
2. **Improve Visual Realism** — Current flat-color rendering has a large gap from the real world. Path: Rasterizer (L1) → RayTracer (L2) → **Blender + Gaussian Splatting (L3)**. Key insight: RL training doesn't need realistic visuals (state-based), but VLA training data must be realistic
3. **RL Expert as Recovery Oracle** — Pure IK data lacks recovery supervision. Genesis's official two-stage framework (PPO teacher → DAgger BC student) is the most mature approach, also the core idea behind top-venue works like DiffRL→VLA, SimpleVLA-RL, RL-Co, etc.
4. **VLA Online RL Fine-tuning** (ultimate form) — Reference: SimpleVLA-RL (ICLR 2026): VLAs can be directly fine-tuned with RL, and outcome reward (binary success/failure) outperforms dense reward
5. **Data Diversity > Precision** — DLR paper conclusion: multiple environments (kitchen/factory/office) × multiple policy modes > large amounts of precise data from a single environment

---

## 7. Full Experiment Overview

| Experiment | Architecture | Data | Unseen Success Rate | Training Time | Key Conclusion |
|-----------|-------------|------|-------------------|--------------|----------------|
| Baseline (F1-F3) | MLP BC 71K | 200ep, 9D, no goal | **0%** | 1min | Goal ambiguity → inevitable failure |
| V3a Fixed Cube | MLP BC 71K | 10ep, fixed cube | **100%** | 1min | Root cause confirmed |
| V3b→V4b MLP+goal | MLP BC 71K | 100ep, 11D | **30%** | 15min | MLP generalization ceiling |
| **V4c ACT** | **ACT ~45M** | **100ep, 11D+vision** | **60%** | **42min** | Transformer generalization |
| **V5 SmolVLA** | **SmolVLA 450M** | **100ep, 11D+vision** | **60%** | **6min** | Pre-trained vision efficiency |

---

*This article is based on experiment records from the [lerobot_from_zero_to_expert](https://github.com/ZJLi2013/lerobot_from_zero_to_expert) project. The project uses the Genesis simulator + LeRobot framework + Franka Panda arm. Complete experiment code is open-sourced.*
