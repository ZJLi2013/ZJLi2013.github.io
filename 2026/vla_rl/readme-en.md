# From Imitation to Imagination: How VLA Breaks Through the Ceiling with RL and World Models

> English | [中文](readme.md)

Over the past two years, VLA (Vision-Language-Action models) has raced ahead on imitation learning, yet keeps hitting the same ceiling: **no matter how well it mimics, it can't surpass the demonstration data**. In 2026, the whole field's attention converged on one question — how to let robots **self-improve with reward**, and how to sidestep the "real-robot trial-and-error is too expensive" barrier by simply **trying and failing cheaply inside a learned "world model."**

This article follows that line and explains it top-down: why VLA needs RL (§1) → how the choice of action head decides the entire RL route (§2) → the core obstacle and solution for doing RL with a flow head (§3) → using three orthogonal axes to place the sprawling body of work in one shot (§4); then turning to "where experience comes from" (§5), leading into the full world-model-RFT spectrum of using a world model as a low-cost rollout engine (§6), and its reward-free parallel line, WAM (§7); finally converging on an underrated but worthwhile direction (§8) — **rather than curing a world model's "hallucination," just use real physics to build a world model that inherently cannot hallucinate.**

---

## §1 The paradigm: why VLA needs RL

The mainstream way to train a VLA (Vision-Language-Action) is imitation learning — behavior cloning on expert-demonstrated `(observation, action)` pairs. The fundamental limitation of this path is that **the ceiling equals data quality**: the policy learns "what action the expert chose at this step," so it can approach the expert but cannot discover a policy better than the expert; and for long-tail states or error-recovery situations the demonstrations don't cover, imitation learning has nothing to work with.

Reinforcement learning fills exactly this gap. It doesn't ask "what would the expert do," but "what action earns higher return," and so it can **surpass the demonstration** and learn to correct itself in closed loop. So-called VLA + RL is, in essence, upgrading VLA from "mimicking the expert" to "self-improving with reward."

But this step isn't free. RL introduces three old hard problems — credit assignment, exploration, and reward design — and in the embodied setting adds the most expensive constraint of all: **real-robot interaction is expensive, slow, dangerous, and sample-inefficient.** This constraint runs through the whole article — it's both the reason the world model appears later and the origin of this article's entry point.

---

## §2 The fork: how the action-head debate decides the RL route

To do RL on a VLA, the first thing isn't to pick an algorithm, but to look at how its **action head** emits actions — because that directly decides whether RL can be done, and how.

Today's action heads come in three paradigms:

| Paradigm | Representative | How it produces actions | Naturally has computable log π? |
|---|---|---|---|
| **Discrete token autoregression** | OpenVLA, π0-FAST | actions binned into tokens, generated token-by-token like an LLM | **Yes** (softmax naturally) |
| **Diffusion head** | Octo, Diffusion Policy | iteratively denoise from Gaussian into continuous action chunks (16–32 steps) | No (deterministic, needs modification) |
| **Flow-matching head** | **π0 / π0.5 / GR00T** | learn straight-line transport noise→action (4–8 steps, several times faster than diffusion) | No (deterministic ODE, needs modification) |

There's no single winner, but the trend is clear: in high-frequency, dexterous, contact-rich manipulation, generative heads (flow / diffusion) dominate — π0 reaches 50Hz with flow, while autoregressive discrete heads have a hard temporal-resolution ceiling; and flow beats diffusion on latency (fewer steps on a straight-line path), so the current frontier is flow-matching.

The key is the last column. **Whether the action head has a naturally computable log π directly decides the first hurdle of RL:**

- **Discrete autoregression**: softmax naturally has log π, RL applies directly, PPO / GRPO both work (representative: SimpleVLA-RL).
- **flow / diffusion**: actions are integrated step by step by a **deterministic ODE**, so **log π can't be computed** — and policy gradients must have log π (to compute the importance ratio). So for this path, RL's first step isn't optimization, but **manufacturing a computable log π first.**

This is the starting point of the entire flow-RL stack.

---

## §3 The core obstacle and solution: manufacturing a log π for deterministic flow

A deterministic ODE has no log π, and the solution has only one idea: **inject randomness so each denoising step becomes a Gaussian transition with computable probability.** There are two implementations:

- **Flow-Noise**: attach a learnable noise network (Sigma Net); the mean is still the original flow's one-step Euler, and the variance is output by the noise network — so each denoising step becomes an isotropic Gaussian, and log π can be computed exactly. Simple, exact, with learnable exploration strength.
- **Flow-SDE**: mathematically rewrite the probability-flow ODE into an equivalent SDE (adding a score-compensation drift + noise schedule), likewise turning each step into a Gaussian transition. More "orthodox," but introduces approximation.

Both turn "one deterministic trajectory" into "a string of Gaussian samples," so the joint log π of K denoising steps = the sum (or average) of each step's Gaussian log-prob, and the importance ratio and policy gradient can both be computed.

There's also a structure specific to embodied flow-RL hidden here — the **two-layer MDP**:

```
Outer (environment):  o_t --a_t--> o_{t+1} --a_{t+1}--> ...     real task reward each step
                          ↑
Inner (denoising):    noise → ... K Gaussian transitions ... → action a_t     inner-step reward = 0, hand off to outer at τ=1
```

The outer layer is the real environment interaction (task reward each step), and the inner layer is the K-step denoising process of "generating one action" (intermediate-step reward is 0, and at τ=1 it hands the action to the outer layer). What RL optimizes is actually this nested structure. Only by getting it straight can you understand which layer Flow-GRPO, π-RL, and VLA-RFT are each talking about.

---

## §4 Three orthogonal axes: placing all the work

Once you understand how log π is manufactured, you'll find that although there's a lot of work in this field, it all lands on three **mutually orthogonal** axes:

```
Axis A  action head:      discrete autoregression ── diffusion ── flow-matching     (decides whether to manufacture log π)
Axis B  optimizer:        PPO (with critic + GAE) ── GRPO (group-relative advantage, no critic)
Axis C  env transition:   real robot ── simulation ── world-model imagination        (who provides o_{t+1}=f(o_t,a_t))
```

A given work is one point on each of the three axes:

| Work | Action head | Optimizer | Env transition | In one line |
|---|---|---|---|---|
| **Flow-GRPO** | flow (image generation) | GRPO | — (image gen, non-embodied) | pioneer of manufacturing log π for deterministic flow + GRPO, non-embodied |
| **π-RL** | flow (π0/π0.5) | **PPO** | simulation | Flow-Noise / Flow-SDE two implementations; on long horizon, PPO > GRPO empirically |
| **SimpleVLA-RL** | **discrete autoregression** | GRPO | simulation | naturally has log π, skips manufacturing it, compares directly within a group |
| **VLA-RFT** | flow (Flow-Noise) | GRPO | **world model** | swaps the experience source from simulation to a learned world model |

Two points worth remembering. First, **the action head and the optimizer are orthogonal**: Flow-Noise can pair with PPO (π-RL) or GRPO (VLA-RFT) — don't bind "flow" to a particular optimizer. Second, the PPO vs GRPO trade-off is about the critic — PPO has critic + GAE, stronger at long-horizon, fine-grained credit assignment (π-RL empirically wins with PPO on embodied long horizons); GRPO uses the relative goodness of a group of samples as the baseline, critic-free, suited to single-step / short-horizon.

What really made this field branch into a whole new area of work in 2026 is **Axis C (env transition).**

---

## §5 Where env transition comes from: real / sim / world model

RL must interact with an environment to gather experience, and the most expensive thing about embodied RL is exactly this environment. The three options of Axis C are essentially answering "**who provides the env transition `o_{t+1}=f(o_t,a_t)`**" (note: what these three differ on is transition/dynamics; reward is often configured separately, and world models in particular need an external reward):

```
real robot   → most real experience, perfect grounding → but expensive, slow, dangerous, sample-inefficient
simulation   → cheap and parallel, ground-truth reward → but has asset/scene build cost + sim-to-real gap
world model  → use the "learned environment" as an imagination engine → cheap, parallel, but hallucinates, has grounding gap
```

The third is the hot spot of 2026. So-called **world model as rollout engine** means replacing the "environment layer" in the previous section's two-layer MDP — from real robot/simulation — with a **learned, action-conditioned world model**: give it the current observation and an action, and it predicts the next frame. This way the policy can try and fail en masse in "imagination," precisely striking the bottleneck of expensive real-robot interaction.

But recognize its position: **the world model doesn't replace RL**, it only changes "where experience comes from"; the RL hard problems of credit assignment, exploration, and reward design remain untouched — so this kind of work **still runs RL internally.** It trades for sample savings, at the cost of a new tax: **the grounding gap between imagination and reality** (model bias compounds and amplifies along the rollout), which needs a little real-robot interaction + representation alignment to correct.

And precisely because of this "hallucination tax," the entire spectrum that follows is essentially wrestling with it.

---

## §6 The world-model-RFT spectrum: five generations of evolution and two fatal flaws

The pioneer of this line is **VLA-RFT** (2510.00406): the policy uses Flow-Noise, the optimizer is GRPO, and experience comes from a learned autoregressive world model. Its most ingenious part is **how the reward is defined** — not an oracle of task success, but "the visual similarity between the policy rollout and the expert rollout."

There's an easily confusing point here: where does the ground-truth come from to compare the frames the world model generates? The answer is roundabout but crucial: **the reference side is also generated by the world model.** Feed the real expert actions **back through the same world model once more**, get WM(expert), and compare it against WM(policy). Why so roundabout? Because the world model is imperfect and its generated frames carry artifacts; if you compare WM(policy) against real frames, you mix "action difference" with "generation-quality difference." Whereas WM(policy) vs WM(expert) uses the same model, so **the generation bias appears on both sides and cancels out under subtraction, leaving only the action difference** — like using the same color-skewed camera to take one shot each and then comparing, so the color skew cancels naturally.

But this also exposes VLA-RFT's ceiling: what it verifies is "**how much it looks like the expert**," not "**whether it's correct**" — it's essentially imitation-in-outcome-space, and **its ceiling is still expert quality.** Add that its world model is only a 138M task-specific small model, and these are the two fatal flaws named in its own Limitations: **① the WM is too weak; ② the reward relies on expert similarity and can't scale.**

Almost all the 2026 follow-ups attack these two points, forming a rapidly iterating spectrum (conceptual generations, not a strict timeline):

| Gen | Theme | Representative | The problem it attacks |
|---|---|---|---|
| **Gen 1** | prove feasibility | **VLA-RFT** (2510.00406) | first complete "learned WM as simulator + RL fine-tuning of flow VLA" pipeline |
| **Gen 2** | cure hallucination | **WoVR** (2602.13977) | the longer the rollout the more off the rails — that's not the reward's disease, it's the WM's own |
| **Gen 3** | task-agnostic WM | **RAW-Dream** (2605.12334) | why retrain a WM for every task? decouple physics from task + VLM as reward |
| **Gen 4** | physical consistency | **RehearseVLA** (CVPR 2026) | video WM "looks good, physics bad" → need physically trustworthy + able to judge termination |
| **Gen 5** | runtime verification | **Pre-VLA** (2605.22446) | stop blindly improving the WM; verify before feeding the WM/executing, and block bad rollouts |

The through-line across these five generations is a shift of gravity: **from "make the world model more accurate" to "how to still do stable, effective RL when the world model is imperfect."** WoVR limits the impact of hallucination on optimization, RAW-Dream decouples physics from task and uses an off-the-shelf VLM to zero-shot judge success/failure as reward (incidentally solving "reward doesn't scale"), RehearseVLA introduces physical consistency and termination judgment, and Pre-VLA simply blocks bad samples at the entrance — all different facets of the same idea.

---

## §7 Another path: WAM — raising the ceiling reward-free

The entire spectrum above is doing RL (using reward to break through the ceiling). But the same "world model" also grew a parallel line that does no RL at all: the **World-Action Model (WAM).**

The crux of the distinction is the **direction of action flow**:

```
Action-Conditional WM (the simulator of Axis C):  (o, a) ──► o'      action is the [input]; it is the "environment"
World-Action Model (WAM):                         o ──► ô_future ──► a   action is the [output]; it is an "agent that imagines"
```

The former (ACWM) is the kind used as an RL environment in §5 — action is input, it predicts consequences, doesn't decide for itself, and must have an external policy attached. The latter (WAM) wires "predict the future" back into the action path: first predict the future (pixels/latent/features), then emit an action based on that. It's trained with behavior cloning (the action loss is BC, the world-model loss is self-supervised prediction), and is therefore **reward-free** — it doesn't break through the ceiling, but uses a smarter architecture + broader data (it can ingest action-free human video) to **raise the ceiling of imitation learning higher.**

Interestingly, ACWM and WAM are often **two sides of one coin**: many unified models (UWM, Motus, DreamZero) can switch identity by "which modality is the condition and which is the generation target" — give `(o,a)` and produce `o'` and it's ACWM; give `o` and produce `a` and it's WAM. This is exactly what "world model as substrate" means.

The 2026 leitmotif of the WAM line is **Dream Less, Act More**: Fast-WAM (2603.16666) offers an almost knockout conclusion — **the value of future prediction is in representation learning during training, not in imagination at inference** — so it keeps video co-training but simply skips future-pixel generation at inference, for 4× inference efficiency; LaWAM (2606.15768) moves the future into latent space and emits a subgoal in one shot, for 24× inference efficiency. This line is converging on "learn representations from video during training, generate as little as possible at inference," and is currently the **most deployable** branch.

So the core: **WAM (axis 2) raises the imitation ceiling, RL (axis 1) breaks through it — the two are orthogonal and stackable.**

---

## §8 A worthwhile direction: using real physics as a "world model that can't hallucinate"

Threading the above together, this whole swath of 2026 work — WoVR, RAW-Dream, RehearseVLA, Pre-VLA — is almost all battling the same enemy: **the learned world model fabricates physics** (hallucination, imagination drift, physical inconsistency). They spend great effort curing hallucination, forcibly manufacturing physical consistency, and blocking bad rollouts at runtime.

Following this observation, one overlooked lever is worth mentioning: **what if the "world model" is itself real physics?** A GPU physics simulator (like Genesis) plus the photorealistic rendering of 3D Gaussian Splatting is, in essence, a **perfect action-conditional world model** — real physics's `(o,a)→o'`, except it isn't learned, **fundamentally cannot hallucinate, and is inherently physically consistent.** What others spend a CVPR paper approximating, it gets for free; the hallucination others struggle to cure simply doesn't exist for it.

This path is also smooth in engineering: the **RLinf** framework where WoVR lives already has the harness built — supporting π0/π0.5/OpenVLA-OFT + GRPO/PPO + a bunch of real simulators, and the **environment backend is swappable.** WoVR is just the ready-made implementation of "environment backend = learned Wan world model"; swap the environment backend for real physics simulation + photorealistic rendering, and you get a control arm of a "real-physics environment backend." Same framework, same policy, same algorithm, only the environment layer swapped — a textbook-clean comparison.

And this direction need not be tied to RL. The same "real physics + photorealistic rendering" infrastructure can play different roles under different goals:

| | **Research path (RL)** | **Deployment path (WAM / fine-tuning off-the-shelf VLA)** |
|---|---|---|
| Goal | a distinctive contribution | shortest path to a usable system |
| Train a base model? | No | No |
| Do RL? | Yes (core) | No (optional later) |
| Pipeline | heavy (RL closed loop + reward + WM) | light (linear IL fine-tuning) |
| Time to effect | slow, uncertain | fast, deliverable |
| Role of the real-physics infra | hallucination-free RL environment | data engine + automated evaluation base |

A more general judgment: **in this wave, the moat may not be in the model, but in the environment/data layer.** Whoever holds a real-physics, hallucination-free environment/data pipeline can simultaneously supply RL (as a clean environment) and IL/WAM (as a data and evaluation base). Rather than squeezing into the red ocean of "curing learned-world-model hallucination," sidestep hallucination at the source — this is the directional judgment this article wants to leave.

---

## Closing thoughts

Looking back at this line, one sentence threads it: the main axis of VLA + RL has always been "surpass the demonstration with reward" — imitation learning makes a robot learn to be **like** the expert, and reinforcement learning is what makes it learn to be **better.**

And the real excitement of 2026 all happens at the "where experience comes from" link. Real robots are too expensive, simulation has a gap, so everyone turned their eyes to the world model: let the robot try and fail "dreamily" in a learned world. But dream too long and it distorts — hallucination, drift, physical inconsistency became the new tax of this path. So we saw a whole swath of work quietly shift from "make the world model more accurate" to "how to still do RL stably when the world model is imperfect."

And it's exactly at this turn that this article wants to leave a perhaps-underrated judgment: rather than expending effort to cure a world model's hallucination, switch the mindset — use real physics simulation plus photorealistic rendering to build a world model that **inherently cannot hallucinate.** It can serve both as a clean RL environment in research and as a reliable data and evaluation base in deployment.

From "watch how the expert does it," to "try and fail on your own," to "try and fail in imagination" — what robots are learning may not be just actions, but imagination itself.

---

## References

Surveys / frameworks:

- World Action Models: A Survey — [arXiv:2606.20781](https://arxiv.org/abs/2606.20781) (NUS; three families — rendered / latent / video-free — by "how much future is generated")
- RLinf — [github.com/RLinf/RLinf](https://github.com/RLinf/RLinf) (supports π0/π0.5/OpenVLA-OFT + GRPO/PPO + multiple simulators, swappable environment backend)

flow-RL main line:

- Flow-GRPO — manufacturing log π for deterministic flow + GRPO (image generation)
- π-RL — [arXiv:2510.25889](https://arxiv.org/abs/2510.25889) (Flow-Noise vs Flow-SDE; PPO)
- ReinFlow — [arXiv:2505.22094](https://arxiv.org/abs/2505.22094) (origin of Flow-Noise)
- SimpleVLA-RL — discrete action head × GRPO (parallel control)

world-model-RFT spectrum:

- VLA-RFT — [arXiv:2510.00406](https://arxiv.org/abs/2510.00406) (Gen 1, Flow-Noise × GRPO × world model)
- WoVR — [arXiv:2602.13977](https://arxiv.org/abs/2602.13977) (cure hallucination, KIR + masked GRPO + PACE; open-sourced with RLinf)
- RAW-Dream — [arXiv:2605.12334](https://arxiv.org/abs/2605.12334) (task-agnostic WM + VLM reward)
- RehearseVLA — CVPR 2026 (physically consistent simulator + instant reflector)
- Pre-VLA — [arXiv:2605.22446](https://arxiv.org/abs/2605.22446) (runtime verification)

WAM (reward-free parallel line):

- Fast-WAM — [arXiv:2603.16666](https://arxiv.org/abs/2603.16666) (the value of future prediction is in training-time representation; skip generation at inference)
- LaWAM — [arXiv:2606.15768](https://arxiv.org/abs/2606.15768) (latent subgoals, 24× speedup)
- UWM — [weirdlabuw.github.io/uwm](https://weirdlabuw.github.io/uwm/) · DreamZero · Motus (CVPR 2026) (unified models switchable among ACWM/WAM/IDM)
