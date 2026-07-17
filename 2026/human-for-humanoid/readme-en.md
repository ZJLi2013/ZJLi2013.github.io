# Human-Centric Data for Humanoids

> English | [中文](readme.md)

Producing robot manipulation data at scale from **human video** — rather than teleoperation, motion capture, or dedicated capture rigs — is becoming a paradigm-level shift. This article starts from the top-level paradigm and converges, layer by layer, onto one concrete and deployable technical line: dexterous hands plus explicit 3D reconstruction. It then places that line inside three data-generation pipelines — real, world model, and simulation — to see clearly what is changing in this direction.

---

## §1 The paradigm: why human → humanoid

The fundamental bottleneck of robot learning has always been **data**. Teleoperation and motion capture are both expensive, and the data they collect is locked to a single embodiment and a single scene — it doesn't scale. Human video is exactly the opposite: massive, diverse, and nearly zero cost to collect — web video like YouTube and HowTo100M, plus wearable first-person datasets like Ego4D and Ego-Exo4D, offer a scale and diversity that collected data simply can't match. The paradigm shift, in essence, is replacing "data comes from robots" with "data comes from humans."

The catch is that human video can't be used directly. Video has pixels but no robot-executable action labels; there's no proprioception; and the kinematics of human hands and bodies are simply not the same as a robot's kinematics and control interface. In one line: human video ≠ robot action, and essentially all work in this field is about crossing that gap in a principled way.

For a systematic overview, two 2026 surveys are worth reading first. One is *From Human Videos to Robot Manipulation* (arXiv:2606.00054), which takes the VLA perspective and, by "type of representation bridge," splits methods into four bridges: latent, world model, explicit 2D, and explicit 3D. The other is *Robot Learning from Human Videos: A Survey* (arXiv:2604.27621), which takes the more general LfHV perspective and, by "level of information flow," splits into three layers: task, observation, and action. The two are orthogonal ways of slicing the same space and don't conflict — one cares about what representation the bridge is, the other about which layer of information the bridge transfers.

What really ties these methods together is a shared essence: the only thing human video lacks is the **action** label, so crossing the human-to-humanoid gap amounts to finding a way to manufacture the missing action channel. The four seemingly disparate bridges are really four different means of manufacturing action supervision:

| Bridge | How it manufactures action supervision |
|---|---|
| Explicit 3D | Directly recover the 3D hand trajectory (MANO) — literally the action label |
| Latent action | Self-supervised compression into pseudo-action latents |
| World model | Predict future frames; action is a byproduct of internal features |
| Explicit 2D | Point trajectories / optical flow as weak action proxies |

(This table only concerns how each bridge fills in action; each bridge's representative works, and its position on the scalable⟷grounded spectrum, are left to §2.)

Following this view, one common misconception is worth clearing up: many assume human video can only feed world models or visual representations — things that don't need action labels. Not so. As long as you manufacture the action channel, human video can be used directly to pretrain a VLA. VITRA does exactly this, using 3D hand trajectories as action labels, proving the path of "human video as VLA pretraining data" works. The world model is just one of four ways to fill in action — not the only outlet.

---

## §2 Landscape: three axes and the representation-bridge spectrum

Work in this direction looks messy, but almost any of it lands on three axes:

```
Axis A  representation bridge:  latent action ── world model ── 2D cues ── 3D reconstruction
                                (scalable / weakly grounded)              (strongly grounded / executable)   ← HaWoR / Do-as-I-Do

Axis B  target embodiment:      floating dexhand ── fixed-base bimanual ── whole-body humanoid (loco-manipulation)
                                                                          ← EgoHumanoid / Human-as-Humanoid

Axis C  data source:            internet third-person video ── wearable ego capture ── generative world model (synthetic)
                                                                                       ← Wh0
```

A given work is just one point on each of the three axes. For example, Do-as-I-Do is "3D reconstruction / dexhand / real internet video," Wh0 is "world model with 3D annotation / dexhand policy / synthetic video," EgoHumanoid (2602.10106) and Human-as-Humanoid (2606.32009) are "3D whole-body retarget / whole-body humanoid / ego(-exo) capture," and LUCID (2606.11628) is "latent intent / embodiment-agnostic / unstructured internet video."

Of the three axes, Axis A (representation bridge) is the main axis of this field and the main thread of this article, so it deserves a dedicated treatment. Laying out the intermediate bridge from human video to robot action gives a continuous spectrum from implicit and scalable to explicit and executable:

| Bridge / direction | Representative works (2026) |
|---|---|
| Latent action (implicit action) | Being-H0, VITRA, EgoVLA, LAPA/Genie-style latent action |
| Predictive world model | Wh0 (2606.22136), UniPi-style |
| Embodiment-agnostic intent | LUCID (2606.11628) |
| Explicit 2D cues | Track2Act / ATM family, point-flow methods |
| Explicit 3D reconstruction | HaWoR, Do-as-I-Do (2606.19333), VideoManip (2602.09013), V2P-Manip (2606.16436), DexMan (ICLR'26), DexMit (2602.10105) |

Each end of the spectrum has its trade-offs. The implicit end scales strongly across video sources but is weakly grounded; the explicit-geometry end is strongly grounded with a clearer executable interface to the robot, at the cost of a heavy reconstruction pipeline. The core judgment offered by the surveys is that the two tracks run in parallel, not that one replaces the other: explicit geometry is the more immediately deployable track — high dexterous precision, clear executable interface — and is the current answer for dexterous hands; while latent action and latent world models are the track that can truly ingest web-scale video, self-supervising pseudo-actions from raw video without 3D labels, and therefore absorbing internet-scale footage, whereas explicit 3D must reconstruct frame by frame, making data both expensive and scarce, and its scale inevitably limited. In short, explicit 3D wins on precision and near-term deployability; latent wins on scale and the long-term ceiling.

On a longer timescale, three trends are underway. First, from hands to whole body: from tabletop bimanual manipulation to whole-body loco-manipulation — humanoid-ization in the true sense. Second, from real to synthetic: generative world models are hoped to serve as "infinite data generators" to fill the distribution blind spots of real video — NVIDIA Cosmos+GR00T-Dreams, DreamGen, and Wh0 are all on this line, though it remains more of a directional investment than a mature paradigm (the physical consistency of generated video is unsolved, and 3D labels still have to be back-inferred by a perception stack). Third, the decoupling of two kinds of supervision: separating intent and representation (learned from video, scalable) from control (learned in massively parallel simulation, executable) is widely seen as the most promising route, with LUCID as the representative.

---

## §3 Converging to the dexhand: why explicit 3D reconstruction is mandatory

The dexterous hand sits at the most explicit end of Axis A — not out of preference, but forced by the task objective. A parallel gripper only needs a grasp point plus open/close, for which 2D cues or latent action suffice; but a dexterous hand must control the contact and force closure of every finger, which requires fingertip-level 3D hand pose, the object's 6-DoF, and metric scale — otherwise the retargeted action is simply not physically executable.

So emphasizing 3D hand-motion reconstruction is a hard requirement for the dexhand direction: it provides the only physically grounded signal that can feed dexhand retargeting and RL. Switch the embodiment — say, to a gripper — or learn only high-level intent, and you no longer need to shoulder such heavy 3D reconstruction.

To be clear: this article converges on explicit 3D because of the dexhand's hard requirement and its near-term deployability, not because it is more scalable. Quite the opposite — as already noted, the track that can truly absorb web-scale video is latent, and the ceiling of explicit 3D is precisely frame-by-frame reconstruction with expensive, scarce data. The fuller picture: short term, use explicit 3D for precision and to win deployment; long term, rely on latent for scale — the two tracks are complementary.

---

## §4 The perception primitive: HaWoR

The explicit-3D bridge of §3, brought down to implementation, is a perception primitive — turning video into metric-scale 3D hand motion. The most mature piece today is HaWoR (CVPR'25, *World-Space Hand Motion Reconstruction from Egocentric Videos*), which can turn arbitrary first-person or internet human video into continuous, metric-scale 3D hand-motion trajectories in a unified world coordinate frame.

Internally it's stitched from several parts: hand detection uses WiLoR's YOLO detector, giving per-frame hand boxes, left/right handedness, track ids, and segmentation masks; hand-pose regression is HaWoR's own ViT plus temporal IAM plus MANO, outputting hand pose in the camera frame; the camera trajectory uses masked DROID-SLAM, masking out the dynamic hands and estimating camera pose only from the static background; metric scale comes from Metric3D v2's ViT-Large depth, used to align SLAM's up-to-scale trajectory; and when hands leave the view, an infiller network completes the gaps.

HaWoR's significance is that it's a reusable perception primitive: downstream work doesn't have to reinvent "how to understand hands" — it just builds on HaWoR's output. But as the next section shows, what really determines where the data comes from is not the primitive itself, but the video source feeding it — or bypassing it.

---

## §5 Three data generators: real, WM, sim

With a perception primitive in hand, the real question becomes where to continuously manufacture training data with 3D grounding. Abstracting the "generator," the difference is not in the video but in how the 3D grounding labels are obtained — three pipelines in total:

```
raw real video   → video is real but unlabeled → use the perception stack (HaWoR) to "reconstruct" labels (back-inferred, noisy)
WM gen (Wh0)     → video is synthetic but unlabeled → still use the perception stack (HaWoR) to "reconstruct" labels (back-inferred, noisy)
sim engine gen   → state is the label → read engine ground-truth directly (zero reconstruction, zero noise)
```

The first is real: real video in, represented by Do-as-I-Do (arXiv:2606.19333). It's a three-stage pipeline of reconstruction, retargeting, and deployment: the reconstruction stage (SAM3 segmentation, SAM-3D mesh, MoGe point maps, HaWoR hand reconstruction, GeoCalib gravity alignment, TAPIR, pose tracking) extracts hands as 3D motion from everyday human video; the retargeting stage uses sampling-based MPC on MuJoCo Warp to physically optimize and retarget the human hand trajectory onto the robot dexhand; finally deployment on a real robot (UR3e plus a Sharpa dexterous hand). In this line, HaWoR plays the role of the hand-perception front-end for real video.

The second is WM: generated video in, represented by Wh0 (arXiv:2606.22136), with a flow of generating world-model video, perception annotation, then feeding VITRA for training. The data source is swapped to a generative world model — Qwen-Image-Edit for scene editing, Qwen3-VL for prompt augmentation, Wan 2.2 I2V for video generation — treated as an infinite video source; but the generated video still has to be labeled with 3D hand motion by HaWoR before it can become supervision for a VLA policy (VITRA is PaliGemma2-3B plus a DiT action head). Here HaWoR plays the role of the annotation back-end for synthetic video. The video source switched from real to synthetic, but the video→3D-hand-trajectory step in the middle didn't change.

The third is sim, and its idea is entirely different: rather than back-inferring from pixels, let the physics engine give the state directly. Inside the engine, the hand pose, the object's 6-DoF, contact forces, and depth are all known ground-truth; the scene state itself is the 3D hand-trajectory ground-truth, so the HaWoR perception stack isn't needed at all — there isn't even perception error. This is the key watershed of the three pipelines: real and WM both go through perception back-inference and both depend on the perception stack of §4, while sim skips it entirely.

Laying out the three sources dimension by dimension:

| Dimension | 1. raw real video | 2. WM gen (Wh0) | 3. sim engine gen |
|---|---|---|---|
| Nature of data | internet / ego real footage | generative synthetic video (Qwen-Edit→Wan I2V) | physics-engine rendering (MuJoCo / Isaac) |
| Visual realism | ★★★ most real | ★★☆ photorealistic but with gen artifacts | ★☆☆ sim look, rendering gap |
| Distribution / diversity | ★★★ natural long tail, massive | ★★☆ prompt-extensible, bounded by base-model distribution | ★☆☆ bounded by asset library (objects / scenes) |
| Where 3D labels come from | HaWoR perception back-inference | HaWoR perception back-inference | engine outputs ground-truth directly ✅ |
| Label quality / automation | poor: occlusion, scale ambiguity, noise | medium: same back-inference, viewpoint slightly controllable | ★★★ perfect ground-truth, fully automatic, zero cost |
| Physical consistency | ★★★ inherently real physics | ★☆☆ not guaranteed (generation may not be physically feasible) | ★★★ engine enforces physical consistency |
| Controllability (make what you want) | ★☆☆ can only grab what exists | ★★☆ prompt / edit steerable | ★★★ task / object / camera / lighting all parameterizable |
| Cost / scalability | collection free but cleaning expensive | medium (inference compute) | heavy upfront (build scenario/assets/task/physics tuning), then very low marginal cost per task |
| Main weakness | label noise + copyright | physics not trustworthy + labels still back-inferred | sim-to-real visual gap + heavy upfront engineering of scenario/assets/task |
| Representative works | Do-as-I-Do, EgoMimic | Wh0, DreamGen, Cosmos | DexMimicGen, MimicGen, RoboCasa, DexMan(sim), DexGraspNet |

The sim column is the easiest to misread and worth spelling out. Its dividend is very real: skip the perception stack and get ground-truth directly — real and WM are "video → HaWoR (SLAM+MANO+depth) → back-inferred 3D hand trajectory (with error)," while sim is directly "scene state = 3D hand-trajectory ground-truth," fully automatic labels, physically consistent, without even perception error. But its cost is far more than the sim-to-real gap; the truly heavy part is the upfront infra. Building on-demand scenes like "tidy the living room" or "wash the dishes" inside an engine — involving assets, layout, task and reward design, contact-physics tuning — is heavy work in itself; the mainstream industry usage (DexMimicGen, MimicGen, RoboCasa, Isaac Lab) is actually to amplify a few demos into large amounts of data inside already-built scenes, not to author arbitrary tasks from scratch. In sum, sim has two costs: one is the visual sim-to-real gap, where a VLA's visual encoder easily overfits to the sim look; the other is the scene/asset/task construction engineering, which directly caps the diversity ceiling at the size of the asset library.

So which one should you pick? The three sources aren't peer candidates; different goals call for different sources.

```
        visual realism      physics / label truth      distribution diversity
real  →    ★★★               ★ (labels back-inferred)    ★★★
WM    →    ★★                ✗ (physics not guaranteed)  ★★
sim   →    ★                 ★★★ (ground-truth)          ★
```

If the goal is broad-coverage pretraining corpus, prioritizing scale and diversity, then base it on raw video. This is exactly the thesis of the human-video paradigm: the long-tail diversity of real video is natural, free, and unbounded, and open-world long-tail tasks like "tidy the living room" or "wash the dishes" are precisely what a sim asset library struggles most to cover and what raw video does best; pretraining inherently favors "large and noisy" over "small and clean," so the label noise from back-inference is acceptable. If the goal is a handful of skills to deploy, needing closed-loop or RL fine-tuning, prioritizing label truth and physics, then let sim (or the real robot outright) come in at pinpoint spots — here precise action ground-truth, contact physics, and closed-loop rollout are irreplaceable, but this belongs to the terminal stage of a narrow goal, not the body of the data engine; and one shouldn't author scenes from scratch, but use the two things sim does best: DexMimicGen-style amplification when a few demos already exist, and physical fine-tuning and evaluation of terminal tasks. As for WM, its place is to fill the distribution blind spots of raw video in a targeted way — a supplement, not the mainstay, and don't forget its physics is untrustworthy and its labels still need back-inference.

Put another way, sim's heavy upfront infra plus its diversity ceiling by themselves negate the idea of "using sim as a general data generator": broad coverage goes to raw video, and sim retreats to the narrow-precision stage that needs physics ground-truth. This is consistent with the two-track judgment of §2 — raw video feeds the scale track, sim feeds the narrow stage of physics ground-truth. Of course a natural hybrid exists: let sim produce grounded ground-truth actions, then use real or generated video to fill in visual realism — for example, neural-rendering sim trajectories into photorealistic video, or jointly training sim and real.

---

## §6 Extension: explicit 3D state as the state front-end of a physics-grounded world model

The three pipelines above all land on "3D hand-object state as label or demo." But the same 3D state can leverage a third use: as the state space of a world model.

Today's mainstream world model is a pixel WM that predicts the next frame — Wh0 and Cosmos are both of this kind — and its common flaw is generating physically impossible futures, because it only pursues "looks right." The number-one future direction named by both surveys is the physics-grounded world model: swapping the WM's state space from "next-frame pixels" to "the next-moment 3D hand-object state." DexWM and DWM have made a start, only neither has imposed strict physical constraints yet. And "per-frame 3D hand-object state" is precisely the native output of the explicit-3D pipeline — real and WM obtain it via HaWoR back-inference, sim gives it as ground-truth; in terms of WM state quality, sim's ground-truth beats back-inference.

This direction already has a more mature, scaled instance than DexWM or DWM: NVIDIA and Stanford's PointWorld (arXiv:2601.03782). It unifies scene state and robot action into a 3D point flow; given RGB-D observation and a sequence of actions, it predicts future per-point 3D displacement, infers in real time (0.1s), and plugs into MPC — achieving zero-shot object pushing, articulation, and tool use starting from a single in-the-wild image. It validates that "explicit 3D state + world model + planning" can indeed scale. But two boundaries should be drawn clearly: first, it trains on robot data (DROID teleoperation plus BEHAVIOR simulation) and does not go through the human-video bridge; second, its real-robot demos are mostly single-arm with a gripper, and its state is whole-scene dense point flow, not the fingertip-level 3D hand-object state this article cares about. So it's a parallel path outside this article's main thread — also an explicit-3D world model, but with a different data source and terminal.

In other words, the explicit-3D pipeline can not only feed VLA labels, but also serve directly as the state source of a physics-grounded WM. But one must be honest here: this line does not make "manufacturing 3D state" cheaper; it's still fundamentally the heavy pipeline of §5 — real and WM rely on HaWoR back-inference, noisy and scale-limited, while sim gives ground-truth directly but has a sim-to-real gap. This is exactly the old trade-off behind "training a WM with explicit 3D seems obvious yet few do it": strong grounding and expensive data are two sides of one coin, and HaWoR hasn't eliminated it. The real differentiator isn't that the pipeline is solved, but who already has a usable 3D-state pipeline in hand — with it, one can migrate from an explicit-3D VLA to a physics-grounded WM, without starting over from a pure-pixel WM.

---

## Summary

From the global to the concrete, this line can be reviewed as follows:

| Level | Question | Landing point |
|---|---|---|
| Global paradigm | Why use human video | data bottleneck + the human≠action gap (§1) |
| Landscape | How to cross the gap / what directions exist | representation-bridge spectrum + three axes (§2) |
| Converge to dexhand | Why emphasize 3D hand reconstruction | dexhand needs fingertip-level physical grounding (§3) |
| Perception primitive | Who does the video→3D step | HaWoR (implementation of the explicit-3D bridge, §4) |
| Data generator | How many pipelines to manufacture data | real (Do-as-I-Do) / WM (Wh0) / sim, three sources (§5) |
| Extension | What else 3D state can be | state front-end of a physics-grounded world model (§6) |

Threading these together, the paradigm is indeed expanding from "hands" to "whole body," and from "real video" to "generative world models and simulation"; but for the dexhand line, the step of turning a scene into robot-learnable 3D hand motion remains the core — real and synthetic video rely on perception back-inference, simulation relies on the engine's direct output. Looking ahead, the near-term certain dividend is in the already-mature deployment of explicit 3D, usable to manufacture humanoid-ready data and fine-tune small VLAs; the mid-term leverage is in episode-ization tooling and latent pretraining; and the long term can use the output of 3D state to support physics-grounded world models. One explicit-3D data pipeline can simultaneously feed VLA, manufacture synthetic data, and serve as the state source of a WM.

---

## References

Surveys:

- *From Human Videos to Robot Manipulation: A Survey on Scalable VLA with Human-Centric Data* — [arXiv:2606.00054](https://arxiv.org/abs/2606.00054) (VLA perspective, four bridges by representation type)
- *Robot Learning from Human Videos: A Survey* — [arXiv:2604.27621](https://arxiv.org/abs/2604.27621) (general LfHV perspective, three layers by information flow; companion list [IRMVLab/awesome-robot-learning-from-human-videos](https://github.com/IRMVLab/awesome-robot-learning-from-human-videos))

Main cited works:

- HaWoR — *World-Space Hand Motion Reconstruction from Egocentric Videos* (CVPR'25)
- Do-as-I-Do — [arXiv:2606.19333](https://arxiv.org/abs/2606.19333)
- Wh0 — [arXiv:2606.22136](https://arxiv.org/abs/2606.22136)
- VITRA — [arXiv:2510.21571](https://arxiv.org/abs/2510.21571) · [microsoft/VITRA](https://github.com/microsoft/VITRA)
- LUCID — [arXiv:2606.11628](https://arxiv.org/abs/2606.11628)
- DexWM — [arXiv:2512.13644](https://arxiv.org/abs/2512.13644) · [facebookresearch/dexwm](https://github.com/facebookresearch/dexwm)
- PointWorld — [arXiv:2601.03782](https://arxiv.org/abs/2601.03782) · [NVlabs/PointWorld](https://github.com/NVlabs/PointWorld) (3D point-flow world model, trained on robot data, non-human-video)
