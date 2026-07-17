# Rendering a 3DGS Scene and a Robot in One Pass on AMD: A Pure-Vulkan Nyx Alternative

> English | [中文](readme-cn.md)

Generative world models (like World Labs' Marble) can turn a single prompt into a photorealistic, simulation-ready 3D scene, compressing "build a robot sim environment" from weeks to hours. NVIDIA recently walked this pipeline end to end: Marble generates a 3DGS kitchen → NuRec converts the format → import into Isaac Sim, add physics → drop a robot in and run — a mature workflow built around the CUDA ecosystem.

This article fills in the other half: the feasibility of the same pipeline **on different hardware**. If all you have is an AMD card, can you take the **same** Marble kitchen and turn it into a world a robot physically walks in? Top-down: why move generative 3DGS scenes into simulation (§1) → the two layers of this pipeline, and why AMD needs a different rendering path (§2) → the core rendering-layer choice: unified rendering, not 2D compositing (§3) → wiring the renderer back into the Genesis physics loop (§4) → Go2 actually walking the kitchen under an RL policy (§5) → a four-step hands-on reproduction from the open-source repo (§6) → an honest comparison with NVIDIA's pipeline, and what's still missing (§7).

---

## §1 The paradigm: why move generative 3DGS scenes into robot simulation

Robot simulation has long been bottlenecked on **building the scene**. Hand-modeling a decent kitchen or living room takes weeks, and the result often looks plasticky, far from the real world. Generative world models change this: from a prompt or an image, they emit a photorealistic 3D scene with geometry, exported as a **3D Gaussian Splatting (3DGS)** point cloud (`.ply`). The paradigm shift, in essence, is replacing "humans build the sim scene" with "a world model generates the sim scene."

3DGS happens to be the ideal carrier for this. It represents a scene with millions of semi-transparent Gaussian "points" — **very high visual fidelity, and fast to render** — and it is **pre-lit**: each Gaussian already bakes lighting into its color, so it looks realistic in any renderer without relighting. Tools like World Labs Marble export exactly this `.ply` Gaussian cloud, plus a collider mesh (`.glb`) for physics.

But a scene alone isn't enough. What robot simulation actually needs is a loop: **scene (splat) + robot (mesh) + physics**, all in one cycle, where the robot is lit by the scene, occluded by its geometry, and constrained by physics.

And what hosts this loop is the **simulator** itself — here NVIDIA and AMD face a key difference. NVIDIA has its first-party **Isaac Sim** ecosystem: physics, rendering, sensors, and robot assets are all officially integrated and deeply tied to CUDA. AMD embraces open source, choosing an **open, vendor-agnostic** physics simulator — and **Genesis** is exactly such a recently popular open-source simulator. In other words, for AMD users, Genesis is nearly the mandatory foundation for moving generative 3DGS scenes into robot sim; which makes **whether Genesis's 3DGS rendering runs on AMD a question that matters a great deal to AMD users**.

---

## §2 Landscape: the two-layer pipeline, and the reality on AMD

Break "generative 3DGS scene → robot simulation" apart, and it's really **two orthogonal layers**:

```
Render layer:      how to render a 3DGS scene + robot mesh into one unified-lit image?
Integration layer: how to bind that renderer into the physics engine's camera / scene / control loop?
```

NVIDIA's pipeline solves both: **NuRec / 3DGRUT** (based on the 3DGUT algorithm) handles the render layer, converting `.ply` to USDZ and rendering in Omniverse; **Isaac Sim** handles the integration layer — aligning coordinates, adding physical lighting, placing the robot. Clean and mature, and oriented around NVIDIA + CUDA: the format conversion needs CUDA, and rendering runs on Omniverse's CUDA stack.

Back to the open-source foundation from §1, Genesis: its 3DGS rendering path is the **Nyx** renderer (a camera-sensor form). Like most cutting-edge rendering stacks, it is currently **NVIDIA-first** — on AMD, `gs-nyx-plugin`'s `scene.build()` hard-depends on `libcuda.so`/`cuInit`, and the core is closed-source, so it can't run on AMD for now (we've filed the request upstream at [genesis-nyx #18](https://github.com/Genesis-Embodied-AI/genesis-nyx/issues/18); community AMD support is only a matter of time). This isn't about who did better — new hardware ecosystems just need time to catch up.

Worth noting: the obstacle is neither the hardware nor the engine. AMD RDNA4 + Mesa RADV already fully support Vulkan hardware ray tracing (`vulkaninfo` enumerates `RADV GFX1201` with `rayTracingPipeline`/`accelerationStructure`), and the 3DGS rendering core is Vulkan, not CUDA-only. So for our situation — **only AMD cards on hand, and wanting the pipeline running now** — the most direct route is to build a pure-Vulkan rendering path ourselves. Not to replace anyone, but to give AMD users a working, hackable option in the meantime.

---

## §3 Render layer: why it must be "unified rendering," not 2D compositing

Building the render layer, the first fork is: **should the splat and the robot mesh be drawn in the same rendering pass?**

Two roads. One is **2D compositing**: render the splat to one image, the mesh to another, then stack the two 2D images by depth. The cross-repo project MuGS (MuJoCo + `amd_gsplat`) has shown this works on AMD (MI300X/gfx942) with zero changes — **but it can only match occlusion**: front/back ordering lines up, yet the two pixel sets come from two independent lighting worlds. The robot isn't lit by the scene, shares no shadows, and looks fake at a glance.

The other is **unified rendering**: put mesh and Gaussians into the **same render pass**, sharing one z-buffer and one lighting setup. Only this achieves **Isaac Sim-grade unified lighting** — the key to putting the robot *inside* the scene rather than *pasted in front of it*. If realism is the goal, the render layer must go unified.

For the implementation, we chose **`vk_gaussian_splatting` (vk_gs)** — a **native Vulkan** Gaussian-splatting renderer from NVIDIA's nvpro-samples (note: written in Vulkan ≠ requires NVIDIA hardware). Forked to the `rdna4_support` branch and compiled to run on AMD RADV/RDNA4. Key points:

- **Pure Vulkan, zero CUDA**: runs on RDNA4 via Mesa RADV; the only hard bar is **Vulkan SDK ≥ 1.4.341** (the RADV driver has long been at 1.4.318 — only the SDK header layer was missing) plus `-DUSE_DLSS=OFF` (to strip NVIDIA NGX/DLSS dependencies).
- **All three pipelines work**: rasterization, hardware ray tracing, and hybrid all run; 15.3M splats + OBJ mesh draw in the **same depth, unified rasterization** — unified rendering established.
- **Reads `.ply` directly, no conversion**: unlike NVIDIA needing 3DGRUT to convert PLY→USDZ (a CUDA step), vk_gs **eats `.ply` directly**, skipping the whole conversion stage.

> In one line: 2D compositing aligns occlusion but not lighting; unified lighting requires the same-pass unified render — and vk_gs gives AMD a pure-Vulkan, direct-PLY unified rendering foundation.

---

## §4 Integration layer: wiring the renderer back into the Genesis physics loop

Rendering an image is only half of it. To make the robot *live* in the scene, vk_gs must be wired back into the Genesis physics loop — the integration layer, and the part that had to be built from scratch once Nyx was unavailable.

Fortunately Genesis's architecture offers a clean extension point: renderers are all made into **camera sensors** (`scene.add_sensor(...)`), and an external pip package can register a new sensor type with zero kernel changes (`gs-nyx-plugin` is the proof). So the integration layer lands as a standalone plugin, **[`vk-gsplat-plugin`](https://github.com/ZJLi2013/vk-gsplat-plugin)** (Apache-2.0, open source) — rendering a robot **into** a 3DGS scene, in-loop with physics, as a first-class Genesis camera sensor (an open-source, AMD-friendly counterpart to the closed-source Nyx).

Its design principle is **"robot/scene knowledge is data, not code"**: the sensor itself is generic; everything robot- and scene-specific is passed as data through `GsplatCameraOptions`:

- `mesh_specs` — which visual meshes attach to which link, and their colors;
- `splat_r / splat_t / splat_scale` — the asset similarity transform `p_splat = scale·R·p_genesis + T`, i.e. the Genesis-world ↔ splat-coordinate calibration.

That calibration is not trivial. Marble's exported kitchen uses the OpenCV convention (+x left, +y down, +z forward), while Genesis is Z-up; alignment must register rotation, scale, and translation in one shot (our result: `R` = Z-up→Y-up, `s ≈ 1`, `t` = floor center, `up = Y`). Get it wrong and the robot floats in the air or sinks into the floor — exactly corresponding to NVIDIA's "manually align the Gaussian volume to the ground and tune scale against a 1-meter reference cube," only here frozen into reusable calibration parameters.

At run time, vk_gs's pybind renderer exposes `add_mesh` / `set_mesh_transform` / `set_mesh_color` / per-frame TLAS refresh / RGBA8 readback to numpy. Every `scene.step()`, Genesis computes each robot link's pose (forward kinematics, FK), bridges the coordinates, feeds vk_gs, and renders a frame — so the robot "walks frame by frame" in the kitchen. The pose side can run on CPU, using zero GPU.

> In one line: the integration layer is "adding a thin control entry," not rewriting the renderer — hook per-frame poses, coordinate calibration, and readback into a Genesis camera sensor, and the robot is in the scene.

---

## §5 The payoff: Go2 walks into the Marble kitchen

With both layers together, it comes down to one demo that makes the point: **a Unitree Go2 physically walking in a Marble-generated kitchen** (`examples/go2_kitchen/`). It runs on an AMD R9700 (RDNA4/gfx1201) node, in the very **same free asset** from NVIDIA's tutorial — World Labs Marble's "Rustic kitchen with natural light" (a 2M-Gaussian `.ply`).

<video src="/2026/vkgs_scene_on_amd/media/walk_fixedcam_long.mp4" controls width="100%" muted loop></video>

*Go2 walking in the Marble-generated 3DGS kitchen: robot mesh and Gaussian splats rendered in one pass, obeying scene geometry.*

Three things happen at once in this demo, covering each layer above:

- **Rendering**: Go2's multi-material `.glb` and the kitchen's Gaussian splats are drawn in **one render pass** (§3's unified rendering), with a follow camera.
- **Physics**: collision geometry is aligned from Marble's collider `.glb`, and Go2 is driven by an RL-trained locomotion policy (4096 parallel training envs), **walking on the floor, not falling, blocked by cabinets** — real physics, not a keyboard-driven animation.
- **Integration**: all of it happens frame by frame inside Genesis's `scene.step()` loop (§4's camera sensor).

This completes NVIDIA's last step — "drive the robot, verify it respects the kitchen geometry" — and goes a step further: real physics driven by an RL policy. One note on collision geometry: **3DGS is always just a render asset** (true for Nyx / NuRec / vk_gs alike; NVIDIA's tutorial also imports a separate collider `.glb` for physics). An interactive scene must carry separate collision geometry — luckily Marble's free sample ships the PLY and the collider GLB from the same source and coordinate frame, ready to use.

> In one line: the same free kitchen, on AMD, lets a quadruped physically walk under an RL policy — rendering, physics, and integration all working in one demo.

---

## §6 Hands-on reproduction: four steps to put a robot in the kitchen on AMD

The above is all "why"; this section is "how" — following the open-source repo [`vk-gsplat-plugin`](https://github.com/ZJLi2013/vk-gsplat-plugin). To read alongside NVIDIA's tutorial, I split it into the same four steps.

**Step 1: Get the scene (exactly the same asset as NVIDIA's tutorial).** From World Labs Marble's free sample gallery, download "Rustic kitchen with natural light" — the Gaussian `.ply` (for rendering) and the collider `.glb` (for physics), both from the same source and coordinate frame. Place it at `assets/rustic_kitchen_2m.ply`. This is the same thing as NVIDIA's "Step 1: get the kitchen from Marble."

**Step 2: Prepare the render backend (corresponds to NVIDIA's "PLY→USDZ conversion," but conversion-free here).** NVIDIA's step uses 3DGRUT to convert PLY to USDZ (needing CUDA); on AMD we don't convert formats, we build a Docker image that compiles the Vulkan renderer `vkgs` and installs the plugin:

```bash
docker build -t vk-gsplat:latest -f docker/Dockerfile .   # builds vkgs + installs the plugin
```

This has been **reproduced end to end** from a clean clone on an AMD R9700 node (`rocm7.2.3_ubuntu24.04_py3.12`): the image compiles `vkgs` from source and the sanity check passes (`vkgs native OK: ['Renderer']`, `vk_gsplat_plugin installed OK: 0.1.0`). Every dependency (Vulkan SDK, GLFW, pybind11, `-fPIC`, ...) is pinned in `docker/Dockerfile` + `scripts/build_vkgs.sh`, with no manual steps.

**Step 3: Place the robot + calibrate (corresponds to NVIDIA's "import into Isaac Sim, align, add physical lighting").** All robot and scene info is written as **data** into `GsplatCameraOptions` — `mesh_specs` says which visual meshes attach to which link and their colors; `splat_r/splat_t/splat_scale` give the Genesis↔splat similarity calibration. This corresponds exactly to NVIDIA's "manually align the Gaussian volume to the ground and tune scale," only frozen into parameters:

```python
import genesis as gs
import vk_gsplat_plugin                       # registers the sensor
from vk_gsplat_plugin import GsplatCameraOptions

gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

cam = scene.add_sensor(GsplatCameraOptions(
    res=(1280, 720), ply="kitchen.ply", robot_entity_idx=1,
    mesh_specs=(("link0.obj", "link0", (0.9, 0.9, 0.9)), ...),
    splat_r=(1,0,0, 0,0,1, 0,-1,0), splat_t=(0,-1.1,0.92), splat_scale=1.0,  # kitchen calibration
    cam_eye=(0,-0.125,-0.28), cam_center=(0,-0.425,1.82), cam_up=(0,1,0), cam_fovy=65.0,
))
scene.build()
scene.step()
rgb = cam.read().rgb            # (H, W, 3) uint8 torch tensor
```

**Step 4: Run it (corresponds to NVIDIA's "add robot, drive, verify").** On a headless node, run the ready-made example in the container with `xvfb-run` and get frames:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --security-opt seccomp=unconfined \
  -v $(pwd):/work -w /work vk-gsplat:latest \
  xvfb-run -a python examples/franka_kitchen/s1_sensor_demo.py \
    --ply /work/assets/rustic_kitchen_2m.ply --assets /work/assets/franka \
    --gpu 2 --res 1280 720 --out-dir out/f6_sensor
```

For quadruped physical walking, switch to the self-contained `examples/go2_kitchen/` example (it ships its own follow camera + multi-material `.glb` support).

Two honest caveats: first, **assets are not shipped with the repo** — Franka's visual `.obj` comes from `genesis-world`'s bundled assets, the kitchen `.ply` from Marble, both placed into `assets/` manually; second, **the current reproduction boundary** — the image build is reproduced end to end, but running the Franka example in-container still needs the assets pre-placed on the node (flagged as a TODO in the repo Status). Environment hard requirements: `genesis-world==1.2.1` (the plugin uses internal modules, version-sensitive), CPython 3.12, AMD GPU + Mesa RADV ≥ 24.3.

---

## §7 An honest comparison: with NVIDIA's pipeline, and what's still missing

Laying out both pipelines stage by stage:

| Stage | NVIDIA (Isaac Sim + NuRec) | This work (Genesis + vk_gs on AMD) |
|---|---|---|
| GPU / driver | NVIDIA + CUDA 11.8+ | AMD RDNA4 + Mesa RADV, **zero CUDA** |
| Scene source | World Labs Marble 3DGS | the **same** Marble kitchen |
| PLY handling | 3DGRUT to USDZ (needs CUDA) | vk_gs **reads `.ply` directly**, no conversion |
| Rendering | NuRec / 3DGUT (Omniverse) | vk_gs three pipelines (raster/RT/hybrid) |
| Unified lighting | ✅ mature | ✅ established via same-pass render |
| Physics engine | Isaac Sim | Genesis |
| Integration form | Isaac Sim native | Genesis camera-sensor plugin (Nyx counterpart) |
| Open / hackable | closed stack | ✅ fully open, self-modifiable |
| Maturity | product-grade | PoC, render layer working |

No hype — two **current weaknesses** to be clear about:

First, **injected robot meshes are currently "flat-lit"** — grayish, lacking depth, like paper cutouts. The root cause is that the vk_gs renderer defaults to `LIGHTING_DISABLED`, so mesh color equals baseColor directly — no N·L, no light source, no ambient. The good news: the full PBR + directional/point lights + physical sky + HDRI/IBL + shadow infrastructure is **actually all in the renderer**, just off by default, and the headless pybind hasn't yet exposed these switches; the splat itself is pre-lit and unaffected — only the injected mesh lacks lighting. This is **engineering to fill in**, not an architectural flaw.

Second, **it's still a PoC overall**: both the render layer (unified rendering) and the integration layer (online sensor + physical walking) work, but it's some distance from NVIDIA's product-grade "describe a world and test it the same day" — DLPack zero-copy (Vulkan→HIP), dual-camera grasp episodes, automating scene collision geometry, and more are still planned.

The judgment I want to leave: **there is no fundamental obstacle to this pipeline on AMD; what was missing is just a rendering path that doesn't touch CUDA and that you can modify yourself.** vk_gs + Genesis walked that path — the same generative kitchen NVIDIA moved into Isaac Sim with a full CUDA stack, we moved into Genesis with pure Vulkan. As "generative world models feeding robot simulation" heats up, freeing this pipeline from a single vendor's CUDA ecosystem is itself worth doing.

---

## Closing thoughts

Put plainly, the thing is simple: have AI generate a photorealistic kitchen, drop a robot in, and let it walk, be blocked by a table, be lit by the lamps — as if it were really in that room. NVIDIA already made this path smooth with its own full toolchain; that toolchain is just built for NVIDIA cards.

What we wanted to know: on an AMD card, same kitchen, same task, can it still be done? Yes. The key step is to not "paste" the robot in front of the scene, but draw it and the scene in the same rendering pass — so the lighting lines up and it doesn't look fake. The rest is wiring that renderer back into a physics engine so the robot can actually walk, collide, and stand. In the end Go2, with the walking policy it learned itself, took a stroll through this AI-generated kitchen — blocked by cabinets, staying on the floor, without falling.

There are of course limitations: the injected robot looks a bit "gray" and flat, and the whole thing is still a working prototype, far from "describe a world and test it the same day." But the point is that there is no insurmountable barrier on AMD — what was missing is simply someone to blaze the path first, and open-source it, so others can build on it.

---

## References

- [vk-gsplat-plugin](https://github.com/ZJLi2013/vk-gsplat-plugin)
- [vk_gaussian_splatting](https://github.com/ZJLi2013/vk_gaussian_splatting) (render backend, nvpro-samples fork, `rdna4_support`)
- [Genesis](https://github.com/Genesis-Embodied-AI/genesis) · [genesis #1358](https://github.com/Genesis-Embodied-AI/genesis-world/issues/1358) · [genesis-nyx #18](https://github.com/Genesis-Embodied-AI/genesis-nyx/issues/18)
- [NVIDIA Isaac Sim × World Labs Marble tutorial](https://developer.nvidia.com/blog/simulate-robotic-environments-faster-with-nvidia-isaac-sim-and-world-labs-marble/)
- [World Labs Marble export specs](https://docs.worldlabs.ai/marble/export/specs) (kitchen sample asset downloads)
- [MuGS](https://github.com/Renforce-Dynamics/MuGS) · [ROCm/gsplat (amd_gsplat)](https://github.com/ROCm/gsplat)
