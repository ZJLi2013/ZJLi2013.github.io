# AMD 上把 3DGS 场景和机器人渲进同一个渲染 pass：一个纯 Vulkan 的 Nyx 替代

> [English](readme-en.md) | 中文

生成式世界模型（如 World Labs 的 Marble）能从一句话生成照片级、可仿真的 3D 场景，正把"搭一个机器人仿真环境"从几周压缩到几小时。NVIDIA 最近的教程把这条链路完整走通了：Marble 生成 3DGS 厨房 → NuRec 转格式 → 导进 Isaac Sim 加物理 → 放机器人进去跑——一套围绕 CUDA 生态打磨得很成熟的方案。

这篇文章想补的，是这条链路在**另一种硬件上**的可行性：手里只有 AMD 卡时，能不能也拿**同一个** Marble 厨房，把它变成机器人能物理行走的仿真世界。从上往下：为什么要把生成式 3DGS 场景搬进仿真（§1）→ 这条链路拆成哪两层、AMD 上为什么要走另一条渲染路径（§2）→ 渲染层的核心抉择：统一渲染而非 2D 合成（§3）→ 怎么把渲染器接回 Genesis 物理循环（§4）→ Go2 真的走进厨房、带 RL 策略物理行走（§5）→ 跟着开源仓库四步动手复现（§6）→ 与 NVIDIA 那条链路诚实对比、以及现在还差什么（§7）。

---

## §1 范式：为什么要把生成式 3DGS 场景搬进机器人仿真

机器人仿真长期卡在**造场景**这一步。手工建模一个像样的厨房、客厅要几周，做出来还往往"塑料味"十足，和真实世界差距大。生成式世界模型改变了这件事：给一句话或一张图，它直接吐出一个照片级、带几何的 3D 场景，导出成 **3D 高斯泼溅（3D Gaussian Splatting，3DGS）** 的点云文件（`.ply`）。所谓范式转移，本质就是把"人肉建仿真场景"换成"世界模型生成仿真场景"。

3DGS 恰好是这套流程的理想载体。它用几百万个半透明的高斯"点"表示场景，**视觉真实度极高、渲染又快**；而且它是 **pre-lit** 的——每个高斯的颜色已经把光照烘焙进去了，导进任何渲染器都自带真实感，不用重打光。World Labs Marble 这类工具导出的正是 `.ply` 高斯点云，配一份碰撞网格（collider mesh，`.glb`）就能拿去做物理。

但只有场景还不够。机器人仿真真正要的是一个闭环：**场景（splat）+ 机器人（mesh）+ 物理**，三者同处一个循环，机器人能在场景里被光照亮、被几何挡住、被物理约束。

而承载这个闭环的，是**仿真器**本身——这里 NVIDIA 和 AMD 的处境有一处关键差异。NVIDIA 有自家第一方的 **Isaac Sim** 生态：物理、渲染、传感器、机器人资产一整套都是官方打通、深度绑定 CUDA 的。AMD 拥抱开源，选择**开源、不挑硬件厂商**的物理仿真器——而 **Genesis** 正是这样一个近来很受关注的开源仿真器。换句话说，对 AMD 用户，Genesis 几乎是把生成式 3DGS 场景搬进机器人仿真的必经底座；也正因如此，**Genesis 的 3DGS 渲染能不能在 AMD 上跑通，就成了一件对 AMD 用户格外要紧的事**。

---

## §2 Landscape：两层链路，与 AMD 上的现实约束

把"生成式 3DGS 场景 → 机器人仿真"这条链路拆开，本质是**两个正交的层**：

```
渲染层：怎么把 3DGS 场景 + 机器人 mesh 渲成一张统一光照的图？
集成层：怎么把这个渲染器绑进物理引擎的相机 / 场景 / 操作闭环？
```

NVIDIA 那条链路其实同时解决了这两层：**NuRec / 3DGRUT**（基于 3DGUT 算法）负责渲染层，把 `.ply` 转成 USDZ、在 Omniverse 里渲染；**Isaac Sim** 负责集成层，对齐坐标、加物理光照、放机器人。干净、成熟，主要面向 NVIDIA + CUDA 生态：格式转换要 CUDA，渲染走 Omniverse 的 CUDA 栈。

回到 §1 说的那个开源底座 Genesis，它对应的 3DGS 渲染路径是 **Nyx** 渲染器（camera sensor 形态）。和大多数前沿渲染栈一样，它目前也是 **NVIDIA 优先**——在 AMD 上，`gs-nyx-plugin` 的 `scene.build()` 阶段会硬依赖 `libcuda.so`/`cuInit`，而核心是闭源发行的，暂时没法在 AMD 上跑起来（我们已把这个需求提到上游 [genesis-nyx #18](https://github.com/Genesis-Embodied-AI/genesis-nyx/issues/18)，社区支持 AMD 只是时间问题）。这不是谁做得好不好的问题，只是新硬件生态需要时间补齐。

值得一提的是，障碍并不在硬件或引擎本身：AMD RDNA4 + Mesa RADV 已完整支持 Vulkan 硬件光追（`vulkaninfo` 能枚举到 `RADV GFX1201` 的 `rayTracingPipeline`/`accelerationStructure`），而 3DGS 渲染核心本就是 Vulkan 而非 CUDA-only。所以对我们这种**手上只有 AMD 卡、又想现在就把链路跑起来**的场景，最直接的办法就是自建一条纯 Vulkan 的渲染路径——不是要替代谁，而是给 AMD 用户先补上一个能用、可改的选项。

---

## §3 渲染层：为什么必须"统一渲染"，而不是 2D 合成

自建渲染层，第一个岔路口是：**splat 和机器人 mesh，要不要在同一遍渲染里画出来？**

有两条路。一条是 **2D 合成（composite）**：splat 渲一张图、mesh 渲一张图，再按深度（depth）把两张 2D 图叠起来。跨仓库的 MuGS（MuJoCo + `amd_gsplat`）已经证明这条路在 AMD（MI300X/gfx942）上零改动能跑通——**但它只能追平"遮挡"**：谁在前谁在后对得上，可两套像素来自两个独立的光照世界，机器人不会被场景的光照亮、不共享阴影，拼出来一眼假。

另一条是 **统一渲染（unified rendering）**：让 mesh 和 gaussian 进**同一个 render pass**，共享同一个 z-buffer 和同一套光照。只有这样才能做到 **Isaac Sim 级的统一光照（unified lighting）**——这正是把机器人"真正放进"场景、而不是"贴在场景前面"的关键。目标要真实感，渲染层就只能走统一渲染。

落到实现，我们选的底座是 **`vk_gaussian_splatting`（vk_gs）**——NVIDIA nvpro-samples 出的一个**原生 Vulkan** 高斯泼溅渲染器（注意：是 Vulkan 写的，不等于要 NVIDIA 卡）。把它 fork 到 `rdna4_support` 分支，在 AMD RADV/RDNA4 上编译跑通，关键几点：

- **纯 Vulkan、零 CUDA**：经 Mesa RADV 在 RDNA4 上跑，硬门槛只是 **Vulkan SDK ≥ 1.4.341**（RADV 驱动早已 1.4.318，缺的只是 SDK 头文件层）+ `-DUSE_DLSS=OFF`（剔掉 NVIDIA NGX/DLSS 那几处依赖）。
- **三条管线齐活**：光栅（raster）、硬件光追（ray tracing）、混合（hybrid）三条渲染管线都跑通了，15.3M splat + OBJ mesh 能在**同一深度、统一光栅**里画出来——统一渲染由此成立。
- **直接读 `.ply`，不转格式**：对比 NVIDIA 要用 3DGRUT 把 PLY 转成 USDZ（且这一步要 CUDA），vk_gs **直接吃 `.ply`**，省掉整个转换环节。

> 一句话：2D 合成只能对齐遮挡、对不齐光照；要统一光照就必须同 pass 统一渲染，而 vk_gs 给了 AMD 一个纯 Vulkan、直读 PLY 的统一渲染底座。

---

## §4 集成层：把渲染器接回 Genesis 物理循环

渲染层出图只是一半。要让机器人"活在"场景里，还得把 vk_gs 接回 Genesis 的物理循环——这就是集成层，也是 Nyx 被堵死后**必须自建、无处可抄**的一环。

好在 Genesis 的架构给了干净的扩展点：渲染器统一做成 **camera sensor**（`scene.add_sensor(...)`），而且外部 pip 包可以零改内核地注册新 sensor 类型（`gs-nyx-plugin` 本身就是实证）。于是集成层就落成一个独立插件 **[`vk-gsplat-plugin`](https://github.com/ZJLi2013/vk-gsplat-plugin)**（Apache-2.0，已开源）——把机器人**渲染进** 3DGS 场景、与物理同一个循环，作为 Genesis 的第一类相机 sensor（对标闭源的 Nyx，做一个开源、AMD 友好的版本）。

它的设计原则是**"机器人/场景信息是数据，不是代码"**：sensor 本体是通用的，所有和具体机器人、具体场景相关的东西都以数据形式传进 `GsplatCameraOptions`：

- `mesh_specs` —— 哪些 visual mesh 挂到机器人的哪个 link，各自什么颜色；
- `splat_r / splat_t / splat_scale` —— 资产的相似变换 `p_splat = scale·R·p_genesis + T`，也就是 Genesis 世界坐标 ↔ splat 坐标的标定。

这套标定不是小事。World Labs 导出的厨房用的是 OpenCV 坐标系（+x left, +y down, +z forward），Genesis 是 Z-up；对齐要把旋转、尺度、平移一次配准（我们的结论：`R` = Z-up→Y-up、`s ≈ 1`、`t` = 地板中心、`up = Y`）。这一步做歪了，机器人要么飘在天上、要么陷进地板——和 NVIDIA 教程里"手动对齐 Gaussian volume 到地面、按 1 米参考立方体调 scale"是完全对应的一道工序，只是我们把它固化成了可复用的标定参数。

运行时，vk_gs 的 pybind 渲染器暴露 `add_mesh` / `set_mesh_transform` / `set_mesh_color` / 逐帧 TLAS 刷新 / RGBA8 回读 numpy——每个 `scene.step()`，Genesis 侧算出机器人各 link 的位姿（正向运动学 FK），桥接坐标后喂给 vk_gs，渲出一帧，机器人就"逐帧走"在了厨房里。pose 侧可跑在 CPU、零占 GPU。

> 一句话：集成层的本质是"加一层薄控制入口"而非重写渲染器——把逐帧位姿、坐标标定、回读接成 Genesis 的一个 camera sensor，机器人就进了场景。

---

## §5 落地：Go2 走进 Marble 厨房

前面两层拼起来，落到一个能说明问题的 demo：**Unitree Go2 在 Marble 生成的厨房里物理行走**（`examples/go2_kitchen/`）。跑在 AMD R9700（RDNA4/gfx1201）节点上，场景正是 NVIDIA 教程里**同一个免费资产**——World Labs Marble 的「Rustic kitchen with natural light」（2M 高斯的 `.ply`）。

<video src="media/walk_fixedcam_long.mp4" controls width="100%" muted loop></video>

*Go2 在 Marble 生成的 3DGS 厨房里行走：机器人 mesh 与高斯 splat 同一次渲染，遵守场景几何。*

这个 demo 里同时发生了三件事，正好覆盖前面每一层：

- **渲染**：Go2 的多材质 `.glb` 和厨房的高斯 splat 在**同一次渲染**里画出来（§3 的统一渲染），配一台跟拍相机。
- **物理**：从 Marble 的 collider `.glb` 对齐出**碰撞几何**，Go2 由一个 RL 训出来的 locomotion 策略（4096 环境并行训练）驱动，在厨房里**贴地行走、不摔、被橱柜挡住**——是真物理，不是键盘遥控的动画。
- **集成**：这一切都在 Genesis 的 `scene.step()` 循环里逐帧发生（§4 的 camera sensor）。

这正好补上 NVIDIA 教程最后"驱动机器人、验证它尊重厨房几何"那一步，而且更进一步——带 RL 策略的真物理行走。碰撞几何值得单说一句：**3DGS 永远只是渲染资产**（Nyx / NuRec / vk_gs 皆然，NVIDIA 教程里也靠额外导入 collider `.glb` 做物理）；可交互场景必须另配碰撞几何，好在 Marble 免费 sample 里 PLY 和 collider GLB 同源同坐标，直接能用。

> 一句话：同一个免费厨房，AMD 上让四足带着 RL 策略在里面物理行走——渲染、物理、集成三层在一个 demo 里全跑通了。

---

## §6 动手复现：四步在 AMD 上把机器人放进厨房

前面讲的都是"为什么"，这一节给"怎么做"——跟着开源仓库 [`vk-gsplat-plugin`](https://github.com/ZJLi2013/vk-gsplat-plugin) 走一遍。为方便和 NVIDIA 那篇教程对读，我把它也拆成同样的四步。

**第 1 步：拿场景（和 NVIDIA 教程完全同一个资产）。** 从 World Labs Marble 的免费 sample gallery 下载「Rustic kitchen with natural light」——高斯点云 `.ply`（渲染用）和碰撞网格 collider `.glb`（物理用），两者同源同坐标。放到 `assets/rustic_kitchen_2m.ply`。这一步与 NVIDIA 教程的「Step 1: 从 Marble 拿厨房」是同一件事。

**第 2 步：备好渲染后端（对应 NVIDIA 的「PLY→USDZ 转换」，但这里免转换）。** NVIDIA 那步要用 3DGRUT 把 PLY 转成 USDZ（且需 CUDA）；AMD 这边不转格式，而是构建一个把 Vulkan 渲染器 `vkgs` 编好、插件装好的 Docker 镜像：

```bash
docker build -t vk-gsplat:latest -f docker/Dockerfile .   # 编 vkgs + 装插件
```

这一步在 AMD R9700 节点（`rocm7.2.3_ubuntu24.04_py3.12`）上已从干净 clone **端到端复现**：镜像内从源码编出 `vkgs`，sanity 通过（`vkgs native OK: ['Renderer']`、`vk_gsplat_plugin installed OK: 0.1.0`）。所有依赖（Vulkan SDK、GLFW、pybind11、`-fPIC` 等）都钉死在 `docker/Dockerfile` + `scripts/build_vkgs.sh` 里，无手工步骤。

**第 3 步：放机器人 + 标定（对应 NVIDIA 的「导入 Isaac Sim、对齐、加物理光照」）。** 机器人和场景的所有信息都以**数据**形式写进 `GsplatCameraOptions`——`mesh_specs` 指定哪些 visual mesh 挂到哪个 link、什么颜色，`splat_r/splat_t/splat_scale` 给出 Genesis↔splat 的相似变换标定。这正对应 NVIDIA 教程里"手动把 Gaussian volume 对齐到地面、调 scale"那道工序，只是固化成了参数：

```python
import genesis as gs
import vk_gsplat_plugin                       # import 即注册 sensor
from vk_gsplat_plugin import GsplatCameraOptions

gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

cam = scene.add_sensor(GsplatCameraOptions(
    res=(1280, 720), ply="kitchen.ply", robot_entity_idx=1,
    mesh_specs=(("link0.obj", "link0", (0.9, 0.9, 0.9)), ...),
    splat_r=(1,0,0, 0,0,1, 0,-1,0), splat_t=(0,-1.1,0.92), splat_scale=1.0,  # 厨房标定
    cam_eye=(0,-0.125,-0.28), cam_center=(0,-0.425,1.82), cam_up=(0,1,0), cam_fovy=65.0,
))
scene.build()
scene.step()
rgb = cam.read().rgb            # (H, W, 3) uint8 torch tensor
```

**第 4 步：跑起来（对应 NVIDIA 的「加机器人、驱动、验证」）。** 无头节点上用 `xvfb-run` 起容器跑现成示例，出帧：

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --security-opt seccomp=unconfined \
  -v $(pwd):/work -w /work vk-gsplat:latest \
  xvfb-run -a python examples/franka_kitchen/s1_sensor_demo.py \
    --ply /work/assets/rustic_kitchen_2m.ply --assets /work/assets/franka \
    --gpu 2 --res 1280 720 --out-dir out/f6_sensor
```

想看四足物理行走，换 `examples/go2_kitchen/` 那个自包含示例即可（自带跟拍相机 + 多材质 `.glb`）。

需要诚实标注的两点：一是**资产不随仓库分发**——Franka 的 visual `.obj` 来自 `genesis-world` 自带资产、厨房 `.ply` 来自 Marble，都要手动放到 `assets/`；二是**当前复现边界**——镜像构建已端到端跑通，容器内跑 Franka 示例还需把资产提前放到节点上（仓库 Status 里标着这条 TODO）。环境硬要求也记一下：`genesis-world==1.2.1`（插件用到内部模块，版本敏感）、CPython 3.12、AMD GPU + Mesa RADV ≥ 24.3。

---

## §7 诚实对比：与 NVIDIA 那条链路，以及现在还差什么

把两条链路逐环节摊开：

| 环节 | NVIDIA（Isaac Sim + NuRec） | 本文（Genesis + vk_gs on AMD） |
|---|---|---|
| GPU / 驱动 | NVIDIA + CUDA 11.8+ | AMD RDNA4 + Mesa RADV，**零 CUDA** |
| 场景来源 | World Labs Marble 3DGS | **同一个** Marble 厨房 |
| PLY 处理 | 3DGRUT 转 USDZ（需 CUDA） | vk_gs **直读 `.ply`**，免转换 |
| 渲染 | NuRec / 3DGUT（Omniverse） | vk_gs 三管线（raster/RT/hybrid） |
| 统一光照 | ✅ 成熟 | ✅ 同 pass 统一渲染成立 |
| 物理引擎 | Isaac Sim | Genesis |
| 集成形态 | Isaac Sim 原生 | Genesis camera sensor 插件（对标 Nyx） |
| 开源 / 可改 | 闭源栈 | ✅ 全开源、可自修 |
| 成熟度 | 产品级 | PoC，渲染层已跑通 |

不吹，得说清两个**当前软肋**：

其一，**注入的机器人 mesh 现在是"平光照"**——偏灰、缺立体感，像剪纸贴片。根因是 vk_gs 渲染器默认 `LIGHTING_DISABLED`，mesh 颜色直接等于 baseColor，没有 N·L、没有光源、没有环境光。好消息是完整的 PBR + 方向光/点光 + 物理天空 + HDRI/IBL + 阴影基建**其实都在渲染器里**，只是默认关着、且 headless 的 pybind 还没把这些开关暴露出来；splat 本身 pre-lit 不受影响，缺光的只有注入的 mesh。这是**工程补齐**，不是架构缺陷。

其二，**整体还是 PoC**：渲染层（统一渲染）和集成层（在线 sensor + 物理行走）都已跑通，但离 NVIDIA 那种"描述一个世界、当天就能测"的产品成熟度还有距离——DLPack 零拷贝（Vulkan→HIP）、双相机抓取 episode、场景碰撞几何的自动化等还在规划。

真正想留下的判断是：**这条链路在 AMD 上不存在原理性障碍，缺的只是一条不碰 CUDA、还能自己改的开源渲染路径。** vk_gs + Genesis 把这条路走通了——同一个生成式厨房，NVIDIA 用一整套 CUDA 栈搬进 Isaac Sim，我们用纯 Vulkan 搬进了 Genesis。在"生成式世界模型喂机器人仿真"越来越热的当下，把这条链路从单一厂商的 CUDA 生态里解放出来，本身就是一件值得做的事。

---

## 写在最后

这件事说白了很简单：让 AI 生成一个照片级的厨房，再把机器人放进去，让它在里面走、被桌子挡住、被灯光照亮——像真的在那个房间里一样。NVIDIA 已经用自家一整套工具把这条路走顺了，只是那套工具是给 N 卡准备的。

我们想知道的是：换成 AMD 卡，同一个厨房、同一件事，还能不能做成？答案是能。关键的一步，是别把机器人"贴"在场景前面，而是让它和场景在同一次渲染里一起画出来——这样光影才对得上，看着才不假。剩下的，就是把这个渲染器接回物理引擎，让机器人真的能走、能撞、能站稳。最后 Go2 带着自己学来的走路策略，在这个 AI 生成的厨房里溜达了一圈，被橱柜挡住、贴着地走，没摔。

当然当前还有些 limitations：注入的机器人现在有点"发灰"、缺立体感，整套东西也还是个能跑通的雏形，离"描述一个世界、当天就能测"还差得远。但重要的是，这条路在 AMD 上并没有什么迈不过去的坎，缺的只是有人先把它趟出来、而且开源出来，让别人能接着改。

---

## References

- [vk-gsplat-plugin](https://github.com/ZJLi2013/vk-gsplat-plugin)
- [vk_gaussian_splatting](https://github.com/ZJLi2013/vk_gaussian_splatting)（渲染后端，nvpro-samples fork，`rdna4_support`）
- [Genesis](https://github.com/Genesis-Embodied-AI/genesis) · [genesis #1358](https://github.com/Genesis-Embodied-AI/genesis-world/issues/1358) · [genesis-nyx #18](https://github.com/Genesis-Embodied-AI/genesis-nyx/issues/18)
- [NVIDIA Isaac Sim × World Labs Marble 教程](https://developer.nvidia.com/blog/simulate-robotic-environments-faster-with-nvidia-isaac-sim-and-world-labs-marble/)
- [World Labs Marble export specs](https://docs.worldlabs.ai/marble/export/specs)（厨房 sample 资产下载）
- [MuGS](https://github.com/Renforce-Dynamics/MuGS) · [ROCm/gsplat（amd_gsplat）](https://github.com/ROCm/gsplat)
