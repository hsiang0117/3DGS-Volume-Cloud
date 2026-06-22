# 环境光接入设计方案（Stage 2：固定几何 + PRT 环境光）

> 状态：设计文档（2026-06-16；2026-06-18 修订）。Stage 1（仅太阳光、黑背景数据集）
> 已达优秀水平（uniform 数据集 ~33.3 PSNR，held-out 太阳零泛化差距）。本文档定义
> Stage 2：冻结已标定的高斯点集，以预计算辐射传输（PRT）的形式叠加环境光。

---

## 1. 背景与两阶段设计

**目标**：在不破坏 Stage 1 已标定物理参数（β/ρ/g/octave/位置/协方差）的前提下，
给着色叠加天空环境光的贡献，实现真正的全光照 relighting（任意太阳 + 任意天空）。

**两阶段划分**（用户设计，本文档采纳）：

- **Stage 1（已完成）**：仅太阳光、黑背景数据集 → 得到物理参数稳定的高斯点集。
- **Stage 2（本文档）**：**冻结**高斯点集（几何 + 物理参数），仅训练环境光相关的
  少量自由度，以某种形式给着色加上环境光。

**关键洞察**：这正是 **Precomputed Radiance Transfer（PRT, Sloan et al. 2002）**
的标准框架。"冻结几何"不是工程妥协，而是让环境光变得极廉价且稳定的**前提**——
几何固定 ⇒ 光传输（遮挡）固定 ⇒ 可一次性预计算 ⇒ Stage 2 前向只剩一个 SH 点积。

**场景现状（2026-06-16 / 06-18 经 MCP 核实 /Game/Maps/Cloud）**：场景已配
`SkyAtmosphere` + `SkyLight`(bAffectsWorld, RealTimeCapture, LowerHemisphereIsBlack)
+ `SceneCaptureCube → /Game/VolumetricCloud/RT_SkyCube`（HDR, bCaptureEveryFrame）。
**环境 cubemap 捕获机制已现成**，采集侧不需新建。云本体是 `HeterogeneousVolume`。

**env-off / env-on 的开关机制**：Stage 1 的黑背景是把 **`SkyAtmosphere` 设为不可视**
实现的（背景纯黑、云只受太阳）；改回**可视**即得 env-on——蓝天背景 + 天空环境光
对云着色（06-18 视口确认：受光面亮白，背光面带淡蓝天光填充、非纯黑，薄边透出天空）。
所以 `img_env_off`（SkyAtmosphere 不可视）和 `img_env_on`（可视）只差这一个开关，
天然成对，正是 Stage 2 监督所需的两套数据。

---

## 2. 物理基础

某高斯点 x 处、朝相机方向 ω_o 的环境内散射辐亮度：

```
L_env(x, ω_o) = ρ(x) · ∫_{sphere} p(ω_i · ω_o) · E_sky(ω_i) · T_sky(x, ω_i) dω_i
                                  └─相函数─┘   └─天空辐射─┘  └─该方向天空可见度─┘
```

三个因子：
- **p**：相函数（HG）。高反照率云的环境观感由**多次散射主导**，趋于各向同性，
  v1 可近似为各向同性（丢掉 view-dependent 的相函数瓣，作为一阶足够）。
- **E_sky(ω_i)**：天空辐射场 = cubemap。**已知输入**（UE 场景我们控制）。
- **T_sky(x, ω_i)**：x 沿 ω_i 方向对天空的透射率（环境遮挡 / 天空可见度）。
  **这是环境光的灵魂**——云核 T_sky≈0（天光进不去），云边 T_sky≈1。

> ⚠️ **没有 T_sky 的环境项 = 给所有高斯加常数底光**，会均匀提亮深核——正是
> Stage 1 一路在防的"假底光"形态（见 [[octave 假底光证伪]] 教训）。遮挡结构
> 不是可选项，是这一项的全部价值所在。

---

## 3. 核心方法：PRT 分解

各向同性近似下（环境项相函数 ≈ const），积分塌成**逐高斯传输向量**与
**逐帧天空 SH** 的点积：

```
L_env(x) ≈ ρ(x) · Σ_{lm}  E_lm · V_lm(x)
                          └天空SH┘ └传输向量┘
```

- **E_lm**：天空 cubemap 的 SH 系数（逐帧输入，随太阳方向变）。
- **V_lm(x)**：逐高斯天空可见度 T_sky(x, ·) 的 SH 投影 = PRT "transfer 向量"。
  **只依赖冻结的几何（β、位置、协方差）** ⇒ **Stage 2 只需算一次，缓存复用**。

存储量：200k 高斯 × 9 系数（SH2）× float32 ≈ **7 MB**，缓存到 PLY 同目录
（镜像 `tonemap.json` 的 sidecar 套路，见 §7）。

着色总式：

```
L = L_sun_term(已冻结、已标定)  +  ρ · Σ_{lm} E_lm · V_lm(x)
    └────── Stage 1 原封不动 ──────┘   └──── 叠加的新环境项 ────┘
```

方向性（云朝亮天一侧更亮）由 V_lm 与 E_lm 共同保留——**不是平铺 ambient**。

### env-on 整图 = 内散射 + 背景透射（关键分解）

注意 env-on 画面里有**两个**环境效应，必须分清（否则监督会混淆）：

```
img_env_on(px) = [ρ·(sun_scatter + k_env·Σ E_lm V_lm)] 经相机透射累积   ← 云内散射
                + T_cam(px) · E_sky(view_dir(px))                      ← 背景天空透射
```

- **内散射**：唯一的**新学习项**（锁在共享 ρ + 全局 `k_env`）。
- **背景天空透射**（天空从云薄处/背后透过来）：**不是新东西**。光栅化器本就做
  `C + T·bg`（`forward.cu:485`），背景经云的逐像素最终透射率 T 合成——viewer 切
  背景色看到的就是它。**零新增学习参数**，唯一改动是把常数 bg 换成方向性天空（§5）。

这条很重要：env-on 里**绝大部分是确定性合成**（背景透射 = 已知天空 × 模型自身 T），
可学的只有内散射的 k_env(/可选 ρ)——进一步缩小了可退化面（见 §9 决策 3 / 退化担忧）。

---

## 4. 问题一：cubemap 如何接入训练管线

**结论：不喂原始 cubemap，投到低阶 SH，作为逐帧输入，与 `sun_dir` 完全同等待遇。**

1. **不可学**：E_sky 是已知输入。一旦把它设成可学习参数，就把"某一个特定天空"
   烤进模型 → relighting 报废。与太阳同理：光源是输入，云的物理参数才是学习量。
2. **SH 阶数**：环境光低频，**SH2（9 系数/通道）通常够**；UE SkyLight 内部本就
   按 SH 存，天作之合。SH3（16/通道）作为精度后备（**待定，见 §9**）。
3. **坐标系**：world 空间（OpenGL Y-up，与模型一致）。SH 基定义在 world，
   与 `convert_transforms.py` 的太阳方向同一坐标约定，避免二次旋转。
4. **数据格式**：每帧 JSON 加 `sky_sh: [[r,g,b]×系数数]`，key 跟太阳方向走
   （"每个太阳方向一张 cubemap" 正确——SkyAtmosphere 让天空随太阳变）。

### ⚠️ 采集的关键坑：捕获 sky cubemap 时必须隐藏云体

cubemap 要的是"**云所在位置看到的纯天空辐射场**"，云的遮挡由 T_sky 单独算。
若捕获时云（HeterogeneousVolume）在场，则**遮挡被算两遍**（cubemap 里一次、
V_lm 里一次）→ 双重压暗。采集脚本须在 cube 捕获前隐藏云体、捕获后恢复。

---

## 5. 问题二：环境光以什么形式融入着色

**结论：PRT 传输向量 V_lm，复用现有 `compute_T_light_raster` 预计算。**

### V_lm 怎么算（复用现有机器，不写新光追）

`gaussian_renderer/compute_T_light_raster(means3D, tau, scales, rotations, L_dir, ...)`
本就接受**任意方向** `L_dir` 算逐高斯透射率。环境可见度 = 它对半球的积分：

1. 在（上）半球取 **N≈32–64 个 Fibonacci 方向**；
2. 每方向调一次 `compute_T_light_raster` → 逐高斯 T_sky(x, ω_j)；
3. 把 {ω_j → T_sky} 投到 SH2 → V_lm(x)，缓存。

> `bLowerHemisphereIsBlack=true`：UE SkyLight 下半球为黑（地面方向不补光），
> 采样方向与 SH 投影应与之一致（仅上半球，或下半球权重置零），保证物理对齐。

代价：预计算 N×（一次 raster pass 很快），**仅一次**。可微性不需要——V_lm 是
冻结几何的常量。

### 前向（Stage 2 训练 + 推理）

逐高斯一个 SH 点积，**极便宜**。这正是"冻结几何"的回报：传输固定 ⇒ 预计算掉
⇒ 训练只剩 `E_lm · V_lm` 点积 + 少量自由度。

### 环境多次散射（可选精化）

环境项可复用 Stage 1 的六阶 octave：`T_sky^(bⁿ)` + **同一套已学到的 octave
权重**，环境多次散射免费复用、无新参数。**v1 先纯各向同性，octave 复用待定（§9）**。

### 背景透射：复用现有 `C+T·bg`，唯一改动是 bg → 方向性天空

背景透射**不需要新着色项**——光栅化器 `forward.cu:485` 已做
`out_color = C + T·bg_color`：背景经逐像素最终透射率 T 合成（viewer 切 bg 色即此）。
逐像素剩余透射率 `final_T` 也已在 kernel 算好（`forward.cu:482`），只是当前**未经
Python 绑定返回**（色彩路径只返回 color/radii/depth/contribution）。

env-on 唯一变化：bg 从**常数色**变成**方向性天空**（cubemap 按每像素视线方向采样）。
两条实现路径：

1. **暴露 `final_T`**（已存在，只差绑定层加一个返回），外部合成
   `img = cloud_render(bg=0) + final_T · sky(view_dir)`——最干净，改动是把已有缓冲透出来；
2. 改 kernel 让 `bg` 接受逐像素背景图。

背景透射**零新增学习参数** = 已知天空 × 模型自身透射率。

---

## 6. Stage 2 训练什么 + 监督方式

E_lm 已知、V_lm 预计算、ρ 冻结 ⇒ L_env **几乎完全确定**。这是好事（高度受约束，
不会破坏已标定几何）。需要决定加几个自由度：

- **最小（推荐起点）**：一个全局环境强度标量 `k_env`。先做 sanity——纯物理预测
  能否对上 GT？`k_env≈1` 且残差小 ⇒ 物理正确实锤。
- **稍多**：解冻 ρ 微调（太阳光与环境光**共享 albedo**，物理正确，且让 ρ 吸收
  更多打光信息）；或学一个环境相函数阶数。**待定（§9）**。

### 监督：env-on 直接监督（背景用方向性天空合成）

模型在 env-on 模式下渲：云内散射(sun+env) 经现有 `C+T·bg` 机制合成到**方向性天空
背景**上（§5）。直接对 `img_env_on` 监督。背景透射由已知天空 × 模型 T 确定（零参数），
太阳项冻结，所以 **loss 能动的只有内散射的 `k_env`(/可选 ρ)**。任何残差纯属内散射
模型本身。与 tonemap/octave 实验同一方法论：先隔离、再合并。

> **为什么不直接拿差分图**：`img_env_on − img_env_off` = 内散射 + 背景透射 之和
> （env_off 是黑底、无透射）。直接用差分监督内散射会把背景天空透射混进来，归因不干净。
> 背景透射既然是确定性的，更干净的做法是把它当**已知合成掉、直接监督 env_on**，而非相减。
> 若仍想用差分，必须额外减去已知的 `T_cam·E_sky` 背景项才等于纯内散射。

---

## 7. 实现改动清单（复用 tonemap 基建）

环境光的全局/逐高斯参数完全可以照搬刚为 learnable tonemap 建好的模式。

### `scene/gaussian_model.py`
- `_sky_transfer`（(P, n_sh) buffer，**非 nn.Parameter**，预计算的 V_lm）+
  `_k_env`（标量 nn.Parameter，可选解冻的环境强度）。
- 新方法 `precompute_sky_transfer(N_dirs, sh_order)`：复用
  `compute_T_light_raster` 取 N 方向 → 投 SH → 写入 `_sky_transfer`。
- 新方法 `apply_env(E_lm)`：返回逐高斯 `ρ · Σ E_lm V_lm`（可微 in E_lm 与 _k_env）。
- **独立优化器** `env_optimizer`（仅 `_k_env` / 解冻的 ρ），隔离 densify/prune
  （Stage 2 不 densify，但隔离仍是安全惯例；理由同 tonemap，见 `_prune_optimizer`
  对每组无脑 `[mask]` 的坑）。
- **sidecar 持久化**：`sky_transfer.npy` + `env.json`（k_env、sh_order、N_dirs）
  写到 PLY 同目录，镜像 `tonemap.json`。`load_ply` 自动恢复。

### `arguments/__init__.py`
- `PipelineParams`：`env_lighting=False`（开关）、`env_sh_order=2`、
  `env_transfer_dirs=48`、`env_octave_reuse=False`。
- `OptimizationParams`：`env_lr`、`unfreeze_albedo_in_env=False`。

### `gaussian_renderer/__init__.py`（render）
- 网关：`if getattr(pipe,"env_lighting",False) and pc 有 transfer`：
  `Lk = sun_term + pc.apply_env(viewpoint_camera.sky_sh)`（内散射）。
- **背景透射**：渲云用 bg=0，取逐像素 `final_T`，外部合成
  `img = cloud + final_T · sky(view_dir)`，`sky(view_dir)` = cubemap 按视线方向采样。
- 逐帧 `sky_sh`（投影系数）+ sky cubemap 从 camera 取（dataset_readers 注入，
  同 `sun_dir` 路径）。诊断通道（override_color）与现有 tonemap 网关不变。

### `submodules/diff-gaussian-rasterization`（绑定层）
- 暴露 kernel 已算的逐像素 `final_T`（`forward.cu:482`）到 Python 返回值，供背景
  方向性天空外部合成。当前色彩路径只返回 color/radii/depth/contribution。

### `train.py`（Stage 2 入口，可能是新脚本 `train_env.py`）
- 加载 Stage 1 的 PLY（冻结），调 `precompute_sky_transfer` 一次。
- 冻结主优化器（或只建 `env_optimizer`），用差分图 / env_on 图监督。
- 复用 tonemap optimizer 的 step / sidecar 保存模式。

### `scene/dataset_readers.py` + `cameras.py`
- 读取每帧 `sky_sh`，挂到 camera（同 `sun_direction` 现有路径）。

### `viewer.py`
- auto-detect `env_lighting` + 加载 sky_transfer/env sidecar。
- GUI：天空 SH 编辑（或预设若干天空）→ 实时 relighting 演示。

### `tools/`（数据采集，**第一步、不需训练代码**）
- 扩 `cloud_dataset_generator.py`（手动 py 模式，避开会崩的 MCP 插件），每太阳方向三采：
  - （a）**sky cubemap**：SkyAtmosphere 可视 + **隐藏 HeterogeneousVolume** → 捕获
    `RT_SkyCube` → 投 radiance SH（`E_lm`）；
  - （b）**env_on 图**：SkyAtmosphere 可视 + 云可见（蓝天背景 + 环境光着色）；
  - （c）**env_off 图**：SkyAtmosphere 不可视 + 云可见 = **Stage 1 现有数据**，无需重采。
  即 env_off/on 只差 SkyAtmosphere 可见性一个开关。

---

## 8. 复用的既有实现

- `compute_T_light_raster`（gaussian_renderer/__init__.py）：V_lm 预计算的逐方向
  透射率，直接复用，不写新光追。
- tonemap 的**独立优化器 + sidecar**模式（gaussian_model.py 的 `setup_tonemap` /
  `tonemap.json` save/load）：env 参数照搬。
- octave 着色循环（render 的 `T_eff = T_light^(ms_b^n)` + 学到的 octave_w）：
  环境多次散射复用（可选）。
- `metrics.json` / `tonemap.json` sidecar 写法：`env.json` / `sky_transfer.npy` 镜像。
- `cloud_dataset_generator.py` 手动 py 采集模式 + throttle 关闭：env 双渲扩展。

---

## 9. 待定的设计决策（需用户拍板）

1. **SH 阶数**：SH2（9/通道，够低频）还是 SH3（16/通道，更准但 transfer 翻倍）？
2. **环境多次散射**：v1 纯各向同性，还是复用 octave 做 `T_sky^(bⁿ)`？
3. **Stage 2 自由度**：只放 `k_env`（纯验证物理自洽）还是同时解冻 ρ 微调？
4. **监督信号**：env-on 直接监督（背景方向性天空合成）为主；差分图为备选（须额外
   减去已知 `T_cam·E_sky` 背景项才等于纯内散射）。倾向直接监督。
5. **数据规模**：每个太阳方向各一张 sky cubemap + env_on 图，复用现有 60 太阳？

---

## 10. 验证（端到端）

1. **物理自洽 sanity**：仅 `k_env`、冻结一切，看预测能否对上差分图；`k_env≈1` 且
   残差小 = 物理正确。
2. **残差分桶**：复用 `tools/residual_buckets.py` / `penumbra_residual.py`，确认
   环境光是否补对了"朝亮天一侧"，且**没有均匀提亮深核**（假底光回归检查）。
3. **relighting 泛化**：held-out 天空（留几个太阳方向的 cubemap 不参与训练）测
   外推；viewer 里手动换天空 SH 看观感。
4. **回归**：`env_lighting=False` 时行为与 Stage 1 完全一致（不加载 transfer、
   不改 sun_term）。

---

## 11. 风险与坑（务必前置）

- **双重遮挡**：捕获 sky cubemap 时未隐藏云 → cubemap 和 V_lm 各算一遍遮挡 →
  深核双重压暗。**采集脚本必须隐藏云体**（§4）。
- **环境光变可学习 fudge**：若 E_sky 不当输入而去学，或 `k_env` 在无差分监督下
  自由漂移，会变成补别的误差的 fudge（重蹈 aniso 针 / octave 假底光覆辙）。
  **E_sky 必须是输入，监督必须用差分图隔离**。
- **下半球**：UE `bLowerHemisphereIsBlack=true`，采样/投影须一致，否则给云底
  补了 UE 里不存在的光。
- **tonemap 叠加**：Stage 2 仍在 ACES tonemap 空间训练（默认）；环境项在**线性
  空间**加进 sun_term 之后、tonemap 之前。顺序：`(sun+env) → ACES → loss`。
- **背景须用方向性天空，不是常数**：env-on 背景是蓝天渐变 + 太阳侧偏亮，常数 bg
  合成会在背景及云薄边与 GT 失配。须把 bg 喂方向性天空（cubemap 视线采样）。注意：
  show-through 本身零参数、是确定性合成，**不是**退化风险，只是要喂对背景。
- **部署**：线性辐亮度场塞回 UE 时输出线性、让 UE 自己 tonemap，勿双重 ACES。

---

## 12. v1 明确不做

- view-dependent 环境相函数瓣（v1 各向同性；若残差显方向性再加）。
- 环境光下重新 densify / 动几何（Stage 2 几何严格冻结）。
- 学习天空（E_sky 永远是输入）。
- per-channel 之外的色彩自由度（色度锁在 ρ，原则不变）。
- 动态时变天空插值（先离散每太阳一张，插值留后续）。

---

## 13. 建议的第一步

**不写训练代码**：先准备 UE 采集脚本（隐藏云 + sky cubemap 双渲 + SH 导出），
把干净的 env_on / sky_sh 监督信号拿到手。数据一到，按 §7 接入即可。

相关记忆：[[uniform-sun-dataset]]（env-off 控制变量设计、采集走手动 py）、
[[tlight_raster_arc]]（compute_T_light_raster 复用）。
