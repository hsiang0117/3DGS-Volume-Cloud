# 环境光接入设计方案（Stage 2：冻结几何 + 大气入射光照 = T_sun 消光 + E_θ 天空填充，黑底无 cubemap）

> 状态：设计文档。2026-06-16 初稿（cubemap-PRT）；2026-06-22 改定「无 cubemap」；
> **2026-06-22 再修订**：① 加入太阳大气透射 `T_sun`（**纯加性 env 表达不出「变暗」**，UE 实证）；
> ② 已能采到**纯黑底 env-on**（SkyAtmosphere 瑞利/米氏/臭氧三系数为真值 + 天空亮度因子=0）
> → **删除全部 mask 监督**，改为黑底全图直接监督。Stage 1（仅太阳、黑底）已达 37.55，
> held-out 太阳零泛化差距。

---

## 1. 两阶段设计与一句话

**目标**：不破坏 Stage 1 已标定物理参数（β/ρ/g/octave/位置/协方差）的前提下，叠加天空大气
环境光，实现任意太阳的全光照 relighting。

- **Stage 1（已完成）**：仅太阳、黑底 → 物理参数稳定的高斯点集（37.55）。
- **Stage 2（本文档）**：**冻结**点集，新增**两个全局、只依赖太阳方向的可学项**——乘性太阳
  大气透射 `T_sun(sun_dir)` 与加性天空填充 `E_θ(sun_dir)`——配合**预计算无色传输 `V_lm`** 与
  **已有 albedo `ρ`**，把大气环境光加进着色。

---

## 2. 物理基础：环境效应 = 大气（瑞利+米氏+臭氧）对光做的两件事

**UE 实证（2026-06-22）**：把 SkyAtmosphere 的**瑞利散射 / 米氏散射 / 吸收(臭氧)** 三项强度全调 0，
画面就退回「只有方向光」(= env-off)。所以 env-on 相对 env-off 的**全部差异**，就是这三项产生的
**两个效应**：

1. **消光（乘性，≤1）**：三项把能量从**直射太阳光束**移除 → 太阳**变暗 + 变红**（瑞利 ∝λ⁻⁴
   把蓝散掉 → 偏红；米氏前向 → 红晕/光晕；臭氧吸收）。这是 `T_sun(sun_dir)`。
2. **内散射（加性，≥0）**：散进云体/视线的天空填充（瑞利 → 蓝天填充；米氏 → 雾状前向）。
   这是 `E_θ(sun_dir)`。

加性项（内散射）某高斯点 x、朝相机 ω_o：

```
L_fill(x, ω_o) = ρ(x) · ∫_{sphere} p(ω_i·ω_o) · E_sky(ω_i; sun) · T_sky(x, ω_i) dω_i
                                    └─相函数─┘  └──天空辐亮度──┘  └─该方向天空可见度─┘
```

- **p**：相函数（HG）。高反照率云的环境观感由多次散射主导、趋各向同性，v1 取各向同性。
- **E_sky**：天空**辐亮度场（radiance）**，随太阳方向变。
- **T_sky(x, ω_i)**：x 沿 ω_i 的天空可见度（环境遮挡）——**环境光的灵魂**：云核≈0、云边≈1。
  > ⚠️ 没有 T_sky 的填充 = 给所有高斯加常数底光，均匀提亮深核——正是 Stage 1 一路在防的
  > 「假底光」形态。遮挡结构不是可选项，是这一项的全部价值。

**太阳消光不是积分项,是对整个太阳项的乘性缩放**(消光发生在阳光进云之前)。

---

## 3. 核心方法：消光 `T_sun` ⊙ 太阳项 + PRT 填充 `E·V`

```
L = T_sun(sun_dir) ⊙ [ρ·L_sun·Σ wₙ T_light^bⁿ HG(...)]   ← 太阳项 ×全局RGB透射(≤1)：变暗+红
  + ρ · Σ_{lm} E_lm(sun_dir) · V_lm(x)                     ← 加性天空填充（≥0）
```

四类量的角色与可学性：

| 项 | 是什么 | 来源 | 可学？ |
|---|---|---|---|
| Stage-1 太阳项形状 | β/g/octave/T_light/位置/协方差 | Stage 1 | 冻结 |
| `ρ(x)` | 反照率，**逐高斯**，色度锁这里 | Stage 1 | 冻结 |
| `V_lm(x)` | 天空可见度 SH 投影，**逐高斯、无色、纯几何** | 预计算（§6） | 否 |
| `T_sun(sun_dir)` | 太阳大气透射，**全局** RGB，≤1 | 学 / 解析大气（§4） | **是（新增）** |
| `E_lm(sun_dir)` | 天空**辐亮度场**(radiance,非 irradiance)SH，**全局** RGB | 学 / 解析大气（§4） | **是（新增）** |

### 🔴 红线：env 绝不能引入「逐高斯的色彩自由度」

新增可学的只有**两个全局逐太阳函数** `T_sun`、`E_θ`；逐高斯自由度一个不加（还是 ρ/几何/β，
全冻结）→ 物理上**无法退回 vanilla 3DGS**、relighting 保住。

- `T_sun`/`E_θ` 是 RGB（太阳/天空本就有色），但**全局**（整场景共享）——逐高斯颜色仍只有 ρ。
- 全局色调歧义（T_sun/E 色 vs ρ 色）由**冻结 ρ**（env-off 白太阳下已定）解掉。
- 反例（必避）：env = 逐高斯可学颜色 / sun-conditioned per-Gaussian MLP 出色 = 退回 3DGS、杀 relighting。

### ⚠ 为什么必须乘性 `T_sun`（不能只加 `E_θ`）

加性项 `E_θ ≥ 0` **只能加亮、永远不能减暗**。低太阳时 env-on 比 env-off **更暗**，那必然是太阳
被大气消光到了白太阳以下（填充再多只能往回补、补不出更暗）。**纯加性环境光在结构上表达不出
「变暗」**，必须让太阳项乘 `T_sun(≤1)`——这是本次修订的核心。

---

## 4. 入射光照三种拿法：测 cubemap / 学 / 解析大气

环境 = 瑞利+米氏+臭氧大气 = **UE SkyAtmosphere = Hillaire 2020 / Bruneton 式解析模型**。它从
`sun_dir` + 三个（你已知的）系数**同时**吐出 `T_sun`（沿太阳光路 Beer-Lambert）和 `E`
（内散射积分 → SH）。所以入射光照有三种来源：

| 来源 | 怎么得 | 泛化 | 代价 |
|---|---|---|---|
| 测 cubemap | 采图投 SH | 离散需球面插值 | 整条采集/SH/抠盘/地平线管线 |
| **学（v1 推荐起步）** | 从 env_on 图学全局 `T_sun/E_θ` | held-out 太阳验证 | 砍掉 cubemap 管线；起步快 |
| **解析大气（endgame）** | 同系数跑 Hillaire 大气算 `T_sun/E` | **完美**（连续、无盘） | 复刻多次散射 LUT 才能像素级对上 UE |

因 L 对 `E`、`T_sun` 都线性（§7），可**先学后换解析、零重训**：先用学的 `T_sun/E_θ` 把云
**响应**机器验证好，以后把同系数的解析大气热插拔进去。
> 注:「零重训」以学到的 `(T_sun, E)` 能复刻解析输出为前提;若解析大气保真更高（multi-scatter
> LUT vs 学到的近似），可选 ≪100 步微调 `E_θ` 收尾，残差更小。

---

## 5. `T_sun(sun_dir)` 与 `E_θ(sun_dir)`：两个全局逐太阳函数

- **`E_θ`**：RGB 低阶 SH（SH2=9 系数/通道够低频）或小 MLP，**全局**。天生连续 + 无太阳盘
  （散射天空的低频积分）。
- **`T_sun`**：**全局** RGB，`exp(−τ_atm(sun_dir))` 或 sigmoid 保证 ∈[0,1]；几个标量或小 MLP。
  全局(云上均匀)成立于**云尺度 ≪ 大气标高(~8km)**;未来多高度/大尺度云再细化为 `T_sun(h)` 或逐光线查表。
- **输入** `sun_dir`（OpenGL world，同 `convert_transforms.py` 约定），逐帧从 camera 取，无需新字段。
- **太阳盘**：直射归 Stage-1 DirectionalLight 项；`E_θ` 必须**排除太阳盘**（否则与显式太阳重复
  计数）。监督目标 `env_on−env_off` 天然不含盘，自洽。

> 复杂度阶梯：`E_θ` 最简 SH0（逐高斯一个无色 AO × 全局天空色），SH2 拿「天顶暗、近地平线亮」
> 的梯度，建议直接 SH2。`T_sun` 起步可只学一个 RGB 标量场（按太阳俯仰/方位）。

---

## 6. `V_lm` 预计算（复用 `compute_T_light_raster`，不写新光追）

`gaussian_renderer/compute_T_light_raster(means3D, tau, scales, rotations, L_dir, ...)` 本就接受
**任意方向** `L_dir` 算逐高斯透射率。环境可见度 = 它对（上）半球的积分：

1. 上半球取 **N≈32–64 个 Fibonacci 方向**；
2. 每方向调一次 → 逐高斯 `T_sky(x, ω_j)`；
3. `{ω_j → T_sky}` 投到 SH2 → `V_lm(x)`，缓存。**无色、纯几何、只算一次。**

- **下半球**：UE `bLowerHemisphereIsBlack=true` → 采样/投影**只取上半球**（或下半球权重置零）。
- 存储：200k 高斯 × 9 系数 × float32 ≈ **7 MB**，sidecar 缓存（§10）。
- 可微性不需要——`V_lm` 是冻结几何的常量。（可选 v0：只投 SH0 = 逐高斯环境遮挡标量，先验管线。）

---

## 7. PRT 线性 → `E`/`T_sun` 可热插拔（关键性质 + 未来方向）

`L` 对 `E` 与 `T_sun` 都**线性**（**前提：`V_lm` 与 `ρ` 冻结**——任一可学/动态则线性破裂），而
transfer（`V_lm` / 可选 env-MS / `ρ`）**与天空长相无关**。因此：

- **transfer 只标定一次**；
- **运行时 `(T_sun, E)` 可来自任何地方**：学到的全局函数，或同系数的解析大气。

**未来方向（用户规划）**：自实现基于物理的大气（瑞利+米氏+臭氧），从 `sun_dir` 同时输出 `T_sun`
和 `E`——**云体环境着色与可见天空着色解耦、只共用 `sun_dir`**；因二者可热插拔，随时收紧到「共用
整片天空、完全一致、零重训」。**实现铁律**：env 接口做成 **`(T_sun, E_lm) 输入 → L 输出`**，
**别把天空烘进 `V_lm`**。

---

## 8. 数据采集：纯黑底 env-on（已跑通）

**配方（2026-06-22 已验证）**：SkyAtmosphere 三系数（瑞利/米氏/臭氧）保持**真实值**（→ 大气照云
+ 压暗太阳），**把天空亮度因子调 0** → **背景纯黑、云仍有大气环境光**。env-off = Stage 1 现有数据
（等价于三系数=0，只有方向光），**不重采**。所以只新增一套：**纯黑底 env-on 图**（同太阳/相机/曝光）。

- **不需要**冻结 SkyLight、**不需要**采 sky cubemap（`E` 是学/解析的，非测）。
- **曝光必须与 env-off 同一 EV0**，否则差分无意义（见 [[uniform-sun-dataset]] 教训）。
- **一个验证点**：跨太阳角确认 `E_θ`（蓝色填充）在「天空亮度因子=0」下是否仍在——若该因子把
  内散射填充也压没了，则采到的 env-on 主要是 `T_sun`（太阳消光）、`E_θ≈0`。预期：**低太阳 `T_sun`
  主导（红+暗），高太阳填充更明显**。据此决定 `E_θ` 是否需要、还是 `T_sun` 独挑。

---

## 9. Stage 2 训练什么 + 监督（黑底 → 全图直接监督，**无 mask**）

冻结 Stage 1（其渲染 = 纯太阳项）。新增可学：**`T_sun(sun_dir)`、`E_θ(sun_dir)`**（均全局），
可选**环境云内多次散射**（复用六阶 octave 思路 `T_sky^(bⁿ)` + 无色能量权重，类比 `octave_w`）。

**监督 = 纯黑底全图直接监督**：模型渲云 on 黑（bg=0），直接对 `env_on`（黑底）算 L1 + DSSIM。

- **不需要任何 mask**：背景两边都 0（GT 黑底、模型 bg=0）→ 背景像素 **0=0 平凡匹配**、无 show-through，
  loss 信号天然集中在云上。这正是「纯黑底 env-on」相对带天空背景方案的最大简化——**整套掩膜 /
  逐像素 final_T / 方向性天空合成全部不再需要**。
  > 前提:模型 bg **固定为 0、不可学**;GT 与模型**抗锯齿核一致**(否则云边像素会在黑区引入泄漏)。
  > 可选自检:统计 GT 黑区里 `0 < 模型色 < 1e-3` 的像素数,确认无意外泄漏。
- 等价地可监督 `env_on − env_off`（黑底下 = 纯大气效应：`T_sun` 对太阳项的缩放 + `E·V` 填充）。
- 太阳/环境分离由**数据**给定（两套图，非模型猜）；冻结 `ρ` 解全局色调歧义。

---

## 10. 实现改动清单（复用 tonemap 的独立优化器 + sidecar）

### `scene/gaussian_model.py`
- `_sky_transfer`（(P, n_sh) buffer，**非 nn.Parameter**，预计算 `V_lm`）。
- `_sky` / `E_θ`（全局天空 SH (n_sh,3) 或小 MLP）；`_t_sun` / `T_sun`（全局 RGB，sigmoid/exp 保证 ≤1）；可选 `_k_env`。
- `precompute_sky_transfer(N, order)`：复用 `compute_T_light_raster` → SH（上半球，下半球 0）。
- `apply_env(sun_dir)` → 逐高斯 `ρ·Σ E_θ(sun)·V_lm`；`sun_atten(sun_dir)` → `T_sun(sun)`（可微）。
- **独立优化器** `env_optimizer`（仅全局新参数 `E_θ`/`T_sun`/`k_env`）；隔离 densify/prune（理由同
  tonemap：`_prune_optimizer` 对每组无脑 `[mask]`，全局参数会崩）。
- **sidecar**：`sky_transfer.npy` + `env.json`（`E_θ`/`T_sun` 参数、sh_order、N），镜像 `tonemap.json`，`load_ply` 自动恢复。

### `arguments/__init__.py`
- `PipelineParams`：`env_lighting=False`、`env_sh_order=2`、`env_transfer_dirs=48`、`env_octave_reuse=False`。
- `OptimizationParams`：`env_lr`。**ρ 保持冻结**（不提供解冻开关——解冻会重开全局色调歧义、破坏
  已冻结太阳项的标定，红线）。

### `gaussian_renderer/__init__.py`（render）
- 网关：`if pipe.env_lighting and pc 有 transfer`：
  `Lk = T_sun(cam.sun_dir) ⊙ sun_term + pc.apply_env(cam.sun_dir)`。
- 环境项与 `T_sun` 缩放都在**线性空间**、tonemap 之前：`(T_sun⊙sun + E·V) → ACES → loss`。
  诊断通道（override_color）与现有 tonemap 网关不变。

### `train_env.py`（Stage 2 入口，新脚本）
- 加载 Stage 1 PLY（冻结），`precompute_sky_transfer` 一次；只建 `env_optimizer`；
  **黑底全图直接监督 `env_on`（无 mask）**；复用 tonemap 的 step / sidecar 保存模式。

### `scene/dataset_readers.py` + `cameras.py`
- env-on 图作为第二套图读入（同太阳同相机），挂到 camera；`sun_dir` 已有路径不变。

### `viewer.py`
- auto-detect `env_lighting` + 加载 sidecar；GUI 环境光开关/强度；**无极太阳直接 evaluate
  `(T_sun, E_θ)` 做 relighting**（连续、无盘、无插值、无冻结亮点）。背景默认黑。

### （可选）光栅化器 `final_T` —— 仅 viewer/配图想把云衬在天空前时才需
- **训练黑底不需要 `final_T`**。仅当 viewer/配图要合成方向性天空背景，才暴露逐像素 `final_T`
  （当前 forward 返回 `color, radii, invdepths, contribution, tau_front_sum, tau_front_wsum`，
  **无逐像素 alpha**；`diff_gaussian_rasterization/__init__.py:96`）+ 重编译。属可选 cosmetic，不挡训练。

---

## 11. viewer / 配图（背景默认黑，sky 合成是可选 cosmetic）

- 训练与默认 viewer：**黑底**。relighting = 无极太阳直接 evaluate `(T_sun, E_θ)`。
- 想配图把云衬在天空前（可选）：用 `E_θ`/解析大气重建低频天空 + 沿**连续 sun_dir** 叠**解析太阳盘**
  （随滑块走、无冻结错位），框架外合成 `C + final_T·SKY`（需上面 `final_T`）。背景**从不被训练**
  → 随便好看，不破坏一致性（模型=云+环境光两端一致，背景只是末端贴图）。

---

## 12. 跨框架对比协议（DSYG）

DSYG 的 VPRF = 纯发射 SH、无光照输入 → **不能 relighting**；只能当**固定光照新视角重建/紧凑度
基线**，**不是 held-out 太阳 relighting 基线**（结构上做不到）。

- **黑底消变量**：GT / 本方法 / DSYG 全用黑底渲染 → 可见背景恒 0、平凡一致；**无需 mask**（两边
  0=0）。DSYG(Mitsuba) 不挂 environment emitter → 打空返回黑。
- **指标可全图（黑底）算**；若担心大片黑像素抬高 PSNR，可**附报**云区指标（GT `invdepth>0`），
  但对**公平性非必需**（两边同样的黑像素）。
- **同 tonemap/色彩空间**：两边输出施加**同一变换**再算指标（本管线 ACES，Mitsuba 出线性 HDR）——
  **最易翻车的跨框架坑**。
- **同数据契约**：同图/位姿/分辨率/split/曝光。
- **定性天空配图**：两边只输出 `(C, final_T)`，框架外共享脚本合成同一片天空 → 像素级一致。
- **范围声明**：① 固定太阳「体积云重建质量 + 紧凑度」与 DSYG 比；② **relighting(held-out 太阳)
  是本方法独有，不与 DSYG 比**（同时凸显贡献）。

---

## 13. 验证（端到端）

1. **物理 sanity**：仅 `T_sun`+`E_θ`、冻结一切，看预测对不对得上 `env_on`（或 `env_on−env_off`）；残差小=结构正确。
2. **变暗/变红方向性**：低太阳 `T_sun<1` 且偏红是否复现；**加性-only 基线对照应做不出「变暗」**
   → 反向佐证 `T_sun` 的必要性。
3. **残差分桶**：复用 `tools/residual_buckets.py` / `penumbra_residual.py`，确认填充补对「朝亮天
   一侧」、且**无均匀提亮深核**（假底光回归检查）。
4. **relighting 泛化**：**4 个 held-out 太阳**（time_index 7/22/37/52）测 `T_sun/E_θ` 外推；viewer
   无极太阳看观感。残差小 = 学到的大气泛化成立。
5. **回归**：`env_lighting=False` 时行为与 Stage 1 完全一致（不加载 transfer、不改 sun_term）。

---

## 14. 风险与坑（务必前置）

- **🔴 退化为 3DGS**：env 只许「全局 `T_sun/E_θ` × 逐高斯无色 `V` × 已有 `ρ`」，**禁逐高斯色彩 DOF**（含 sun-conditioned per-Gaussian MLP）。
- **⚠ 加性表达不出变暗**：必须乘性 `T_sun`（§3）。
- **太阳盘双重计数**：`E_θ` 必须无盘；直射归 DirectionalLight；监督目标 `env_on−env_off` 天然无盘。
- **下半球**：`bLowerHemisphereIsBlack=true`，`V_lm` 采样/投影只取上半球。
- **曝光必须锁定**：env-on 与 env-off 同一 EV0。
- **sun_dir 约定**：云 HG 用 `ω_in=−sun_dir`（见 [[g-collapse-was-phase-sign-bug]]）；`T_sun`/`E_θ`/未来大气同约定。
- **ρ 保持冻结**：解冻会重开全局色调歧义、破坏太阳项标定。
- **差分二阶耦合**：`env_on−env_off=纯大气效应` 假设太阳项与环境散射不耦合；实际多次散射可能引入小残差，由可选 env-MS 项吸收（v1 可接受）。典型云(ρ~0.7-0.9、τ~1-2)残差约 **5–15%**;训练时记录 `‖env_on−env_off−(T_sun⊙L_sun+E·V)‖`,**>20% 才开 env-MS**。
- **解析大气 vs UE 多散射**：单次散射对不上 UE 的 multi-scatter LUT；要像素级匹配需复刻 Hillaire。**先学后换**兜底。
- **部署**：线性辐亮度塞回 UE 输出线性、让 UE 自己 tonemap，勿双重 ACES。

---

## 15. v1 明确不做

- view-dependent 环境相函数瓣（v1 各向同性；残差显方向性再加）。
- 环境光下重新 densify / 动几何（几何严格冻结）。
- **逐高斯色彩自由度 / 学习逐高斯颜色**（红线）。
- 采 sky cubemap（学/解析，除非泛化不过关才回退）。
- 动态时变天空插值（`E_θ`/`T_sun` 已连续）。
- **训练期 mask / 方向性天空背景合成**（纯黑底 → 全图直接监督；天空合成是 viewer 可选 cosmetic）。

---

## 16. 建议的第一步

数据已可得（§8 纯黑底 env-on）。直接按 §10 接入：① 预计算 `V_lm`；② 加全局 `T_sun` + `E_θ`；
③ **黑底全图直接监督 `env_on`（无 mask）**；④ sanity（`T_sun` 复现变暗/红）+ 4 个 held-out 太阳
验证泛化。之后（endgame）把同系数解析大气的 `(T_sun, E)` 热插拔进来（§4 / §7）。

相关记忆：[[env-lighting-no-cubemap]]（本方案决策、红线、T_sun、黑底采集）、[[uniform-sun-dataset]]
（env-off 控制变量、固定曝光、手动 py 采集）、[[tlight_raster_arc]]（compute_T_light_raster 复用）、
[[g-collapse-was-phase-sign-bug]]（sun_dir 约定）。
