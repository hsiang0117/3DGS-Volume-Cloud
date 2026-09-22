# 3DGS-Volume-Cloud

用物理参数化的 3D Gaussian Splatting 替代游戏引擎中 ray-marching 体积云的研究项目,目标是**实时渲染 + 动态打光**(任意太阳方向 relighting)。

基于 [3D Gaussian Splatting (Kerbl et al., 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 的代码框架,对表示、着色、光栅化器和训练管线做了体积介质方向的重构。

**两阶段设计**:
- **Stage 1** —— 仅太阳光、黑背景数据集,训出物理参数稳定的高斯点集(关键差异见下文)。
- **Stage 2** —— **冻结** Stage 1 的几何与物理参数,只训一个全局环境光网络,叠加天空大气对云的着色(任意太阳的全光照 relighting,见下文「环境光」小节)。

## 与原版 3DGS 的核心差异

### 物理化的高斯参数

每个高斯不再携带 SH 颜色 + 经验 opacity,而是一组参与介质(participating medium)物理量:

| 参数 | 含义 | 激活 |
|---|---|---|
| `σ_t` | 消光截面(长度²,消光系数的体积积分) | softplus, clamp 5 |
| `ω` | 散射反照率(RGB) | sigmoid |
| `g` | Henyey-Greenstein 相函数各向异性因子 | 0.8·tanh,前向散射 |
| `w_n` | 6 阶多次散射八度能量权重(可学习,Frostbite/Wrenninge 八度近似) | softplus |

### 解析光学厚度光栅化

光栅化器(fork 的 diff-gaussian-rasterization)支持 `use_analytic_tau` 分支:per-pixel 累积的是高斯沿视线的**解析线积分光学厚度** τ,α = 1−exp(−τ),物理正确的 Beer-Lambert 消光,而非启发式 alpha。

### 物理着色

逐高斯计算 `L = ω · L_light · Σₙ wₙ · T_light^(bⁿ) · P(v_i, v_o; g·cⁿ)`:HG 相函数(cosθ = v_i·v_o,入射方向 v_i = −v_l)+ 六阶多次散射八度 + 自阴影透射率 T_light。太阳方向 v_l 逐帧来自数据集,推理时可任意替换 → relighting。

### 输出 tonemap(默认匹配 GT 显示空间)

UE 的 HighResScreenshot GT 是 **filmic-tonemapped LDR**,而物理着色在**线性空间**。用线性模型拟合非线性目标会表现为动态范围压缩。默认开启 **固定 Narkowicz ACES** 曲线:着色端放宽 per-高斯辐亮度 clamp 到 HDR、图像端套 ACES,使 loss 与全部指标都在 GT 自己的空间比较。

- `--tonemap_learnable`(可选,默认关):把 ACES 的 4 个系数(a,b,c,d)变可学习(e 钉死),自适应 GT 的真实显示曲线,保留为**换其他 filmic 引擎**的保险;系数存进 PLY 同目录 `tonemap.json`,viewer/eval 自动读取。
- **若 GT 是真·线性 HDR**(无 tonemap):应**关闭** tonemap(命令行传 `--no-tonemap_aces`),而非用 learnable——Narkowicz 族无法表示 identity。物理模型本身线性、与渲染器无关;部署回 UE 实时渲染时输出**线性辐亮度**让 UE 自己 tonemap,**勿重复套 ACES**。

### 光源视角自阴影(T_light,默认路径)

T_light = 每个高斯沿太阳方向的"前方遮挡透射率"。默认实现为**光照空间光栅化 pass**:

- 远距窄 FOV 透视相机伪装方向光太阳(视差 <2%,免改 EWA 雅可比);
- CUDA `record_front_tau` 通道:深度序遍历中,每高斯记录其前方累积 τ 的 α·T 加权均值(整个向阳 footprint 上的能量加权,而非中心点采样);
- **原生可微 backward**:默认保留旧的部分梯度(τ 路径,权重与光照足迹冻结);`--tlight_full_grad` 补齐权重、归一化、弱密度备用透射率及投影位置/尺度/旋转的连续梯度。太阳相机构图、排序、覆盖与阈值分支仍视为常量;
- 深埋高斯(early-termination 导致 wsum=0)显式映射为全阴影,防反转;
- `--tlight_voxel` 回退到旧的 128³ 体素缓存路径(与 raster 之前训练的模型配套;viewer 的 `--tlight auto` 读 cfg_args 自动匹配)。

### 针手术(结构性 aniso 控制,默认开启)

软正则压不住的高各向异性尾巴由 `split_needles` 结构性重写:每 1000 迭代,ratio>30 的高斯增肥薄轴 ×2(ratio 减半)、σ_t/3.2 近似守恒消光截面、沿主轴劈成两子。等效硬上限,不与光度梯度拔河。

### 物理化的致密化与维护

- 贡献度 prune(per-Gaussian Σ(α·T) CUDA 通道)替代 opacity 阈值;
- σ_t resurrect 替代 stock 的 reset_opacity(σ_t 参数化下 opacity 是解析量);
- 自适应 densify 阈值(top-K% 梯度分位);
- 维护回路(resurrect→prune→reset)**只在 densify 期运行**,densify 结束即门控关闭,防止 settle 期的净销毁。

### 环境光(Stage 2:冻结几何 + 解析大气)

Stage 1 是 env-off 控制变量(纯太阳、黑背景)。Stage 2 **冻结**已标定的高斯点集,只训一个**全局、仅依赖太阳方向**的环境网络,在太阳着色之上叠加天空大气贡献:

```
L = T_sun(v_l) ⊙ [Stage 1 太阳项]  +  ω · Σ_lm E_lm(v_l) · V_lm(x)
    └── 乘性:太阳大气透射 ──┘            └──── 加性:天空内散射填充 ────┘
```

- **`T_sun(v_l)`** —— 太阳穿过大气的逐通道透射率,**3 参数解析式** `exp(−m(θ)·τ_rgb)`:`m(θ)` 是 Kasten-Young air mass(固定几何),只学天顶光学厚度 `τ=(τ_R,τ_G,τ_B)`。低太阳变暗 + 变红(τ_B>τ_R,Rayleigh ∝λ⁻⁴)、方位对称都从结构自动落出;太阳落到地平线下时 smoothstep 门控熄灭(无直射)。**纯加性项表达不出"变暗",所以太阳项必须乘 T_sun**(≤1)。
- **`E_lm(v_l)`** —— 天空辐亮度场的低阶 SH(小全局 MLP),加性内散射填充。
- **`V_lm(x)`** —— 逐高斯天空可见度的 SH 传输向量(环境遮挡),**无色、纯几何**,在冻结点集上复用 `compute_T_light_raster` 扫半球 N 方向预计算一次。
- **红线**:新增可学的只有全局 `T_sun`/`E_θ`,**逐高斯不加任何色彩自由度**(色度锁在冻结的 ω)→ 物理上无法退回 vanilla 3DGS、relighting 保住。
- **监督**:env-on 数据集是**纯黑背景**(SkyAtmosphere 天空亮度因子=0,但保留瑞利/米氏/臭氧对云的打光),所以全图直接监督、**无需 mask**(背景两边都 0)。
- 环境网络与 `V_lm` 存进 PLY 同目录 sidecar(`env_net.pt` / `sky_transfer.npy` / `env.json`),viewer/eval 自动加载。

**设计取舍 / 未来方向**:这套大气就是标准的瑞利 + 米氏 + 臭氧模型(= UE SkyAtmosphere = Hillaire/Bruneton)。因为 `L` 对 `(T_sun, E_lm)` **线性**(`V_lm`、`ω` 冻结),二者是**可热插拔的输入**——transfer 只标定一次,运行时既可用学到的网络,也可换成同系数的**解析大气**(把物理天空投影成 SH 喂进 `E_lm`),云响应零重训。这正是"云体环境着色与可见天空着色解耦、只共用 `sun_dir`"的接口形态:**别把天空烘进 `V_lm`**,保持 `(T_sun, E_lm) 输入 → L 输出`。

**跨框架对比**(如对照 *Don't Splat your Gaussians* 的 VPRF):用**黑底**把"可见天空"这个变量消掉(GT/本方法/对照方都渲黑底 → 背景恒 0、平凡一致、无需 mask),从 GT 算一次云掩膜套到两边,且两边输出施加**同一 tonemap/色彩空间**再算指标(最易翻车的跨框架坑)。注意纯发射式 SH 重建(VPRF)结构上**不能 relighting**,只能当固定光照的重建/紧凑度基线,held-out 太阳 relighting 是本方法独有。

## 数据集

UE5 渲染的体积云(WDAS cloud VDB),73 个相机(球冠布置:极点 + 六圈,天顶角 0°–135°)× 多太阳方向,NeRF-synthetic transforms 格式 + 逐帧 `sun_direction`。

**现行数据集 `data/CloudDatasetUniform`**:60 个 Fibonacci 均匀半球太阳 × 轮转 1/3 相机(每太阳 24–25 视角、每相机 20 太阳)= 1460 帧;train 1308 / test 152,其中 **4 个太阳方向整体 held-out**(96 帧)作为 relighting 泛化测试。方向均匀覆盖是几何阴影梯度健康工作的前提(方向有偏的数据集会让垂直方向的延展逃逸监督)。

**Stage 2 的 env-on 数据集**(如 `CloudDatasetUniform_envon`):**同位姿、同太阳、同曝光**,只把 UE 的 SkyAtmosphere 天空亮度因子设 0(背景纯黑、保留大气对云的打光)重采一遍 → 云带环境光、背景黑。位姿/split 与 env-off 一致,可复用同一套 `--held-out-suns`。

数据集在磁盘上的形态(`data/CloudDatasetUniform/`,共 4 个 JSON + 73 个 `camXX/` + 初始点云):

| 文件 | 内容 |
|---|---|
| `transforms.json` | UE 左手系原始输出(采集产物,训练不用) |
| `transforms_train_full.json` | 全集 1460 帧的 OpenGL 右手系备份(**重切 split 的唯一依据,勿删**) |
| `transforms_train.json` | 训练集 1308 帧 |
| `transforms_test.json` | 测试集 152 帧 |

划分口径:test = **4 个整太阳方向**(`time_index ∈ {7,22,37,52}`,各 24 帧 = 96 帧)
+ 其余 56 个太阳各 1 个未见视角(56 帧)= 152 帧。
`transforms_train_full.json` 保留全集,因此**任何 split 都可从它重切**(幂等,不会丢帧)。

> 重采(只换曝光/光照、相机位姿与太阳方向不变)时,位姿与 split 完全一致,可直接
> 复用现有 transforms_train/test.json、只替换 cam*/images/,无需重切。

采集与切分管线(`tools/`):

```shell
# 1. [UE 编辑器内执行] 采集:均匀太阳数据集,输出 transforms.json + camXX/images/*.png
#    自动关闭后台 CPU 节流——否则失焦时截图永不落盘
py "<repo>/tools/cloud_dataset_generator.py" -o D:/CloudDatasetUniform

# 2. UE 左手系 → OpenGL 右手系(含 sun_direction),缺省写成训练全集 transforms_train.json;
#    写新全集时会清除过期的 transforms_train_full.json / transforms_test.json
python tools/convert_transforms.py D:/CloudDatasetUniform/transforms.json

# 3. 划分 held-out 测试集(整太阳 relighting 泛化 + 每太阳 1 帧):
python tools/split_test_set.py --data D:/CloudDatasetUniform --held-out-suns 7,22,37,52 --per-sun 1
#    → 备份全集到 transforms_train_full.json,写 transforms_train.json(train)+ transforms_test.json(test)
#    (幂等:重跑不会丢帧。旧 CloudDataset 的 per-camera 切分:省略 --held-out-suns,用 --per-cam 2)
```

> ⚠️ **诊断与天空采集脚本仍不在仓库内**(原 `tools/` 下**尚未恢复**的 6 个):
> `residual_buckets.py`、`penumbra_residual.py`、`analyze_octave_weights.py`、
> `plot_phase_function.py`、`project_pointcloud.py`、`ue_capture_sky_backdrop.py`。
> 采集 / 转换 / 切分 / 评测均已可用,训练与查看不受影响;
> 若要复跑残差诊断或多散射八度分析,需从 git 历史(或本地备份)取回这批脚本。

## 使用

### 训练

```shell
# Stage 1 — 默认:raster T_light + 部分光照梯度 + 针手术 + 固定 ACES tonemap
python train.py -s data/CloudDatasetUniform

# 旧体素 T_light 路径(与 raster 之前训练的模型配套;数据路径按需改)
python train.py -s data/CloudDatasetUniform --tlight_voxel

# 可学习 tonemap(默认关,换 filmic 引擎时的保险)
python train.py -s data/CloudDatasetUniform --tonemap_learnable

# Stage 2 — 冻结 Stage 1 模型,在 env-on 数据集上只训环境光网络
python train.py --stage2 --stage1_model output/<stage1_run> -s data/CloudDatasetUniform_envon
```

eval 默认开启(test split 不并入训练),结束时在 test 集上输出 PSNR/SSIM/LPIPS 并写 `metrics.json`。PipelineParams 持久化进 cfg_args,供 viewer 自动匹配 T_light 源。Stage 2 额外按 held-out / 已见太阳分组报告 PSNR + env 贡献(env-on 减 env-off)。

<details>
<summary><b>训练命令行参数完整说明</b>(点击展开)</summary>

#### 数据与输出(ModelParams)

| 参数 | 默认 | 说明 |
|---|---|---|
| `-s, --source_path` | (必填) | 数据集目录(含 transforms_train/test.json + points3d.ply) |
| `-m, --model_path` | 自动时间戳 | 输出目录(checkpoint / cfg_args / metrics.json) |
| `-r, --resolution` | -1 | 训练分辨率;-1 = 原始(宽 >1.6K 时自动缩到 1.6K),1/2/4/8 = 对应降采样 |
| `-w, --white_background` | False | 白色训练背景(默认黑) |
| `--data_device` | cuda | 图像缓存设备;显存紧张可设 cpu |
| `--eval` | **True** | test split 不并入训练。默认 True 的 bool 用 `BooleanOptionalAction` 注册,可用 **`--no-eval`** 把 test split 并回训练(即全量训练),无需改源码 |

#### 渲染管线(PipelineParams)

| 参数 | 默认 | 说明 |
|---|---|---|
| `--tlight_voxel` | False | **回退**到旧 128³ 体素 T_light(默认为光照空间光栅化;完整连续梯度由 --tlight_full_grad 控制);与 raster 之前训练的模型配套 |
| `--tlight_raster_res` | 512 | 光照 pass 的太阳相机分辨率(阴影分辨率) |
| `--tlight_tau_filter` | False | 光照实验:保留最小像素足迹,通过幅值补偿保持理想二维 τ 积分;不影响相机主 pass |
| `--tlight_full_grad` | False | 光照实验:固定构图和离散分支,补齐光照估计器的连续梯度 |
| `--tonemap_aces` | **True** | 默认开启:图像端套固定 Narkowicz ACES,匹配 UE filmic GT 空间。真·线性 GT 数据用 **`--no-tonemap_aces`** 关闭(默认 True 的 bool 走 `BooleanOptionalAction`),无需改源码 |
| `--tonemap_learnable` | False | 可选:让 ACES 的 4 系数可学习(独立优化器,系数存 `tonemap.json`),保留作换其他 filmic 引擎的保险;开启时优先于固定 ACES |

#### 环境光(Stage 2)

| 参数 | 默认 | 说明 |
|---|---|---|
| `--stage2` | False | 启用 Stage 2:加载并冻结 `--stage1_model`,只训环境光网络;`-s` 指向 env-on 数据集 |
| `--stage1_model` | (Stage 2 必填) | 要冻结的 Stage 1 输出目录(或 point_cloud.ply);其结果不被改动 |
| `--env_sh_order` | 2 | 天空辐亮度 `E_lm` 与可见度 `V_lm` 的 SH 阶数(SH2 = 9 系数) |
| `--env_transfer_dirs` | 48 | 预计算 `V_lm` 时在半球上采样的方向数 |
| `--env_lr` | 1e-3 | 环境光网络(全局 `T_sun` + `E_lm` MLP)学习率;独立优化器,衰到 0.1× |

#### 调度与学习率(OptimizationParams)

| 参数 | 默认 | 说明 |
|---|---|---|
| `--iterations` | 30000 | 总迭代数 |
| `--position_lr_init / _final` | 1.6e-4 / 1.6e-6 | 位置学习率退火起止(×spatial_lr_scale) |
| `--position_lr_max_steps` | 30000 | 位置退火长度。**应与 iterations 同步**,否则各向异性会失控 |
| `--position_lr_delay_mult` | 0.01 | 位置 LR 预热系数 |
| `--sigma_t_lr` | 0.025 | σ_t 学习率 |
| `--omega_lr` | 0.0025 | 反照率 ω 学习率 |
| `--g_lr` | 0.0025 | HG g 学习率 |
| `--w_lr` | 0.0025 | 多次散射八度权重学习率 |
| `--scaling_lr` | 0.005 | 尺度学习率 |
| `--rotation_lr` | 0.001 | 旋转学习率 |

物理参数(σ_t/ω/g/w)的 LR 全程指数退火到 1/10。

#### 损失与正则

| 参数 | 默认 | 说明 |
|---|---|---|
| `--lambda_dssim` | 0.2 | DSSIM 损失权重(L = 0.8·L1 + 0.2·DSSIM) |
| `--lambda_scale` | 0.1 | 体积正则(∏s 均值),抑制高斯无界膨胀 |
| `--lambda_aniso` | 0.001 | 软各向异性正则(log-ratio 二次,超过 aniso_ratio_max 才罚)。数值过大将损伤重建质量;硬约束交给针手术 |
| `--aniso_ratio_max` | 5.0 | 软正则的免罚阈值 |
| `--aniso_until_iter` | 30000 | 软正则作用区间。**必须全程**——各向异性不自收敛,提前关闭会持续恶化 |
| `--tonemap_lr` | 1e-3 | 可学习 tonemap 4 系数的学习率(仅 `--tonemap_learnable` 时生效;独立 Adam,衰到 0.1×) |
| `--lambda_tonemap_mono` | 1e-2 | 可学习 tonemap 单调性惩罚(仅 `--tonemap_learnable` 时;hinge 平方,保证曲线在 [0,8] 不反转,高光不倒挂) |

#### 致密化(densify)

| 参数 | 默认 | 说明 |
|---|---|---|
| `--densify_from_iter / _until_iter` | 500 / 15000 | 致密化区间。**结束后留出 settle 收敛期** |
| `--densification_interval` | 100 | 致密化周期 |
| `--densify_grad_threshold` | 1e-4 | 位置梯度阈值(densify_adaptive=False 时生效) |
| `--densify_adaptive` | True | 自适应阈值:每轮取梯度 top `densify_top_frac`,梯度后期衰减也不停摆 |
| `--densify_top_frac` | 0.005 | 自适应模式的 top 分位(0.5%) |
| `--densify_grad_min` | 5e-5 | 自适应阈值的绝对下限 |
| `--densify_scale_grad_threshold` | 1e-6 | 尺度梯度并入致密化判据的换算阈值 |
| `--percent_dense` | 0.01 | clone/split 的尺寸分界(×场景半径) |

#### 剪枝与维护

| 参数 | 默认 | 说明 |
|---|---|---|
| `--contribution_threshold` | 1e-4 | 贡献度剪枝阈值:mean Σ(α·T) 低于此值剪除(替代 stock 的 opacity 阈值) |
| `--prune_min_visible_frames` | 5 | **仅约束贡献通道**:至少在 N 帧可见才用平均 Σ(α·T) 判定剪枝。另一条"死点"通道(当前窗口内一帧都没被看到)剪枝时**不受此值保护**——从未被看见的高斯没有可判定的均值,若也要求可见就永远剪不掉 |
| `--contribution_reset_interval` | 1000 | 贡献度累计器清零周期(保持统计反映当前模型) |
| `--resurrect_interval` | 3000 | 每 N 迭代把贡献度最低的一批 σ_t 重置回 0.1(替代 stock reset_opacity)。**仅 densify 期间生效**——settle 期运行会与剪枝形成净销毁回路 |
| `--resurrect_fraction` | 0.05 | 每次 resurrect 的点数占比 |
| `--post_densify_prune_interval` | 1000 | densify 期内的额外剪枝周期;0 关闭。**注:维护(resurrect/prune/reset)只在 densify 期运行,densify 结束后即停**——settle 期运行会与剪枝形成净销毁回路 |

#### 针手术(极端各向异性分裂)

| 参数 | 默认 | 说明 |
|---|---|---|
| `--needle_split_interval` | 1000 | 手术周期;**0 = 关闭** |
| `--needle_split_ratio` | 30.0 | 分裂触发阈值(max/min 轴比) |
| `--densify_until_iter` | 15000 | 与增密共用的截止迭代,严格小于该值时才允许针手术 |

针手术直接复用增密门控 `iteration < densify_until_iter`。默认每 1000 步检查一次,最后一次检查为 14000 步;从 15000 步起不再增密或执行针手术,继续优化现有高斯的参数。调整 `--densify_until_iter` 会同步调整两者的截止时间。旧的独立参数 `--needle_split_until_iter` 已移除,训练命令请改用 `--densify_until_iter`。

#### 调试与日志(train.py)

| 参数 | 默认 | 说明 |
|---|---|---|
| `--test_iterations` | 7000 30000 | 在这些迭代做 test/train 评估(PSNR/SSIM/LPIPS) |
| `--save_iterations` | 7000 30000 | 保存 checkpoint 的迭代(末迭代总会保存) |
| `--detect_anomaly` | False | torch autograd 异常检测(很慢) |
| `--quiet` | False | 静默模式 |

</details>

### 评估

```shell
# 训练内评测:`--eval` 默认开启,末次迭代在 test split 上写 metrics.json(PSNR/SSIM/LPIPS)

# 对已保存的 checkpoint 独立复算同一套指标(与 train.py 的 test 循环逐行等价):
python tools/eval_testset.py output/<run> [iteration]     # iteration 缺省=最新

# 分组评估:held-out 太阳组 vs 已见太阳新视角组(T_light 源从 cfg_args 自动读取)
python tools/eval_test_groups.py output/<run> [iteration]
```

两个脚本都从 `cfg_args` 还原 `source_path` / T_light 源 / tonemap 模式,并由 PLY 的 sidecar
自动判断是否启用环境光。注意它们按 `resolution=-1`、黑底加载相机;若某 run 用非默认
`-r/--resolution` 或 `-w` 训练,独立复算的分辨率口径会与训练内评测不同。

### 云区指标

`tools/eval_cloud_region.py` 同时报告全图、原始 VDB mask、扩张 mask、新增外环及矩形 crop。
背景占比会影响指标的绝对值，但不必然缩小方法间的 PSNR 差距。

```shell
# Cloud-GS；掩膜默认 D:\dataset\CloudDatasetMasks，可用 --masks 覆盖：
python tools/eval_cloud_region.py output/<run> --iteration 30000 --save

# 官方 3DGS：在其仓库根目录、使用其 venv（需安装 lpips）：
python D:/PythonProjects/3DGS-Volume-Cloud/tools/eval_cloud_region.py output/<run> --repo official --iteration 30000 --save

# 只评估原始 mask / 全图 / crop，不附加扩张区域：
python tools/eval_cloud_region.py output/<run> --dilate 0 --save
```

- 原始 mask 始终独立报告为 `*_mask_raw`；`--dilate 16` 默认额外报告方形核扩张的
  `*_mask_dilated` 和新增外环 `*_ring`，不将它们混称为原始云体。
- 默认 crop 为**原始 mask 的 bbox + 16 px margin**，与膨胀半径无关。
  `--crop-base dilated --margin 16` 改用扩张 mask 的包围框；`--margin` 可显式调整边距。
- 所有 PSNR 统一为 RGB 合并 MSE 后转 dB，再逐帧等权平均。区域 SSIM 是原图的
  11×11 SSIM map 在 mask 内窗口中心的均值，边界窗口仍可包含区域外像素。
- LPIPS 使用 VGG / v0.1，仅计算全图与矩形 crop；不把 mask 外置零，不对 crop 缩放。
  输入统一为连续 NCHW float32，关闭 cuDNN TF32 并固定卷积选择，避免存储布局造成指标偏移。
  `crop_area_share` 是 crop 占全图比例，`mask_raw_share_in_crop` 是原始 mask 占 crop 比例。
- 掩膜是逐相机单通道二值 PNG（camXXX.png），来自 VDB 密度支持域射线求交，具体定义见
  `_metadata/manifest.json`。相同云体、变换、相机内外参和分辨率下，不同光照与方法共用同一套。
  脚本检查 mask 非空、二值且尺寸匹配，不自动缩放；`per_view` 使用完整帧路径，保留同机位多光照记录。
- `--save` 写入包含迭代数、区域参数与时间戳的新 JSON，保留旧结果。
  `--output FILE` 可指定一个尚不存在的文件。输出包含模型/配置/掩膜/相机文件哈希及指标定义。
  `--iteration` 默认 -1（自动选最新并记录实际迭代），正式对比建议显式固定。
- 当前支持 cloud / official 两种渲染路径、黑底数据，按 resolution=-1 加载相机。


### 交互 Viewer

```shell
python viewer.py --ply output/<run>/point_cloud/iteration_30000/point_cloud.ply

# 可选:HDR cubemap 天空背景(替换纯色底)
python viewer.py --ply output/<run>/.../point_cloud.ply --sky_dir data/sky_backdrop
```

基于 viser:实时改变太阳方向(relighting)、可视化通道(RGB / T_light / σ_t / depth)、可调背景色、snap 到训练相机。`--tlight auto|voxel|raster` 控制阴影源(auto 读训练 run 的 cfg_args)。加载 Stage 2 模型时自动检测 env sidecar 并开启环境光(太阳滑块同时驱动 `T_sun` + `E_lm`,可勾选框 A/B 开关)。

**天空背景(`--sky_dir`,可选,纯展示)**:给定一组 per-太阳高度的 HDR cubemap(见下),viewer 把云合成到真实天空前而非纯色底。太阳**高度**滑块选 cube、**方位**滑块旋转它(SkyAtmosphere 绕天顶轴旋转对称,唯一破对称的太阳随之转)。合成在 **rasterizer 内一趟完成**(逐像素 `bg_image` 线性 over + 单次 tonemap),"Sky backdrop" 勾选框开关,`Sky exposure`(默认 3.35)/`Sky warmth`(默认 0.09)对齐 UE 视口观感。**纯 viewer 展示,不进训练、不碰冻结的 albedo**;诊断通道保持纯色底。

## 已知限制

- **评估范围**:全部实验基于 UE 渲染的单朵合成云资产(WDAS cloud VDB)与受控光照数据;对其他云型、密度分布以及真实拍摄数据的泛化尚未验证。
- **光源近似**:光源空间阴影采用远距离透视相机近似平行太阳光;云体范围较大或太阳方向接近地平线时,该近似可能引入误差。
- **多次散射近似**:六阶 HG 八度展开是实时外观模型,并非严格能量守恒的多重散射解,在半影与光学厚度较大的区域可能留有残差。
- **Stage 1 数据集是刻意 env-off 的控制变量设计**:UE 场景只有云 + 单方向太阳,背景纯黑,无天空/大气环境光。注意 env-off 控制掉的是**环境光**,但 UE 体积管线仍计算**云内多次散射**——模型的六阶 octave 近似即用于拟合该效应。
- **残差诊断结论(脚本已移除)**:近受光半影处存在轻微偏亮残差,主要来自单次散射项与 HG 前向散射,优先级低。结论来自已移除的诊断脚本。

## 环境

Python 3.12 + CUDA 12.8(见 `requirements.txt`):

```shell
pip install -r requirements.txt
```

torch/torchvision 用 PyTorch index 的 cu128 wheel;`submodules/` 下三个 CUDA 扩展
(`diff-gaussian-rasterization` 含本项目的 analytic-tau / record_front_tau /
lightpass-backward 通道、`simple-knn`、`fused-ssim`)是本地编译,需 CUDA 工具链。

### 编译 CUDA 扩展

**改动 CUDA kernel 或更换 torch 后需重新编译**。三个要点:

- **`--no-build-isolation`**:`setup.py` 在构建期 `import torch`,而 pip 的隔离构建环境里没有 torch;
- **`--force-reinstall --no-deps`**:包版本恒为 `0.0.0`,不加这个 pip 会判定"已满足"而直接跳过,编译结果装不进去;
- **Windows 还需选对 MSVC toolset**:CUDA 12.8 的 `host_config.h` 要求 `_MSC_VER < 1950`(即 MSVC 19.4x 及以下),而 VS2026 默认的 14.50 恰好是 1950、会被直接拒绝 —— 用 `vcvarsall.bat x64 -vcvars_ver=14.44` 显式选并排安装的 14.44,不要用 `-allow-unsupported-compiler` 绕过版本检查。

```powershell
# Windows(PowerShell)。<VS> 形如 D:\Program Files\Microsoft Visual Studio\2026\Community
cmd /c "call `"<VS>\VC\Auxiliary\Build\vcvarsall.bat`" x64 -vcvars_ver=14.44 && set" `
  | % { if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($matches[1])" $matches[2] -EA 0 } }
$env:DISTUTILS_USE_SDK = "1"
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --force-reinstall --no-deps `
  ./submodules/diff-gaussian-rasterization
```

```shell
# Linux:上面 Windows 相关的坑都不存在(编译器版本由 gcc/nvcc 组合决定)
pip install --no-build-isolation --force-reinstall --no-deps ./submodules/diff-gaussian-rasterization
```

`simple-knn`、`fused-ssim` 同理,把路径换掉即可(两者改动频率低,只在换 torch 时重编)。

> 两个 Windows 排查提示:① 默认 toolset 本来就 ≤ 14.44 时 `-vcvars_ver` 可省略;
> ② 若终端把控制台代码页设成了 65001(UTF-8),torch 探测 MSVC 版本时会因 `cl.exe`
> 输出 UTF-8 中文、而它固定按 `'oem'`(cp936)解码而报 `UnicodeDecodeError`,先
> `chcp 936` 即可 —— 这不是工具链问题。

## 致谢

代码基于 [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)(Inria & MPII,LICENSE.md 沿用其非商业研究许可)。云资产来自 Walt Disney Animation Studios 公开的 [WDAS Cloud](https://disneyanimation.com/resources/clouds/) 数据集。

```bibtex
@Article{kerbl3Dgaussians,
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal = {ACM Transactions on Graphics},
  number  = {4},
  volume  = {42},
  month   = {July},
  year    = {2023},
  url     = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
