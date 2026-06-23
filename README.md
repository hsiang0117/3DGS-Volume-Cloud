# 3DGS-Volume-Cloud

用物理参数化的 3D Gaussian Splatting 替代游戏引擎中 ray-marching 体积云的研究项目,目标是**实时渲染 + 动态打光**(任意太阳方向 relighting)。

基于 [3D Gaussian Splatting (Kerbl et al., 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 的代码框架,对表示、着色、光栅化器和训练管线做了体积介质方向的重构。

## 与原版 3DGS 的核心差异

### 物理化的高斯参数

每个高斯不再携带 SH 颜色 + 经验 opacity,而是一组参与介质(participating medium)物理量:

| 参数 | 含义 | 激活 |
|---|---|---|
| `β_peak` | 峰值消光系数(1/m) | softplus, clamp 5 |
| `ρ` | 散射反照率(RGB) | sigmoid |
| `g` | Henyey-Greenstein 相函数偏度 | 0.8·tanh,前向散射 |
| `octave_w` | 6 阶多次散射能量权重(可学习,Frostbite/Wrenninge 八度近似) | softplus |

### 解析光学厚度光栅化

光栅化器(fork 的 diff-gaussian-rasterization)支持 `use_analytic_tau` 分支:per-pixel 累积的是高斯沿视线的**解析线积分光学厚度** τ,α = 1−exp(−τ),物理正确的 Beer-Lambert 消光,而非启发式 alpha。

### 物理着色

逐高斯计算 `L = ρ · L_sun · Σₙ wₙ · T_light^(bⁿ) · HG(g·cⁿ, cosθ)`:HG 相函数(ω_in = −sun_dir 约定)+ 六阶多次散射八度 + 自阴影透射率 T_light。太阳方向逐帧来自数据集,推理时可任意替换 → relighting。

### 输出 tonemap(默认匹配 GT 显示空间)

UE 的 HighResScreenshot GT 是 **filmic-tonemapped LDR**,而物理着色在**线性空间**。用线性模型拟合非线性目标会表现为动态范围压缩。默认开启 **固定 Narkowicz ACES** 曲线:着色端放宽 per-高斯辐亮度 clamp 到 HDR、图像端套 ACES,使 loss 与全部指标都在 GT 自己的空间比较(均匀数据集实测 30.80→**33.27**,压缩残差 −67%)。

- `--tonemap_learnable`(可选,默认关):把 ACES 的 4 个系数(a,b,c,d)变可学习(e 钉死),自适应 GT 的真实显示曲线。实测在 UE 数据上 ≈ 固定 ACES(−0.14 dB,否定结果),保留为**换其他 filmic 引擎**的保险;系数存进 PLY 同目录 `tonemap.json`,viewer/eval 自动读取。
- **若 GT 是真·线性 HDR**(无 tonemap):应**关闭** tonemap(源码翻 `tonemap_aces=False`),而非用 learnable——Narkowicz 族无法表示 identity。物理模型本身线性、与渲染器无关;部署回 UE 实时渲染时输出**线性辐亮度**让 UE 自己 tonemap,**勿重复套 ACES**。

### 光源视角自阴影(T_light,默认路径)

T_light = 每个高斯沿太阳方向的"前方遮挡透射率"。默认实现为**光照空间光栅化 pass**:

- 远距窄 FOV 透视相机伪装方向光太阳(视差 <2%,免改 EWA 雅可比);
- CUDA `record_front_tau` 通道:深度序遍历中,每高斯记录其前方累积 τ 的 α·T 加权均值(整个向阳 footprint 上的能量加权,而非中心点采样);
- **原生可微 backward**(`lightpassBackwardCUDA`):back-to-front 重放 + 运行和,把 dL/dτ_front 传播给前方所有遮挡者;完整几何梯度(β 和 σ_d 经 scale/rotation)默认开启;
- 深埋高斯(early-termination 导致 wsum=0)显式映射为全阴影,防反转;
- `--tlight_voxel` 回退到旧的 128³ 体素缓存路径(与 raster 之前训练的模型配套;viewer 的 `--tlight auto` 读 cfg_args 自动匹配)。

四版梯度设计迭代(detach → straight-through → 原生 backward → 完整梯度)的教训:β 必须保留穿过 T_light 的负反馈;前向与反向必须是同一个阴影场;几何阴影梯度需要方向均匀的数据兜底。

### 针手术(结构性 aniso 控制,默认开启)

软正则压不住的高各向异性尾巴(实测 95% 是薄盘而非针)由 `split_needles` 结构性重写:每 1000 迭代,ratio>30 的高斯增肥薄轴 ×2(ratio 减半)、β/3.2 守恒消光质量、沿主轴劈成两子。等效硬上限,不与光度梯度拔河。实测 aniso p99 ~22、max≈阈值,**PSNR 不降反升**。

### 物理化的致密化与维护

- 贡献度 prune(per-Gaussian Σ(α·T) CUDA 通道)替代 opacity 阈值;
- β_peak resurrect 替代 stock 的 reset_opacity(β 参数化下 opacity 是解析量);
- 自适应 densify 阈值(top-K% 梯度分位);
- settle 期(densify 结束后)维护回路门控,防止 resurrect→prune 净销毁。

## 数据集

UE5 渲染的体积云(WDAS cloud VDB),73 个半球相机 × 多太阳方向,NeRF-synthetic transforms 格式 + 逐帧 `sun_direction`。

**现行数据集 `data/CloudDatasetUniform`**:60 个 Fibonacci 均匀半球太阳 × 轮转 1/3 相机 = 1458 帧;train 1306 / test 152,其中 **4 个太阳方向整体 held-out**(96 帧)作为 relighting 泛化测试。方向均匀覆盖是几何阴影梯度健康工作的前提(方向有偏的数据集会让垂直方向的延展逃逸监督)。

采集管线(`tools/`):

```
tools/cloud_dataset_generator.py   # UE 编辑器内执行;均匀太阳数据集(输出目录见文件底部 OUTPUT_DIR)
tools/convert_transforms.py        # UE 左手系 → OpenGL 右手系(含 sun_direction)
tools/split_test_set.py            # test split(幂等):--held-out-suns 整太阳 / --per-cam 逐相机
```

UE 内一行启动(自动关后台 CPU 节流——否则失焦时截图永不落盘):

```
py "D:/3DGS-Volume-Cloud/tools/cloud_dataset_generator.py"
```

采集后处理(从 UE 输出到可训练数据集):

```shell
# 1. generator 在 UE 输出目录写出 transforms.json(UE 左手坐标系)+ cam*/images/*.png
#    (transforms.json 是唯一产物;目录里若有旧的 transforms_opengl.json 是上次 convert 的残留)

# 2. UE → OpenGL 转换,缺省直接写成训练全集 transforms_train.json
python tools/convert_transforms.py D:/CloudDatasetUniform/transforms.json
#    (缺省输出同目录 transforms_train.json;写新全集时会清除过期的
#     transforms_train_full.json / transforms_test.json,避免下一步从旧备份重切)
#    若数据集要进 repo:把 D:/CloudDatasetUniform 整个拷到 data/ 再切

# 3. 划分 held-out 测试集(整太阳 relighting 泛化 + 每太阳 1 帧):
python tools/split_test_set.py --data D:/CloudDatasetUniform --held-out-suns 7,22,37,52 --per-sun 1
#    → 备份全集到 transforms_train_full.json,写 transforms_train.json(train)+ transforms_test.json(test)
#    train.py 默认 --eval 不并回 test
#    (旧 CloudDataset 的 per-camera 切分:省略 --held-out-suns,用 --per-cam 2)
```

> 重采(只换曝光/光照、相机位姿与太阳方向不变)时,位姿与 split 完全一致,可直接
> 复用现有 transforms_train/test.json、只替换 cam*/images/,无需重转重切。

## 使用

### 训练

```shell
# 默认:raster T_light + 完整几何梯度 + 针手术 + 固定 ACES tonemap
python train.py -s data/CloudDatasetUniform

# 旧体素 T_light 路径
python train.py -s data/CloudDataset --tlight_voxel

# 可学习 tonemap(默认关,换 filmic 引擎时的保险)
python train.py -s data/CloudDatasetUniform --tonemap_learnable
```

eval 默认开启(test split 不并入训练),结束时在 test 集上输出 PSNR/SSIM/LPIPS 并写 `metrics.json`。PipelineParams 持久化进 cfg_args,供 viewer 自动匹配 T_light 源。

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
| `--eval` | **True** | test split 不并入训练。store_true 无法从命令行关闭,如需全量训练改源码 |

#### 渲染管线(PipelineParams)

| 参数 | 默认 | 说明 |
|---|---|---|
| `--tlight_voxel` | False | **回退**到旧 128³ 体素 T_light(默认为光照空间光栅化 + 完整几何梯度);与 raster 之前训练的模型配套 |
| `--tlight_raster_res` | 512 | 光照 pass 的太阳相机分辨率(阴影分辨率) |
| `--tonemap_aces` | **True** | 默认开启:图像端套固定 Narkowicz ACES,匹配 UE filmic GT 空间(+2.5 dB)。store_true 无法从命令行关闭,真·线性 GT 数据需改源码 |
| `--tonemap_learnable` | False | 可选:让 ACES 的 4 系数可学习(独立优化器,系数存 `tonemap.json`)。UE 数据上为否定结果(−0.14 dB),保留作换 filmic 引擎的保险;开启时优先于固定 ACES |
| `--k_sigma` | 0.0 | per-tile max-response 深度排序偏移(σ 单位);0 = stock 中心深度排序。曾用于治 popping,因块状伪影弃用,CUDA 路径保留 |

#### 调度与学习率(OptimizationParams)

| 参数 | 默认 | 说明 |
|---|---|---|
| `--iterations` | 30000 | 总迭代数 |
| `--position_lr_init / _final` | 1.6e-4 / 1.6e-6 | 位置学习率退火起止(×spatial_lr_scale) |
| `--position_lr_max_steps` | 30000 | 位置退火长度。**应与 iterations 同步**——拉长会延缓主阶段退火,实测 aniso 失控、-0.6 dB |
| `--position_lr_delay_mult` | 0.01 | 位置 LR 预热系数 |
| `--extiction_lr` | 0.025 | β_peak 学习率 |
| `--feature_lr` | 0.0025 | 反照率 ρ 学习率 |
| `--g_factor_lr` | 0.0025 | HG g 学习率 |
| `--octave_weights_lr` | 0.0025 | 多次散射八度权重学习率 |
| `--scaling_lr` | 0.005 | 尺度学习率 |
| `--rotation_lr` | 0.001 | 旋转学习率 |

物理参数(β/ρ/g/octave)的 LR 全程指数退火到 1/10。

#### 损失与正则

| 参数 | 默认 | 说明 |
|---|---|---|
| `--lambda_dssim` | 0.2 | DSSIM 损失权重(L = 0.8·L1 + 0.2·DSSIM) |
| `--lambda_scale` | 0.1 | 体积正则(∏s 均值),抑制高斯无界膨胀 |
| `--lambda_aniso` | 0.001 | 软各向异性正则(log-ratio 二次,超过 aniso_ratio_max 才罚)。调大伤 PSNR(0.05 → PSNR 崩到 ~25);硬约束交给针手术 |
| `--aniso_ratio_max` | 5.0 | 软正则的免罚阈值 |
| `--aniso_until_iter` | 30000 | 软正则作用区间。**必须全程**——aniso 不自收敛,提前关闭后 p99 单调上涨 |
| `--tonemap_lr` | 1e-3 | 可学习 tonemap 4 系数的学习率(仅 `--tonemap_learnable` 时生效;独立 Adam,衰到 0.1×) |
| `--lambda_tonemap_mono` | 1e-2 | 可学习 tonemap 单调性惩罚(仅 `--tonemap_learnable` 时;hinge 平方,保证曲线在 [0,8] 不反转,高光不倒挂) |

#### 致密化(densify)

| 参数 | 默认 | 说明 |
|---|---|---|
| `--densify_from_iter / _until_iter` | 500 / 15000 | 致密化区间。**15k 后留 settle 抛光期是实测最优**(densify 拉满 30k 反而 -0.2 dB 且 aniso 翻倍) |
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
| `--prune_min_visible_frames` | 5 | 至少在 N 帧可见才参与剪枝判定 |
| `--contribution_reset_interval` | 1000 | 贡献度累计器清零周期(保持统计反映当前模型) |
| `--resurrect_interval` | 3000 | 每 N 迭代把贡献度最低的一批 β_peak 重置回 0.1(替代 stock reset_opacity)。**仅 densify 期间生效**——settle 期运行会与剪枝形成净销毁回路(实测 -17% 点数、-0.7 dB) |
| `--resurrect_fraction` | 0.05 | 每次 resurrect 的点数占比 |
| `--post_densify_prune_interval` | 1000 | settle 期的剪枝周期;0 关闭 |

#### 针手术(结构性 aniso 硬上限)

| 参数 | 默认 | 说明 |
|---|---|---|
| `--needle_split_interval` | 1000 | 手术周期;**0 = 关闭** |
| `--needle_split_ratio` | 30.0 | 触发阈值(max/min 轴比)。每刀 ratio 减半,等效硬上限;想逼近体素量级(p99~12)可降到 15 |
| `--needle_split_until_iter` | 29000 | 最后一次手术的截止迭代(留收尾期让子高斯安定) |

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
# 分组评估:held-out 太阳组 vs 已见太阳新视角组(T_light 源自动从 cfg_args 读取)
python tools/eval_test_groups.py output/<run>
```

### 交互 Viewer

```shell
python viewer.py --ply output/<run>/point_cloud/iteration_30000/point_cloud.ply
```

基于 viser:实时改变太阳方向(relighting)、可视化通道(RGB / T_light / β_peak / depth)、可调背景色、snap 到训练相机。`--tlight auto|voxel|raster` 控制阴影源(auto 读训练 run 的 cfg_args)。

### 工具

```shell
tools/analyze_octave_weights.py    # 多次散射八度权重分析
tools/plot_phase_function.py       # 有效相函数重建
tools/project_pointcloud.py        # 初始点云-图像对齐快检
tools/residual_buckets.py          # 有符号残差分桶(GT 亮度 × 深度覆盖)+ held-out 太阳 PSNR
tools/penumbra_residual.py         # 残差按逐像素 T_light(阴影深度)分桶 × GT 亮度 cross-tab
```

## 当前状态与已知限制

- 均匀数据集上:test PSNR **~33.3**(固定 ACES tonemap,默认),**held-out 太阳与已见太阳零泛化差距**(物理参数化对新光照方向外推有效);aniso p99 ~22,popping 受控。
- **数据集是刻意 env-off 的控制变量设计**:UE 场景只有云 + 单方向太阳,背景纯黑,无天空/大气环境光。注意 env-off 控制掉的是**环境光**,但 UE 体积管线仍计算**云内多次散射**——模型的六阶 octave 正确学到了它(自阴影深核 ~78% 亮度来自多次散射,与 GT 匹配)。
- **残差诊断(tools/residual_buckets.py / penumbra_residual.py)**:深核阴影已标定准(残差 ~0),唯一可见残差是**近受光半影偏亮 +0.013**(仅 ~11% 像素,PSNR 上限 ~0.15 dB),且主要来自单次散射项 / HG 前向散射,octave 杠杆对其结构性无效——优先级低,暂不追。
- viewer 非黑背景下云边缘黑边:黑训练背景导致边缘透射率无监督的已知副作用,暂不修。

## 环境

原版 3DGS 的依赖基础上无新增 Python 包;CUDA 扩展(`submodules/diff-gaussian-rasterization`,含本项目的 analytic-tau / record_front_tau / lightpass-backward 通道)改动后需重新编译:

```shell
pip install ./submodules/diff-gaussian-rasterization
```

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
