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
tools/cloud_dataset_generator.py   # UE 编辑器内执行;MODE="uniform"(现行)/"tod"(历史弧线)
tools/convert_transforms.py        # UE 左手系 → OpenGL 右手系(含 sun_direction)
tools/split_test_set.py            # 分层 test split(幂等)
tools/thin_dataset.py              # (相机+时间) 轮转格子稀疏化
```

UE 内一行启动(自动关后台 CPU 节流——否则失焦时截图永不落盘):

```
py "D:/3DGS-Volume-Cloud/tools/cloud_dataset_generator.py"
```

## 使用

### 训练

```shell
# 默认:raster T_light + 完整几何梯度 + 针手术
python train.py -s data/CloudDatasetUniform

# 旧体素 T_light 路径
python train.py -s data/CloudDataset --tlight_voxel
```

eval 默认开启(test split 不并入训练),结束时在 test 集上输出 PSNR/SSIM/LPIPS 并写 `metrics.json`。PipelineParams 持久化进 cfg_args,供 viewer 自动匹配 T_light 源。

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
tools/compare_tlight_raster.py     # raster vs voxel T_light 分布对照
tools/check_lightpass_grad.py      # lightpass backward 有限差分验证
tools/analyze_octave_weights.py    # 多次散射八度权重分析
tools/plot_phase_function.py       # 有效相函数重建
tools/project_pointcloud.py        # 初始点云-图像对齐快检
tools/notify_lark.py               # 飞书通知(长训练挂机用)
```

## 当前状态与已知限制

- 均匀数据集上:test PSNR ~30.8,**held-out 太阳与已见太阳零泛化差距**(物理参数化对新光照方向外推有效);aniso p99 ~22,popping 受控。
- **缺环境光是当前主要模型偏差**:模型只有太阳单光源,实测呈系统性对比度压缩(暗部偏亮 +0.05 / 亮部偏暗 −0.05 的有符号残差)。环境光(已知天空 cubemap → 各向同性 ambient → SH)是下一步主线,预期同时缓解 aniso 压力源。
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
