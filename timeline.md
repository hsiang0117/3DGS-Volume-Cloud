# 排序工作 Timeline

记录针对 3DGS 体积云渲染中 alpha 合成排序问题的迭代过程。

---

## 2026-05-14 12:01 — 基线：裸版 3DGS 中心深度排序

`duplicateWithKeys` 用 `p_view.z`（Gaussian 中心视空间深度）作为排序键的低 32 位。所有覆盖该 Gaussian 的 tile 共享同一个 depth。

**问题**：
- viewer 旋转视角时大量长椭球 popping。
- iter 30000 PSNR **35.38**，比 iter 7000 (40.30) 显著回落——错误排序不仅是显示问题，还在训练中作为"无效噪声梯度"干扰收敛。

---

## 2026-05-14 13:28 — 方案 A：前缘深度排序

`forward.cu / preprocessCUDA`：把 `depths[idx]` 从 `p_view.z` 改成沿视向 1σ 前缘深度：

```
σ_d = sqrt(d̂ · Σ_v · d̂),   d̂ = μ_v / |μ_v|
depths[idx] = max(p_view.z − σ_d · d.z, 1e-4)
```

各向同性 Gaussian 行为基本不变，视向拉长椭球的前缘前移获得正确排序优先级。

**结果**：
- iter 30000 PSNR **43.31** (+8 dB)。
- 长椭球 popping 明显减弱，但仍可见。
- 视向以外被拉长的椭球这条 fix 解决不了（σ_d 退化）。

---

## 2026-05-14 15:12 → 17:50 — 辅助：anisotropy 正则（多次失败后落定）

承认 A 方案有死角，从另一头削减"长椭球的数量"。在 `train.py` 加正则。经过四次尝试：

| 时间 | 形式 | λ | 结果 |
|---|---|---|---|
| 15:12 | `(s_max/s_min − 3)²` | 0.01 | PSNR **20.5**（崩塌），但 aniso p99 = 3.4 |
| 16:16 | `(s_max − 5·s_min)²` (绝对差) | 0.001 | PSNR 43.0，但 aniso p99 = **20046**（正则形同虚设） |
| 17:11 | log-ratio | 0.05 | PSNR **24.9**（崩塌），aniso p99 = 32 |
| **17:50** | **log-ratio** | **0.001** + `aniso_until_iter=15000` | **PSNR 43.70**, aniso p99 = 556 |

关键认知：
- ratio 形式梯度 ∝ ratio²，必崩。
- 绝对差对 s_min→0 不敏感，正则无效。
- log-ratio 梯度只随 log 发散，配 λ=0.001 是甜区。
- densify 结束后必须关闭，否则和 L_vol 叠加把整个云缓慢压扁。

**收获**：意外发现"修排序 → 训练更稳"这条链路依然在起作用——正则版本比 A 版本再涨 0.4 dB，长椭球种群被压到 1/36。

---

## 2026-05-14 19:24 — 失败尝试：硬分裂长椭球

在 `densify_and_prune` 加 `densify_and_split_anisotropic`，对 ratio > 10 的 Gaussian 沿长轴分裂成两个子节点。

**结果**：点数爆炸。ratio=200 的 Gaussian 需要 7 次连续 split 才能掉到 10 以下 = 2^7 = 128 个子节点。子节点欠拟合 → grad-densify 长出更多长椭球 → 再 split → 级联爆炸。

**回退**：保留 working baseline。后来按要求**删除全部 split 相关代码**。

复盘：要让 split 稳定需要 N-way split（N ≈ ratio）或 split-frozen 标记。两者复杂度都太高，性价比不如方案 B。

---

## 2026-05-15 ~10:38 — 方案 B：per-tile max-response 排序

A 方案的根本限制是"一个 Gaussian 共用一个 depth"。**正确做法**是每个 `(Gaussian, tile)` 对单独算最大响应深度：

$$
t^{*} = \frac{v^{T}\, \Sigma_{v}^{-1}\, \mu_v}{v^{T}\, \Sigma_{v}^{-1}\, v}
$$

其中 `v = ((2px/W − 1)·tanfovx, (2py/H − 1)·tanfovy, 1)` 是穿过 tile 中心像素的视向。

**实现**：
- `preprocessCUDA` 预算 `Σ_v⁻¹`（3×3 余子式法）和 `q = Σ_v⁻¹ μ_v`，存到 `GeometryState` 新增的 `sigma_v_inv` (P×6) / `q_view` (P×3) 缓冲区。
- `duplicateWithKeys` 在 inner loop 内对每个 tile 求 `t*` 作为排序键。
- 反向传播不需要改：backward 按 forward 排好的 list 重新光栅。

**结果**：
- iter 30000 PSNR **43.60**（基本无损）。
- 所有方向的长椭球 popping 都消除。
- **新问题**：tile 边界出现块状伪影——相邻 tile 用不同 t*，长椭球在 Tile 1 排前、Tile 2 排后，边界两侧形成颜色断层。

**Commit**：`6580bfd` "add per-tile max-response sort and anisotropy regularizer"
**PR**：#9（已 merge）

---

## 2026-05-15 ~13:11 — 最终：σ_d clamp 调和

数学观察：t* 偏离中心的"物理合理范围"应由 Gaussian 自己沿视向的实际厚度 σ_d 决定。引入 K·σ 上限：

```cpp
float dev = t_star − centre_depth;
dev = clamp(dev, −K·σ_d, +K·σ_d);
depth_for_key = centre_depth + dev;
```

直觉：
- 小 Gaussian σ_d 几乎为 0 → dev ≈ 0 → 跨 tile 排序完全一致（stock 3DGS 行为）→ **tile 伪影消失**
- 长椭球 σ_d 大 → dev 允许偏几个真实单位 → **长轴 popping 仍消除**

K=1.5 是工程折中：≥1 才覆盖 σ 范围有意义；2~3 接近无 clamp 退化回 tile 伪影。

**端到端可调**：`k_sigma` 沿 `PipelineParams → GaussianRasterizationSettings → C++ → CUDA kernel` 全程透传。`viewer.py` 加 0~3 slider 实时调试。`k_sigma ≤ 0` short-circuit 到 stock 中心排序（便于 ablation）。

**结果**：
- iter 30000 PSNR **43.58**。
- 长椭球 popping 消除 + tile 伪影消除。
- viewer 残留少量小 Gaussian popping（与排序架构无关，是数值噪声问题）。

**Commit**：`664a22b` "clamp per-tile t* by ±k_sigma·σ, expose k_sigma to caller"
**PR**：#10（pending）

---

## 最终成绩对照

| 阶段 | PSNR @30K | 长椭球 popping | tile 伪影 |
|---|---|---|---|
| 裸版 | 35.38 | 严重 | 无 |
| 方案 A（前缘深度） | 43.31 | 减弱 | 无 |
| + aniso 正则（log-ratio λ=0.001） | 43.70 | 进一步减弱 | 无 |
| 方案 B（per-tile max-response，无 clamp） | 43.60 | 消除 | 出现 |
| **+ ±1.5σ_d clamp（最终）** | **43.58** | **消除** | **消除** |

## 关键认知

1. **排序错误是隐藏的训练发散源**。本以为只是显示问题，结果是 PSNR +8 dB 的隐藏天花板。
2. **"全局结构 + 局部修正"是正确范式**：per-tile 排序提供数学正确的局部自由度，σ_d clamp 把自由度限制在 Gaussian 自身物理尺度内。
3. **每步都需独立 ablation**：4 次正则、1 次 hard split、2 次排序架构都通过 PSNR + viewer 反馈独立验证。失败的尝试同样重要（hard split 爆炸 / ratio 形式崩溃 / 绝对差无效），它们排除了一整类不可行方向。
4. **K_SIGMA 必须可调**。1.5 是当前数据集的甜区，不同云形态可能需要 1.0 或 2.0；论文消融也需要这个旋钮。
5. **σ_d 是"免费"物理尺度**。`Σ_v` 反演时几乎不增加计算，但是连接"中心排序"和"per-tile 排序"两端的关键尺度。

## 尚未做的事

- **Per-pixel 排序**：技术上可行（StopThePop 路线），但帧时间 5× 升幅与"实时渲染"硬约束冲突。当前 k_sigma=1.5 的残留 popping 主要来自小 Gaussian 数值噪声，per-pixel 解决不了。优先级排在 HDR clamp、T_light 死亡柱、动态光源支持之后。

---

# Densify / Prune 分析

排序问题告一段落后，下一个瓶颈：**初始 200K 点训练 30K 后只剩 220K**，stock 3DGS 的 densify/prune 逻辑套在物理参数化上完全跑不动。

## 2026-05-15 14:41 — 加入 densify 诊断

每 500 步打印 `n_points / grad max-mean / n_above_thresh / clone-eligible / split-eligible / opacity min-median / n_below_prune`。跑一轮拿到完整诊断数据。

## 现有 densify_and_prune 逻辑

**入口** (`gaussian_model.py:549`)：用屏幕梯度 + 解析 opacity 两个信号驱动三条决策。

**决策 1：clone**（`densify_and_clone:511`）
- 触发：`grad ≥ 1e-4` 且 `scale_max ≤ percent_dense·extent` (默认 0.01)
- 逻辑：原地复制全部物理参数，scale 不变
- 直觉：小高斯欠表达 → 多放一份让它们各自漂移

**决策 2：split**（`densify_and_split:465`）
- 触发：`grad ≥ 1e-4` 且 `scale_max > percent_dense·extent`
- 逻辑：按协方差采 N=2 个新中心，scale 缩到 1/(0.8·N)，β_peak 全部继承（intensive 量），父节点 prune 掉
- 直觉：大高斯想动但动不了 → 拆分

**决策 3：prune**（`densify_and_prune:564`）
- 触发：`get_opacity < min_opacity` (默认 0.001)
- `get_opacity = 1 − exp(−β_peak · √(2π) · gscale)`，是物理参数派生量

## 不适配物理参数化的 7 个问题

### 1. opacity-based prune 是错的目标
- stock 的 sigmoid `_opacity` 是独立学的；你的是 `β_peak × gscale` 的派生。
- split 出的子节点 scale 减半 → opacity 减半 → **新生即贴近剪枝阈值**。
- log 实证：iter 6500-8500 期间 `n_below_prune` 1948→2758→3914 阶跃增长，正是中期 split 高峰之后的"保留筛选"在生效。
- 物理上 opacity 低 ≠ 无贡献：云里大量稀薄点叠加贡献环境光衰减，剪掉就"漂白"。

### 2. 屏幕梯度阈值在物理参数化下偏离原意
- alpha = 1−exp(−τ·G_2D) 比 stock 的 alpha = opacity·G_2D 弱一个 `(1−α)` 的链式抑制因子。
- 实测梯度衰减：iter 2000 grad max=3e-3、iter 9000 = 3e-4、iter 15000 = 1e-4。
- 固定阈值 1e-4：前期 densify 过激（前 2K 步净增 25K 点），后期 densify 完全停转（iter 15K 时仅 4 个点过线）。

### 3. 屏幕梯度选不出"真正该 densify"的位置
- stock 思路：屏幕梯度大 = 想动 = 该 densify。对**离散物体表面**重建是对的。
- 云是**连续介质**，"该 densify"的位置应是"loss 残留高、加点能让残差下降"——这是 per-Gaussian 对像素 loss 的贡献度，跟 xyz 屏幕梯度只有间接关系。
- 反例：体积内部被前后高斯遮挡的 Gaussian，xyz 移动几乎不影响最终图像，屏幕梯度天然小，但**这个位置可能正需要更细分**——触发不了 densify。

### 4. clone vs split 的二分法在物理介质下意义不大
- stock 设计：小→clone（重建表面细节），大→split（拆分大平面）。
- 云的高斯尺度连续分布、形状互相重叠。一个中等尺寸的 Gaussian 也许两条路都该走、也许两条都没必要。
- `percent_dense=0.01` 这个分界值是从场景重建任务调出来的，对体积云没有意义。

### 5. opacity_reset 完全失效
- `gaussian_model.py:319` 的 `reset_opacity()` 是空函数。
- stock 每 3000 步把 sigmoid `_opacity` 重置到 0.01，强迫所有点重新争梯度，**让预算流转**。
- 你的代码里这个机制彻底死掉。早期赢家（β_peak 最先涨起来的点）永久占据预算，后来欠拟合区域无法翻身。

### 6. scale_gradient_accum 路径单向
- `add_densification_stats:570` 只累计"让 scale 变大"方向的梯度（log-space 中负 grad → s 变大）。
- aniso 正则、L_vol 正则都在压制 scale 长大，这条触发途径长期被压低。`max_scale_grad=1e-6` 阈值常年达不到。

### 7. densification_interval 周期固定
- 每 100 步触发一次。前期 grad 量级高，100 步偏少；后期 grad 衰减到 1e-4，100 步根本攒不出多少有效梯度。
- 应按"梯度统计够不够稳定"动态决定周期。

## 实证：两次实验

### 2026-05-15 14:41 — `prune_min_opacity=0.001`（默认）

| iter | n_points | n_below_prune |
|---|---|---|
| 500 | 200,000 | 0 |
| 4500 | 328,146 (峰值) | 853 |
| 6500 | 325,012 | 1,948 |
| 8500 | 289,322 | 3,914 (峰值) |
| 13000 | 240,304 (谷底) | 493 |
| 15000 | 245,767 | 80 |

PSNR @ iter 7000：39.40。最终点数 **比初始增长 23%**。

### 2026-05-15 15:13 — `prune_min_opacity=0.0001`（× 0.1）

| iter | n_points | n_below_prune |
|---|---|---|
| 500 | 200,000 | 0 |
| 9000 | 350,293 (峰值) | 215 |
| 13000 | 296,409 | 4,316 (峰值) |
| 15000 | 259,650 | 1,874 |

PSNR @ iter 7000：40.87 (+0.9 dB)。最终点数 **比初始增长 30%**。

ROI 1（放宽 prune 阈值）只把"翻车点"从 iter 4500 推到 iter 9000，但**根本症结没解**：
- 中期 `n_below_prune` 仍能阶跃到 4316
- 后期 `n_above_thresh` 衰减到 < 0.05%，densify 停转
- prune 持续 + densify 停转 = 点数从峰值缓慢回落

## 综合后果

stock 用的两个信号（屏幕梯度 + opacity）在物理参数化下都不再贴合"该不该 densify / prune"的真实决策面。继续在原逻辑上加 trick（调阈值、加 grace period、改 percent_dense）只能把症状从一处推到另一处，无法根治。

## 下一步：重新设计

需要按物理介质语义重写一套 densify/prune。新决策面应基于：
- **per-Gaussian 图像贡献能量**（`Σ α·T`）替代 opacity 阈值做剪枝
- **自适应梯度阈值**（top-K%）替代固定值
- **contribution-driven clone** 替代 xyz 屏幕梯度（按"loss 残留高"densify）
- **resurrect 机制**（重置最弱 5% 的 β_peak）替代失效的 opacity_reset
- 增加**冗余点合并**（KNN 距离判断）

详细方案见对话讨论；代码改动量预估两步，每步 ~2 小时。

---

# Physical Densify / Prune 实现

## 2026-05-15 17:00 — 第一版：CUDA contribution + physical 策略

**思路**：用 per-Gaussian `Σ(α·T)`（图像贡献度）替代 opacity 阈值；自适应 grad threshold；周期性 β_peak resurrect 替代失效的 `opacity_reset`；split/clone 新生节点给 500 步 grace period。

**代码改动**：
- **CUDA**：`renderCUDA` 加 `gauss_contribution[P]` 输出（每像素在 alpha-blend 内 `atomicAdd(α·T)`）。`Rasterizer::forward` / Torch wrapper / Python autograd 透传。
- **GaussianModel**：新增 `contribution_accum/denom`、`prune_grace` buffer，同步在 `prune_points` / `densify_and_clone` / `densify_and_split` 维护。新增 `physical_densify_and_prune`（含 adaptive grad threshold + contribution prune + resurrect）。
- **train.py**：按 `opt.densify_strategy` 分流。

**结果**：PSNR @30K = **43.52**（持平 stock），点数 **493,999**（+147% vs 上一轮 +30%）。`n_below_contrib` 仅 586（~0.1% 的点真正符合剪枝），证明 contribution-based 路径有效区分弱点。

**问题**：`visible frames p50` 在 iter 15K 后冻结在 81——`add_contribution_stats` 被 `if iteration < densify_until_iter` 包住，stats 也跟着停。后半段 15K 步显示的是 frozen snapshot。

## 2026-05-15 18:14 — Fix A：解耦 contribution 累加器

**改动**：把 `add_contribution_stats` 调用从 densify guard 内挪到外，每步都执行。

**结果**：PSNR @30K = **43.14**（-0.4 dB），点数 **471,407**。`visible frames p50` 一路涨到 **10,781**（修复确认），但 PSNR 反而退步。

**根因**：第一版还有两个隐藏 bug：
- `contribution_accum` 只在 `physical_densify_and_prune` 内部 `zero_()`，densify_until_iter 后从未 reset。`mean_contrib` 变成"从训练开始到现在的累加平均"，被早期数据主导，失去判别力。
- `_resurrect_low_contribution` 也只在 densify 内调用——iter 15K 后预算流转完全停止，长椭球累计无法重置，aniso p99 从 545 涨到 874。

## 2026-05-15 18:58 — Fix B：把 resurrect 和 reset 完全解耦

**改动**：抽出独立 `tick_post_densify_maintenance(opt, iteration)`，每步调用：
- 每 `resurrect_interval=3000` 步执行 β_peak resurrect
- 每 `contribution_reset_interval=1000` 步清零累加器

跟 densify 解耦，全程执行。

**结果**：PSNR @30K = **44.20** ← **历史新高**，比之前最佳 (43.70) 再涨 0.50 dB。

`visible frames p50` 在 193 ↔ 385 之间周期跳变，证实 1000 步 reset 在工作。`contrib mean/median` 稳定在 0.48/0.005，反映当前模型状态。

**剩余问题**：iter 15K 后 densify 完全停（`n_above_thresh=5`），但 grad + resurrect 持续拉伸已存在 Gaussian。aniso p99 从 545 涨到 815，viewer 长椭球 popping 明显加重。

## 2026-05-15 19:33 — Fix C：aniso side-channel prune（待验证）

**思路**：不应该回到全程 aniso 软正则（上次实测会和 L_vol 一起把云压扁）。**应把 anisotropy 接进 prune 决策**——长椭球判定为"对图像贡献模式不健康"，被优先回收。

**改动**：抽出 `_prune_by_contribution_and_aniso(opt)`：

```python
prune_mask = visible_enough & grace_expired & (below_contrib | aniso_too_long)
```

`physical_densify_and_prune` 和 `tick_post_densify_maintenance` 都调用它。后者每 `post_densify_prune_interval=1000` 步触发一次，让后半段 15K 步也持续清理长椭球。

**默认超参**：
- `prune_aniso_ratio = 100`（按 p99 估计剪掉最坏 1~3%）
- `post_densify_prune_interval = 1000`

预期 PSNR ≥43.5、aniso p99 ≤200、viewer 长椭球 popping 基本消除。

## 失败回退记录

| 时间 | 尝试 | 失败原因 |
|---|---|---|
| 14:41 | 加 densify diag log | （成功，但揭示根本问题） |
| 15:13 | `prune_min_opacity=1e-4` 单一调参 | 把翻车点从 iter 4500 推到 9000，没解根本症结 |
| —— | 全程 aniso 软正则 (老结果) | 和 L_vol 联手把云压扁，PSNR 崩到 20.5 |

## 2026-05-15 19:48 — Fix C 验证：aniso side-channel 同时提升 PSNR

跑 30K iter，`prune_aniso_ratio=100`、`post_densify_prune_interval=1000`。

**结果**：

| 指标 | Fix B（44.20 base） | Fix C（aniso prune） |
|---|---|---|
| **PSNR @ 30K** | 44.20 | **45.11** ↑ +0.91 dB |
| 总点数 @ 30K | 472,323 | 479,777 ≈ 持平 |
| aniso p99 @ 30K | 815 | **510** ↓ -37% |
| aniso mean @ 30K | 74 | 47 ↓ -36% |

**反直觉但物理自洽的发现**：剪掉长椭球同时**改善了 PSNR**，不是损失。说明长椭球本身在伤害训练——它们用扁长形状"侥幸"贴近 alpha 合成结果，但占用了应该被更紧凑 Gaussian 占据的预算位置。aniso prune 削掉后，resurrect 重新初始化的点接管这些位置，每个点都更适合自己负责的区域。

形状不合理的点对训练有害，不仅是显示问题。

## 总成绩

```
裸版 baseline:        35.38 PSNR
sort fix:             43.31
+ aniso reg:          43.70
+ per-tile sort:      43.60
+ σ_d clamp:          43.58  (popping 解决路径完结)

物理 densify v1:      43.52  (有 bug)
+ contribution 解耦:   43.14  (暴露更多 bug)
+ resurrect 解耦:      44.20  (历史新高 1)
+ aniso prune:        45.11  (历史新高 2)
```

**总提升：35.38 → 45.11，+9.73 dB。**

## 现状
- contribution-based prune ✓
- adaptive grad threshold ✓
- β_peak resurrect ✓
- new-point grace period (500 steps) ✓
- aniso side-channel prune ✓（PSNR +0.91 dB, aniso p99 -37%）
- 冗余点合并（KNN 距离判断）——未实现，留待第二步
- contribution-driven clone（基于 loss 残留位置）——未实现，留待第三步


