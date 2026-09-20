# Light Pass 分类缺陷修复报告

针对提交 `1941c29 fix(rasterizer): separate "too faint" from "buried" in the light pass` 中发现的两个分类缺陷的修复记录。

---

## 一、背景：被修复提交的两个缺陷

`1941c29` 在内核中增加了两个探针：`tau_front_touch`（每 Gaussian 计数到达过它的像素）与 `ray_cut`（早退像素标志），宿主端 `classify_T_light` 用 `buried = (~covered) & (radii>0) & touched & any_cut` 分类，其中 `any_cut = (ray_cut > 0).any()` 为**全图标量**。

在 RTX 5060 上用合成场景（v_l=+Y 太阳在上方，512×512 光影图）实测：

| 场景 | 提交意图 | 实测结果 | 旧代码（1941c29 之前） |
|---|---|---|---|
| S1: too-faint Gaussian 单独存在（tau=0.001 < 1/255） | 1.0 | 1.0 ✓ | 1e-4 |
| S2: victim 位于厚遮挡堆之后（前方 OD≈30，覆盖像素全部在到达它之前终止） | 1e-4（"-> darken"） | **1.0** ✗ 回归 | 1e-4 |
| S3: too-faint Gaussian + 约 200px 外无关的厚遮挡堆（存在任何早退光线） | 1.0（"-> stay lit"） | **1e-4** ✗ 未修复 | 1e-4 |
| S4: 对照组，前方 OD=2 正常覆盖 | ≈exp(-2)=0.135 | 0.136 ✓ | 同 |

### 缺陷 1（回归）：被埋 Gaussian 误判为全亮

光线在到达 G **之前**终止时，像素退出 j 循环（`!done` 门控），`touch(G)==0`，与"被剔除/亚像素"不可区分，落入 "otherwise → 1.0" 分支。提交信息声称的 case (A)（"every covering pixel terminated early → darken"）根本没被实现——真值表中间行要求 `touched`，而 case (A) 的 Gaussian 恰恰 `touch==0`。

影响：T_light 直接进入 `Lk = ω·L_light·Σ w·T_light^{0.5ⁿ}·HG`，被完全遮挡的 Gaussian（如厚云底部）以 T_light=1 全亮参与散射，云底自阴影消失/发亮，最多差 4 个数量级。

### 缺陷 2（未修复）：too-faint 仅在"全图无任何早退光线"时才修复

`any_cut` 是全局归约，任何无关像素的终止都会把所有 too-faint Gaussian 误判为 buried。厚云场景（光影 pass 最需要的场景）几乎必然存在早退光线（沿光路 OD > 9.2 即触发 `test_T < 1e-4`），所以提交宣称修复的 case (B) 在实际训练中依然是错的。

### 附带发现

- touch 探针注释称"inside the ellipse"，但 `power > 0` 对 PSD conic 恒为假，实际统计的是 tile 内全部存活像素（S1 中 radii=240 的 Gaussian touch=246016 ≈ 全图）。
- `any_cut` 的 `.item()` 每次光影 pass 触发一次设备同步。

---

## 二、修复策略

| 缺陷 | 根因 | 策略 |
|---|---|---|
| (A) 被埋误判全亮 | `touch==0` 与"被剔除"不可区分 | **无需新探针**：`radii>0` 保证 G 进入 ≥1 个 tile 列表；tile 内存活像素必遍历整个列表 ⇒ `G_sum==0 & radii>0` 唯一地意味着"所有覆盖像素都死在它前面"，直接映射回 1e-4 |
| (B) any_cut 全局污染 | 布尔猜测定性 | **把"猜"换成"测"**：touch 点的像素 `T` 恰是该 Gaussian 的前方透射率真值。累加 `Σ(T·G)` 与 `ΣG`，`T_light = ΣTG/ΣG` 为足迹加权实测值，faint / 终止于该 splat / 混合三种子 case 统一解决；`ray_cut` 与全局 `any()` 整体删除 |

核心改动：**两个 int 探针 `(tau_front_touch, ray_cut)` → 两个 float 累加器 `(tau_front_TG_sum, tau_front_G_sum)`**。C++ 返回元组 arity 不变（12），相机 pass 的 `*_rest` 解包无需改动。

---

## 三、逐文件修改（7 个文件，+90/−78）

### 1. `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`

- `renderCUDA` 签名：`int32_t* tau_front_touch, int32_t* ray_cut` → `float* tau_front_TG_sum, float* tau_front_G_sum`（forward.cu:295-296）
- 探针块（forward.cu:370-387）：`const float G = exp(power)` 上移，探针在 alpha 门控**之前**执行：

  ```cpp
  if (tau_front_TG_sum && G > 0.0f)
  {
      atomicAdd(&tau_front_TG_sum[collected_id[j]], T * G);
      atomicAdd(&tau_front_G_sum[collected_id[j]], G);
  }
  ```

  `G > 0.0f` 仅排除 exp 下溢像素（≥13σ，远在 3σ 足迹之外）——它们本不构成几何覆盖，排除后 `G_sum==0` 的语义更纯。
- 早退处删除 `ray_cut[pix_id] = 1` 写入（forward.cu:406-410），`done=true` 早退逻辑本身不变。
- `FORWARD::render` 包装同步改名（forward.cu:478-481, 499-502）。

### 2. `cuda_rasterizer/forward.h`

`render` 声明改为两个 `float*`，注释更新为新语义（forward.h:72-77）。

### 3. `cuda_rasterizer/rasterizer.h`

`Rasterizer::forward` 声明同步（rasterizer.h:59-62）。参数仍在 `antialiasing` 之后、默认参数之前。

### 4. `cuda_rasterizer/rasterizer_impl.cu`

实现签名（:229-230）与内核调用实参（:357-358）同步。

### 5. `rasterize_points.cu`

- 分配（:91-94）：删除 `int_opts_p`、`int32[P] tau_front_touch`、`int32[H·W] ray_cut` → 两个 `torch::zeros({do_front_tau ? P : 0}, float_opts)`。**H·W 缓冲消失**。
- `::forward` 传参与返回元组同步（:146-147, :151）。

### 6. `diff_gaussian_rasterization/__init__.py`

- `_RasterizeLightpass.forward`：解包 / `mark_non_differentiable` / 返回值改为 `(tau_light_sum, tau_light_wsum, radii, tau_front_TG_sum, tau_front_G_sum)`；`backward` 形参同步改名（arity 均不变）。
- `rasterize_lightpass` docstring 更新。
- 相机 pass 的 `*_rest` 尾部解包不变。

### 7. `gaussian_renderer/__init__.py`

`classify_T_light` 重写（:245-275）：

```python
def classify_T_light(covered, T_light, radii, front_TG_sum, front_G_sum,
                     tau_occluded=1e-4, g_eps=1e-12):
    covered = covered.bool()
    reached = front_G_sum > g_eps                        # 有存活像素到达过该 splat
    T_measured = (front_TG_sum / front_G_sum.clamp(min=g_eps)).clamp(0.0, 1.0)
    buried = (~covered) & (~reached) & (radii > 0)       # 覆盖像素全部死在它前面
    return torch.where(
        covered,
        T_light,
        torch.where(reached, T_measured,
                    torch.where(buried, torch.full_like(T_light, tau_occluded),
                                torch.ones_like(T_light))),
    )
```

新真值表：

| 条件 | T_light | 依据 |
|---|---|---|
| `covered`（wsum>0） | `exp(-sum/wsum)` | 不变（可微路径） |
| `~covered & G_sum>0` | **实测 `ΣTG/ΣG`** | 门控前测得的前方透射率真值 |
| `~covered & G_sum==0 & radii>0` | `1e-4` | **修复缺陷 1**：精确判据，恢复旧行为 |
| `radii==0`（剔除/视锥外） | `1.0` | 不变 |

---

## 四、正确性依据（内核不变量）

1. `radii>0` ⇒ 通过视锥剔除且 3σ 矩形裁剪后非空 ⇒ 进入 ≥1 个 tile 列表（preprocessCUDA，forward.cu:186-192, 241-246：空矩形/视锥外都会提前 return 且 `radii[idx]=0`）。
2. tile 内每个存活像素遍历完整 tile 列表，唯一出口是 alpha 门控、早退、列表结束（`power>0` 对 PSD conic 恒假，不构成出口）。
3. 死光线永远到不了探针（`!done` 门控）⇒ `G_sum==0`（radii>0 时）只能是"处处被截断"。
4. 列表按深度全局排序，早退发生在严格更靠前的 splat 上 ⇒ 被截断 ⇒ 真实遮挡。
5. `ΣTG/ΣG` 的误差界与 `tau_accum` 相同（被门控丢弃的前方 splat 每个代价 ≤0.00392，即 ≤0.39%）；终止于该 splat 的像素在早退**之前**已被计入，其 T 即真值——比旧硬编码 1e-4 保真度更高。

**已知限制**（与 covered 分支共有，非回归）：死在 splat 前的光线贡献为零而非 T≈0，半遮挡 Gaussian 只按未遮挡足迹取平均——这是前向早退设计的固有语义，covered 分支的 wsum 加权同此。

---

## 五、未改动项

- 可微路径：`tau_front_sum/wsum` 累加位置、`lightpassBackward`、autograd 结构全部不变；`T_measured` 为不可微常量（与旧 1e-4/1.0 同类）。
- 相机 pass：`do_front_tau=false` → 两指针为 nullptr → 探针跳过，零开销、行为不变。
- 附加收益：删除每帧 `.item()` 设备同步；删除 H·W int32 分配。

---

## 六、验证

已完成：

- grep 确认全仓无 `tau_front_touch | ray_cut | any_cut` 残留；
- `py_compile` 通过（两个 Python 文件）；
- NVCC 编译需手动完成（见下）。

重建命令：

```
.venv\Scripts\pip.exe install --force-reinstall --no-deps --no-build-isolation .\submodules\diff-gaussian-rasterization
```

> 注：自动尝试构建时报 `crt/host_config.h(170): fatal error C1189`，为 MSVC 版本与 CUDA 12.8 不匹配的工具链环境问题，与本次改动无关。

重建后用合成场景复测，预期：

| 场景 | 期望 T_light | 对应 |
|---|---|---|
| S1: too-faint 单独 | 1.0 | 与 1941c29 相同 |
| S2: 被埋在厚遮挡后（覆盖像素全部提前终止） | **1e-4** | 缺陷 1 修复（原 1.0） |
| S3: too-faint + 远处无关 cut | **≈1.0** | 缺陷 2 修复（原 1e-4） |
| S4: covered 对照 | ≈0.136 | 不变 |
| S5（建议加测）: 前方 OD≈6、光线终止于该 splat | ≈exp(-6)≈0.0025 | 比旧硬编码 1e-4 保真度更高 |
| S6（建议加测）: too-faint + 部分遮挡 | (0, 1) 内连续值 | 实测路径工作正常 |
