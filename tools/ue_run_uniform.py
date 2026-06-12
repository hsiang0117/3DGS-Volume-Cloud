"""均匀太阳数据集采集启动脚本 —— 在 UE 编辑器内手动执行。

执行方式(任选其一):
  A. UE 底部 Output Log 的命令行,左侧下拉切到 "Cmd" 模式,输入:
         py "D:/3DGS-Volume-Cloud/tools/ue_run_uniform.py"
  B. 菜单 Tools -> Execute Python Script... 选中本文件。

行为:
  - 关闭后台 CPU 节流(否则编辑器失焦时截图永不落盘,这是上次踩过的坑);
  - 启动 60 个 Fibonacci 半球均匀太阳 x 轮转 1/3 相机 ~= 1460 帧的异步采集;
  - 输出到 D:/CloudDatasetUniform(全新目录,不触碰原 D:/CloudDataset);
  - 进度在 Output Log 里实时打印,完成时打印 "数据集生成完成" 并写出
    transforms.json(它只在全部完成后才写,是可靠的完成信号)。

采集期间编辑器可以最小化,但不要关闭。预计 40-60 分钟。
中途想停止,在同一控制台执行:
    py -c "import cloud_dataset_generator as g; g.ACTIVE_GENERATOR.capture_queue=[]; g.ACTIVE_GENERATOR._clear_pending_capture(); g.ACTIVE_GENERATOR.stop_capture_loop()"
"""
import sys
import importlib

import unreal

# 1) 关闭后台 CPU 节流
try:
    eps = unreal.load_object(None, "/Script/UnrealEd.Default__EditorPerformanceSettings")
    eps.set_editor_property("throttle_cpu_when_not_foreground", False)
    unreal.log("[uniform] CPU throttle (not-foreground) -> OFF")
except Exception as e:  # 属性名/类暴露随版本变化,失败则提醒保持前台
    unreal.log_warning(f"[uniform] 关闭 CPU 节流失败: {e} —— 采集期间请保持编辑器前台")

# 2) 启动采集
p = r"D:/3DGS-Volume-Cloud"
if p not in sys.path:
    sys.path.insert(0, p)
import cloud_dataset_generator
importlib.reload(cloud_dataset_generator)
cloud_dataset_generator.main_uniform()   # -> D:/CloudDatasetUniform
