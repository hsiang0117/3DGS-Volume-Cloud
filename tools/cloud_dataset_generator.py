"""
UE5 体积云数据集采集脚本(在 UE 编辑器内执行)。

均匀太阳数据集:60 个 Fibonacci 均匀半球太阳 × 轮转 1/3 相机 ≈ 1458 帧
(方向均匀覆盖,消除方向有偏导致的 aniso 逃逸)。输出目录见文件底部 OUTPUT_DIR。

执行方式(任选其一):
  A. UE 底部 Output Log 命令行,左侧下拉切到 "Cmd" 模式,输入:
         py "D:/3DGS-Volume-Cloud/tools/cloud_dataset_generator.py" -o D:/CloudDatasetUniform_envon
         (-o 指定输出目录;省略则用底部 DEFAULT_OUTPUT_DIR;env-off/env-on 必须用不同 -o)
  B. 菜单 Tools -> Execute Python Script... 选中本文件(走 DEFAULT_OUTPUT_DIR)。
  C. UE Python 控制台按需调用:
         import sys; sys.path.insert(0, r"D:/3DGS-Volume-Cloud/tools")
         import cloud_dataset_generator as g
         g.main_uniform("D:/CloudDatasetUniform_envon")

注意事项:
  - 脚本会自动关闭"后台 CPU 节流"(否则编辑器失焦时截图永不落盘);
  - 采集期间编辑器可最小化但不要关闭;进度实时打印在 Output Log;
  - 完成信号:输出目录写出 transforms.json(只在全部完成后才写);
  - 之后用 tools/convert_transforms.py 把 transforms.json 转成 OpenGL 训练格式;
  - 中途停止,在控制台执行:
        py -c "import cloud_dataset_generator as g; g.ACTIVE_GENERATOR.capture_queue=[]; g.ACTIVE_GENERATOR._clear_pending_capture(); g.ACTIVE_GENERATOR.stop_capture_loop()"
"""

import unreal
import math
import json
import time
from pathlib import Path


def disable_background_throttle():
    """关闭编辑器后台 CPU 节流——失焦时截图任务会永久挂起(踩过的坑)。"""
    try:
        eps = unreal.load_object(None, "/Script/UnrealEd.Default__EditorPerformanceSettings")
        eps.set_editor_property("throttle_cpu_when_not_foreground", False)
        unreal.log("[capture] CPU throttle (not-foreground) -> OFF")
    except Exception as e:
        unreal.log_warning(f"[capture] 关闭 CPU 节流失败: {e} —— 采集期间请保持编辑器前台")


class CloudDatasetGenerator:
    def __init__(self, output_dir, vdb_file_path="D:/dataset/wdas_cloud_quarter.vdb"):
        """
        初始化数据集生成器

        Args:
            output_dir: 输出目录路径
            vdb_file_path: VDB 文件的完整路径
        """
        self.output_dir = output_dir
        self.vdb_file_path = vdb_file_path
        self.resolution = (1024, 1024)

        # 采样参数
        self.zenith_angles = [0, 20, 40, 60, 80, 100, 135]  # 天顶角
        self.azimuth_interval = 30  # 方位角间隔

        # 创建输出目录结构
        self.setup_output_directories()

        # UE5 系统
        self.editor_level_lib = unreal.EditorLevelLibrary()
        self.editor_actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        self.asset_tools = unreal.AssetToolsHelpers.get_asset_tools()

        # 相机参数
        self.camera_distance = 4000  # 相机到云中心的距离(cm)
        self.cloud_center = unreal.Vector(0, 0, 0)  # 云体几何中心

        # time_idx -> 指向太阳的 UE 方向向量(每个 time_idx 一个显式太阳方向)。
        self.sun_override_by_time = {}

        # 存储相机位姿数据
        self.transforms_data = {
            "camera_angle_x": math.radians(90),  # FOV
            "frames": []
        }

        # 截图任务状态（异步串行）
        self.spawned_cameras = []          # every temp CameraActor we spawn (for final sweep)
        self.capture_task = None
        self.pending_capture_meta = None
        self.capture_queue = []
        self.current_time_idx = -1
        self.tick_handle = None
        self.total_camera_count = 0
        # Max wait for a HighResShot PNG to land before retrying. Generous so
        # cold-start frames (shader/PSO compile, volumetric first-frame cost) are
        # not falsely timed out.
        self.capture_file_timeout_seconds = 20.0
        self.max_retries = 10              # timed-out frames requeue (to tail) up to this many times
        self.expected_frames = 0           # set when the queue is built; checked at completion
        self.warmup_done = 0               # discarded warmup shots taken so far
        self.warmup_target = 3             # warm the render pipeline before the first real frame
        self.warmup_image_path = str(Path(self.output_dir) / "_warmup_screenshot.png")

    def setup_output_directories(self):
        """创建输出目录结构"""
        base_path = Path(self.output_dir)
        base_path.mkdir(parents=True, exist_ok=True)

        # 为每个相机创建目录
        camera_count = self.calculate_camera_count()
        for cam_idx in range(camera_count):
            cam_dir = base_path / f"cam{cam_idx:02d}"
            (cam_dir / "images").mkdir(parents=True, exist_ok=True)

        unreal.log(f"输出目录已创建: {self.output_dir}")

    def calculate_camera_count(self):
        """计算相机总数"""
        count = 1  # 顶部中心1个
        for zenith in self.zenith_angles[1:]:
            azimuth_count = 360 // self.azimuth_interval
            count += azimuth_count
        return count

    def calculate_camera_positions(self):
        """
        计算所有相机位置(分层均匀半球采样)

        Returns:
            list: [(位置, 旋转, 相机索引), ...]
        """
        positions = []
        cam_idx = 0

        # 顶部中心相机 (天顶角0度)
        top_position = unreal.Vector(
            self.cloud_center.x,
            self.cloud_center.y,
            self.cloud_center.z + self.camera_distance
        )
        top_rotation = unreal.Rotator(0, -90, 0)  # 向下看
        positions.append((top_position, top_rotation, cam_idx))
        cam_idx += 1

        # 其他天顶角的相机
        for zenith in self.zenith_angles[1:]:
            azimuth_count = 360 // self.azimuth_interval

            for i in range(azimuth_count):
                azimuth = i * self.azimuth_interval

                # 球面坐标转笛卡尔坐标
                zenith_rad = math.radians(zenith)
                azimuth_rad = math.radians(azimuth)

                x = self.camera_distance * math.sin(zenith_rad) * math.cos(azimuth_rad)
                y = self.camera_distance * math.sin(zenith_rad) * math.sin(azimuth_rad)
                z = self.camera_distance * math.cos(zenith_rad)

                position = unreal.Vector(
                    self.cloud_center.x + x,
                    self.cloud_center.y + y,
                    self.cloud_center.z + z
                )

                # 计算相机旋转(看向云中心)
                rotation = unreal.MathLibrary.find_look_at_rotation(position, self.cloud_center)

                positions.append((position, rotation, cam_idx))
                cam_idx += 1

        unreal.log(f"计算得到 {len(positions)} 个相机位置")
        return positions

    def setup_scene(self):
        """设置场景(加载VDB云模型、配置光照等)"""
        unreal.log("开始设置场景...")

        # 1. 查找体积云Actor
        self.cloud_actor = self.find_vdb_cloud()

        # 2. 设置 Sky Atmosphere
        self.sky_atmosphere = self.setup_sky_atmosphere()

        # 3. 设置 Sky Light
        self.sky_light = self.setup_sky_light()

        # 4. 设置 Directional Light (太阳)
        self.sun_light = self.setup_directional_light()

        unreal.log("场景设置完成")

    def find_vdb_cloud(self):
        """查找VDB云模型"""
        # 这里需要您手动将 wdas_cloud.vdb 导入到 UE5 项目中
        # 脚本将尝试查找已存在的云 Actor

        cloud_actor = None
        all_actors = self.editor_actor_subsystem.get_all_level_actors()

        # 查找包含 "cloud" 或 "wdas" 的 Actor
        for actor in all_actors:
            actor_name = actor.get_name()
            if "HeterogeneousVolume" in actor_name:
                cloud_actor = actor
                unreal.log(f"找到云模型 Actor: {actor.get_name()}")
                break

        if not cloud_actor:
            unreal.log_warning("未找到云模型 Actor,请确保已放置体积云Actor")

        return cloud_actor

    def setup_sky_atmosphere(self):
        """设置 Sky Atmosphere 组件"""
        sky_atm = None
        all_actors = self.editor_actor_subsystem.get_all_level_actors()

        for actor in all_actors:
            if actor.get_class().get_name() == "SkyAtmosphere":
                sky_atm = actor
                break

        unreal.log("已找到现有 Sky Atmosphere")

        return sky_atm

    def setup_sky_light(self):
        """设置 Sky Light 组件"""
        sky_light = None
        all_actors = self.editor_actor_subsystem.get_all_level_actors()

        for actor in all_actors:
            if actor.get_class().get_name() == "SkyLight":
                sky_light = actor
                break

        unreal.log("已找到现有 Sky Light")

        # 设置为可移动,以便实时更新
        light_component = sky_light.get_component_by_class(unreal.SkyLightComponent)
        if light_component:
            light_component.set_mobility(unreal.ComponentMobility.MOVABLE)
            light_component.set_editor_property("real_time_capture", True)

        return sky_light

    def setup_directional_light(self):
        """设置 Directional Light (太阳)"""
        sun_light = None
        all_actors = self.editor_actor_subsystem.get_all_level_actors()

        for actor in all_actors:
            if actor.get_class().get_name() == "DirectionalLight":
                sun_light = actor
                break

        unreal.log("已找到现有 Directional Light")

        # 设置为可移动
        light_component = sun_light.get_component_by_class(unreal.DirectionalLightComponent)
        if light_component:
            light_component.set_mobility(unreal.ComponentMobility.MOVABLE)
            light_component.set_editor_property("atmosphere_sun_light", True)

        return sun_light

    def set_sun_direction(self, d_toward_sun):
        """
        直接设置太阳方向(均匀太阳模式)。

        Args:
            d_toward_sun: 指向太阳的 UE 世界系单位向量 [x, y, z]
        光线传播方向 forward = -d。UE forward 由 (pitch, yaw) 给出:
            forward = (cosP·cosY, cosP·sinY, sinP)
        故 pitch = asin(-d_z), yaw = atan2(-d_y, -d_x)。roll 不影响方向光。
        """
        if not self.sun_light:
            return

        fx, fy, fz = -d_toward_sun[0], -d_toward_sun[1], -d_toward_sun[2]
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, fz))))
        yaw = math.degrees(math.atan2(fy, fx))
        self.sun_light.set_actor_rotation(unreal.Rotator(0.0, pitch, yaw), False)

        if self.sky_light:
            light_component = self.sky_light.get_component_by_class(unreal.SkyLightComponent)
            if light_component:
                light_component.recapture_sky()

        unreal.log(
            f"设置太阳方向: [{d_toward_sun[0]:.3f}, {d_toward_sun[1]:.3f}, {d_toward_sun[2]:.3f}] "
            f"(pitch={pitch:.1f}°, yaw={yaw:.1f}°)"
        )

    def get_sun_direction(self):
        """
        获取太阳在 UE5 世界坐标系下的归一化方向向量
        约定：返回"指向太阳"的方向（即光线的反方向）

        UE5 坐标系: X=forward, Y=right, Z=up (左手系)
        DirectionalLight 的 forward 向量代表光线传播方向，
        因此"指向太阳"的方向为 -forward。
        """
        if not self.sun_light:
            return [0.0, 0.0, 1.0]

        sun_rotation = self.sun_light.get_actor_rotation()
        forward = unreal.MathLibrary.get_forward_vector(sun_rotation)

        # 取反，得到"指向太阳"的方向
        dx, dy, dz = -forward.x, -forward.y, -forward.z

        # 归一化（理论上 forward 已是单位向量，这里再保证一次）
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length > 1e-8:
            dx /= length
            dy /= length
            dz /= length

        return [dx, dy, dz]

    def capture_camera_view(self, camera_position, camera_rotation, cam_idx, time_idx, retry=0, image_path_override=None, is_warmup=False):
        """
        从指定相机位置和旋转捕获图像

        Args:
            camera_position: 相机位置
            camera_rotation: 相机旋转
            cam_idx: 相机索引
            time_idx: 时间帧索引
        """
        # 每张图创建一个临时相机，截图任务完成后销毁；同时登记到 spawned_cameras
        # 供结束时兜底清扫（防止超时路径销毁失败留下的残留相机）。
        camera = self.editor_level_lib.spawn_actor_from_class(
            unreal.CameraActor,
            camera_position,
            camera_rotation
        )
        self.spawned_cameras.append(camera)

        camera_component = camera.get_editor_property("camera_component")
        if camera_component:
            camera_component.set_field_of_view(90.0)
            camera_component.set_aspect_ratio(1.0)

        # 输出路径
        cam_dir = Path(self.output_dir) / f"cam{cam_idx:02d}"
        image_path = image_path_override if image_path_override else str(cam_dir / "images" / f"{time_idx:04d}.png")

        # 发起异步截图任务（必须串行）
        self.capture_task = self.render_image(camera, image_path)
        self.pending_capture_meta = {
            "camera": camera,
            "camera_position": camera_position,
            "camera_rotation": camera_rotation,
            "cam_idx": cam_idx,
            "time_idx": time_idx,
            "image_path": image_path,
            "retry": retry,
            "start_time": time.time(),
            "is_warmup": is_warmup,
        }

    def render_image(self, camera, output_path):
        """渲染RGB图像"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 清理同名旧文件，避免误判本次截图已落盘
        if output_file.exists():
            try:
                output_file.unlink()
            except Exception:
                pass

        # 使用 High Resolution Screenshot 渲染（返回 AutomationEditorTask）
        return unreal.AutomationLibrary.take_high_res_screenshot(
            self.resolution[0],
            self.resolution[1],
            output_path,
            camera,
            False,
            False,
            unreal.ComparisonTolerance.LOW,
            "dataset_capture",
            0.1
        )

    def _destroy_camera(self, camera):
        """Destroy one temp camera. If destroy fails (its async screenshot task
        may still hold it, e.g. on the timeout path), keep it tracked so the
        final sweep retries once the task has released it."""
        if camera is None:
            return
        ok = False
        try:
            ok = bool(self.editor_actor_subsystem.destroy_actor(camera))
        except Exception:
            ok = False
        if ok and camera in self.spawned_cameras:
            self.spawned_cameras.remove(camera)

    def cleanup_all_cameras(self):
        """Final sweep: destroy every temp camera still alive. Runs at completion
        when no screenshot task is in flight, so destroys that failed mid-task
        (timeout path) succeed here -> no CameraActor remnants left in the scene."""
        survivors = list(self.spawned_cameras)
        for cam in survivors:
            try:
                self.editor_actor_subsystem.destroy_actor(cam)
            except Exception:
                pass
        self.spawned_cameras = []
        if survivors:
            unreal.log(f"[capture] 清理残留临时相机 {len(survivors)} 个")

    def _is_current_task_done(self):
        if not self.capture_task:
            return True

        try:
            return self.capture_task.is_task_done()
        except Exception:
            return True

    def _is_capture_file_ready(self, image_path):
        """检查截图文件是否已落盘且非空"""
        try:
            output_file = Path(image_path)
            return output_file.exists() and output_file.stat().st_size > 0
        except Exception:
            return False

    def _clear_pending_capture(self):
        """清理当前待处理截图状态"""
        if self.pending_capture_meta and self.pending_capture_meta.get("camera"):
            self._destroy_camera(self.pending_capture_meta["camera"])

        self.capture_task = None
        self.pending_capture_meta = None

    def _finalize_pending_capture(self):
        if not self.pending_capture_meta:
            return

        meta = self.pending_capture_meta

        if meta.get("is_warmup", False):
            # 丢弃预热截图，仅用于让截图系统稳定绑定相机
            try:
                warmup_file = Path(meta["image_path"])
                if warmup_file.exists():
                    warmup_file.unlink()
            except Exception:
                pass
            unreal.log("截图预热完成")
            self._clear_pending_capture()
            return

        # 保存相机位姿信息
        sun_direction_ue = self.get_sun_direction()
        self.add_transform_data(
            meta["camera_position"],
            meta["camera_rotation"],
            meta["cam_idx"],
            meta["time_idx"],
            sun_direction_ue
        )

        unreal.log(f"已捕获: cam{meta['cam_idx']:02d}, frame {meta['time_idx']:04d}")
        self._clear_pending_capture()

    def _on_capture_tick(self, _delta_seconds):
        # 若有待处理截图，必须先等该目标文件落盘，绝不启动下一张
        if self.pending_capture_meta:
            meta = self.pending_capture_meta
            if self._is_capture_file_ready(meta["image_path"]):
                self._finalize_pending_capture()
            else:
                elapsed = time.time() - float(meta.get("start_time", time.time()))
                if elapsed < self.capture_file_timeout_seconds:
                    return

                if meta.get("is_warmup", False):
                    # Warmup timed out (coldest first frame). Drop it — do NOT
                    # requeue as a real frame (would duplicate the held job).
                    unreal.log_warning("预热截图超时，跳过本次预热")
                    self._clear_pending_capture()
                    return

                retry = int(meta.get("retry", 0))
                if retry < self.max_retries:
                    # Requeue at the TAIL (not the head): retry after the pipeline
                    # is warm, instead of burning attempts while still cold. This
                    # is what guarantees the full set is eventually captured.
                    unreal.log_warning(
                        f"截图超时，排到队尾重试: cam{meta['cam_idx']:02d}, frame {meta['time_idx']:04d}, retry={retry + 1}/{self.max_retries}"
                    )
                    self.capture_queue.append({
                        "position": meta["camera_position"],
                        "rotation": meta["camera_rotation"],
                        "cam_idx": meta["cam_idx"],
                        "time_idx": meta["time_idx"],
                        "retry": retry + 1,
                    })
                else:
                    unreal.log_error(
                        f"截图重试 {self.max_retries} 次仍未落盘，放弃: cam{meta['cam_idx']:02d}, frame {meta['time_idx']:04d}"
                    )

                self._clear_pending_capture()
                return

        # 队列为空：结束
        if not self.capture_queue:
            self.save_transforms_json()
            self.cleanup_all_cameras()        # 兜底清扫所有残留临时相机
            self.stop_capture_loop()
            n = len(self.transforms_data['frames'])
            unreal.log("=" * 60)
            unreal.log("数据集生成完成!")
            unreal.log(f"本次会话共写入 {n} 帧位姿"
                       + (f" / 目标 {self.expected_frames}" if self.expected_frames else ""))
            if self.expected_frames and n < self.expected_frames:
                unreal.log_error(f"⚠ 缺 {self.expected_frames - n} 帧未采到(见上方 log_error)")
            unreal.log(f"输出目录: {self.output_dir}")
            unreal.log("=" * 60)
            return

        job = self.capture_queue.pop(0)

        # 切到新太阳方向时更新光照(每个 time_idx 一个显式太阳方向)
        if job["time_idx"] != self.current_time_idx:
            self.current_time_idx = job["time_idx"]
            d = self.sun_override_by_time[self.current_time_idx]
            unreal.log(f"\n处理太阳 time_idx={self.current_time_idx}")
            self.set_sun_direction(d)

        # 首批截图先做几次预热（丢弃），让渲染管线热起来再采真帧——冷启动帧
        # 渲染最慢，是超时丢帧的主因。
        if self.warmup_done < self.warmup_target:
            self.warmup_done += 1
            self.capture_queue.insert(0, job)
            self.capture_camera_view(
                job["position"],
                job["rotation"],
                job["cam_idx"],
                job["time_idx"],
                0,
                self.warmup_image_path,
                True
            )
            return

        self.capture_camera_view(
            job["position"],
            job["rotation"],
            job["cam_idx"],
            job["time_idx"],
            int(job.get("retry", 0))
        )

    def start_capture_loop(self):
        try:
            self.tick_handle = unreal.register_slate_post_tick_callback(self._on_capture_tick)
            unreal.log("截图队列已启动（异步串行）")
        except Exception as e:
            unreal.log_error(f"无法注册截图 Tick 回调: {e}")
            raise

    def stop_capture_loop(self):
        if self.tick_handle is not None:
            try:
                unreal.unregister_slate_post_tick_callback(self.tick_handle)
            except Exception:
                pass
            self.tick_handle = None

    def add_transform_data(self, position, rotation, cam_idx, time_idx, sun_direction_ue):
        """添加相机位姿数据到 transforms.json"""

        # 构建变换矩阵 (UE5 坐标系)
        transform_matrix = self.build_transform_matrix(position, rotation)

        frame_data = {
            "file_path": f"cam{cam_idx:02d}/images/{time_idx:04d}.png",
            "transform_matrix": transform_matrix,
            "camera_index": cam_idx,
            "time_index": time_idx,
            "sun_direction_ue": sun_direction_ue
        }

        self.transforms_data["frames"].append(frame_data)

    def build_transform_matrix(self, position, rotation):
        # UE 相机局部坐标轴: X=forward, Y=right, Z=up
        forward = unreal.MathLibrary.get_forward_vector(rotation)
        right = unreal.MathLibrary.get_right_vector(rotation)
        up = unreal.MathLibrary.get_up_vector(rotation)

        return [
            [forward.x, right.x, up.x, position.x / 100.0],  # UE5 cm转m
            [forward.y, right.y, up.y, position.y / 100.0],
            [forward.z, right.z, up.z, position.z / 100.0],
            [0, 0, 0, 1]
        ]

    def save_transforms_json(self):
        """保存 transforms.json 文件"""
        json_path = Path(self.output_dir) / "transforms.json"

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.transforms_data, f, indent=2)

        unreal.log(f"已保存 transforms.json: {json_path}")

    def generate_uniform_suns(self, n_suns, min_elevation_deg=8.0, max_elevation_deg=85.0):
        """
        Fibonacci 螺旋在半球上生成 n 个均匀分布的太阳方向(UE 坐标,指向太阳)。

        全方位均匀覆盖,消除"某个方向几乎所有太阳都与之垂直"的 aniso
        逃逸条件(方向有偏的数据集会让垂直方向的延展缺方位角监督)。
        """
        z_lo = math.sin(math.radians(min_elevation_deg))
        z_hi = math.sin(math.radians(max_elevation_deg))
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))

        suns = []
        for i in range(n_suns):
            z = z_lo + (z_hi - z_lo) * (i + 0.5) / n_suns
            r = math.sqrt(max(0.0, 1.0 - z * z))
            phi = i * golden_angle
            suns.append([r * math.cos(phi), r * math.sin(phi), z])

        unreal.log(f"生成 {len(suns)} 个均匀半球太阳方向")
        return suns

    def generate_uniform_dataset(self, n_suns=60, camera_stride=3):
        """
        均匀太阳数据集:n_suns 个 Fibonacci 半球太阳 × 轮转 1/stride 相机。

        独立的完整数据集:time_index 从 0 开始,输出目录应指向一个全新位置。
        transforms 写 transforms.json(标准名,后续直接走 convert_transforms.py)。
        """
        unreal.log("=" * 60)
        unreal.log(f"开始生成均匀太阳数据集: {n_suns} 太阳 × 1/{camera_stride} 相机")
        unreal.log("=" * 60)

        self.setup_scene()

        camera_positions = self.calculate_camera_positions()
        self.total_camera_count = len(camera_positions)
        suns = self.generate_uniform_suns(n_suns)

        stride = max(1, int(camera_stride))
        self.capture_queue = []
        self.current_time_idx = -1
        self.warmup_done = 0
        self.spawned_cameras = []
        self.sun_override_by_time = {}

        n_frames = 0
        for si, d in enumerate(suns):
            self.sun_override_by_time[si] = d
            for position, rotation, cam_idx in camera_positions:
                if cam_idx % stride != si % stride:
                    continue
                self.capture_queue.append({
                    "position": position,
                    "rotation": rotation,
                    "cam_idx": cam_idx,
                    "time_idx": si,
                    "retry": 0,
                })
                n_frames += 1

        self.transforms_data["frames"] = []
        self.expected_frames = n_frames
        unreal.log(f"均匀队列: {len(suns)} 太阳 × ~{self.total_camera_count // stride} 相机 = {n_frames} 张")
        self.start_capture_loop()
        unreal.log("脚本已进入异步采集流程，完成后会自动输出完成日志")


def main_uniform(output_directory="D:/CloudDatasetUniform", n_suns=60, camera_stride=3):
    """均匀太阳数据集入口:n_suns 个 Fibonacci 半球太阳 × 轮转 1/stride 相机。

    默认 60 太阳 × 1/3 相机 ≈ 1458 帧 → D:/CloudDatasetUniform。
    完成后 transforms.json 在输出目录,走 tools/convert_transforms.py 转 OpenGL。
    """
    disable_background_throttle()
    global ACTIVE_GENERATOR
    ACTIVE_GENERATOR = CloudDatasetGenerator(output_dir=output_directory)
    ACTIVE_GENERATOR.generate_uniform_dataset(n_suns=n_suns, camera_stride=camera_stride)


# 缺省输出目录(可被命令行 -o/--output 覆盖)。
# **env-off 与 env-on 必须指向不同目录**,否则后一次采集会覆盖前一次的图:
#   env-off(SkyAtmosphere 不可视)→ "D:/CloudDatasetUniform"      (Stage 1,已采)
#   env-on (SkyAtmosphere 可视)  → "D:/CloudDatasetUniform_envon"(Stage 2)
# 几何确定性:同一 generator 同参数 → 相机位姿与太阳方向完全一致,两套 transforms 可互换。
DEFAULT_OUTPUT_DIR = "D:/CloudDatasetUniform"

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="UE 均匀太阳体积云数据集采集")
    ap.add_argument("-o", "--output", default=DEFAULT_OUTPUT_DIR,
                    help="输出目录(env-off / env-on 必须用不同目录)")
    ap.add_argument("--n-suns", type=int, default=60, help="Fibonacci 半球太阳数")
    ap.add_argument("--camera-stride", type=int, default=3, help="轮转 1/stride 相机")
    # UE 的 py 命令可能注入自身 argv,用 parse_known_args 避免未知参数报错
    cli, _ = ap.parse_known_args()
    main_uniform(cli.output, n_suns=cli.n_suns, camera_stride=cli.camera_stride)
