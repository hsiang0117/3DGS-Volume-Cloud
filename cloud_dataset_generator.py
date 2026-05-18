"""
UE5 体积云数据集采集脚本
用于采集单个静态体积云在不同视角和TOD光照下的渲染图像、深度图和环境光贴图
"""

import unreal
import math
import json
import time
from pathlib import Path


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
        self.zenith_angles = [0, 20, 40, 60, 75]  # 天顶角
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

        # TOD 参数
        self.time_frames = 61  # 12小时分60帧
        self.time_interval = 12.0 / 60  # 每帧间隔0.2小时

        # 存储相机位姿数据
        self.transforms_data = {
            "camera_angle_x": math.radians(90),  # FOV
            "frames": []
        }

        # 截图任务状态（异步串行）
        self.capture_camera_actor = None
        self.capture_task = None
        self.pending_capture_meta = None
        self.capture_queue = []
        self.current_time_idx = -1
        self.tick_handle = None
        self.total_camera_count = 0
        self.capture_file_timeout_seconds = 6.0
        self.screenshot_warmed_up = False
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

    def set_time_of_day(self, hour):
        """
        设置时间(TOD)

        Args:
            hour: 小时数 (0-24)
        """
        if not self.sun_light:
            return

        sun_angle = hour * -15.0

        # 设置太阳旋转
        sun_rotation = unreal.Rotator(0, sun_angle, 0)
        self.sun_light.set_actor_rotation(sun_rotation, False)

        # 更新 Sky Light
        if self.sky_light:
            light_component = self.sky_light.get_component_by_class(unreal.SkyLightComponent)
            if light_component:
                light_component.recapture_sky()

        unreal.log(f"设置 TOD: {hour:.1f} 小时, 太阳角度: {sun_angle:.1f}度")

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
        # 每张图创建一个临时相机，待截图任务完成后销毁
        camera = self.editor_level_lib.spawn_actor_from_class(
            unreal.CameraActor,
            camera_position,
            camera_rotation
        )

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

    def cleanup_capture_camera(self):
        """清理复用的临时相机"""
        if self.capture_camera_actor:
            self.editor_actor_subsystem.destroy_actor(self.capture_camera_actor)
            self.capture_camera_actor = None

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
            try:
                self.editor_actor_subsystem.destroy_actor(self.pending_capture_meta["camera"])
            except Exception:
                pass

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

                retry = int(meta.get("retry", 0))
                if retry < 3:
                    unreal.log_warning(
                        f"截图等待超时，重试: cam{meta['cam_idx']:02d}, frame {meta['time_idx']:04d}, retry={retry + 1}"
                    )
                    self.capture_queue.insert(0, {
                        "position": meta["camera_position"],
                        "rotation": meta["camera_rotation"],
                        "cam_idx": meta["cam_idx"],
                        "time_idx": meta["time_idx"],
                        "retry": retry + 1,
                    })
                else:
                    unreal.log_error(
                        f"截图多次重试仍未落盘，跳过: cam{meta['cam_idx']:02d}, frame {meta['time_idx']:04d}"
                    )

                self._clear_pending_capture()
                return

        # 队列为空：结束
        if not self.capture_queue:
            self.save_transforms_json()
            self.stop_capture_loop()
            unreal.log("=" * 60)
            unreal.log("数据集生成完成!")
            unreal.log(f"总计: {self.total_camera_count} 个相机 × {self.time_frames} 帧 = {self.total_camera_count * self.time_frames} 张图像")
            unreal.log(f"输出目录: {self.output_dir}")
            unreal.log("=" * 60)
            return

        job = self.capture_queue.pop(0)

        # 切换到新的时间帧时，更新 TOD
        if job["time_idx"] != self.current_time_idx:
            self.current_time_idx = job["time_idx"]
            current_hour = self.current_time_idx * self.time_interval
            unreal.log(f"\n处理时间帧 {self.current_time_idx + 1}/{self.time_frames} (Hour: {current_hour:.2f})")
            self.set_time_of_day(current_hour)

        # 首张截图先做一次预热，避免首帧相机绑定滞后导致的错位
        if not self.screenshot_warmed_up:
            self.screenshot_warmed_up = True
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

    def generate_dataset(self):
        """主函数:生成完整数据集（异步串行截图）"""
        unreal.log("=" * 60)
        unreal.log("开始生成体积云数据集")
        unreal.log("=" * 60)

        # 1. 设置场景
        self.setup_scene()

        # 2. 计算所有相机位置
        camera_positions = self.calculate_camera_positions()
        self.total_camera_count = len(camera_positions)

        # 3. 构建任务队列（按 time_idx -> cam_idx）
        self.capture_queue = []
        self.current_time_idx = -1
        self.screenshot_warmed_up = False
        for time_idx in range(self.time_frames):
            for position, rotation, cam_idx in camera_positions:
                self.capture_queue.append({
                    "position": position,
                    "rotation": rotation,
                    "cam_idx": cam_idx,
                    "time_idx": time_idx,
                    "retry": 0,
                })

        # 4. 启动异步串行截图
        self.start_capture_loop()
        unreal.log("脚本已进入异步采集流程，完成后会自动输出完成日志")


def main():
    """主入口函数"""
    # 配置输出目录
    output_directory = "D:/CloudDataset"  # 可根据需要修改
    
    # 创建生成器并运行
    global ACTIVE_GENERATOR
    ACTIVE_GENERATOR = CloudDatasetGenerator(output_dir=output_directory)
    ACTIVE_GENERATOR.generate_dataset()


if __name__ == "__main__":
    main()
