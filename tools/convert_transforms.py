"""
将 transforms.json 中的变换矩阵从 UE5 坐标系转换为 OpenGL/Blender 坐标系

UE5 坐标系 (左手):  X=forward, Y=right, Z=up
OpenGL/Blender 坐标系 (右手): X=right, Y=up, Z=-forward

输入 transforms.json 中的 4x4 矩阵布局 (按 UE 世界轴排列):
    | fx  rx  ux  tx |      f=forward, r=right, u=up
    | fy  ry  uy  ty |      位置已经是米制
    | fz  rz  uz  tz |
    | 0   0   0   1  |

输出 OpenGL 相机矩阵:
    相机局部轴: X=right, Y=up, Z=-forward
    世界轴映射: GL_x = UE_y,  GL_y = UE_z,  GL_z = -UE_x

输出缺省为同目录 transforms_train.json; 设了输出名 transforms_train.json 时,
同时删除同目录的 transforms_train_full.json / transforms_test.json。

用法: python convert_transforms.py <transforms.json> [output.json]
"""

import json
import math
import sys
import copy
from pathlib import Path


def remap_world_vec_ue_to_gl(v):
    """
    将一个 UE5 世界坐标系下的向量重映射到 OpenGL 世界坐标系
        GL_x =  UE_y
        GL_y =  UE_z
        GL_z = -UE_x
    """
    return [v[1], v[2], -v[0]]


def ue_to_opengl(mat):
    """
    将单个 4x4 变换矩阵从 UE5 坐标系转换为 OpenGL/Blender 坐标系

    UE5 世界坐标:  X-forward, Y-right, Z-up  (左手)
    OpenGL 世界坐标: X-right,   Y-up,   Z-back (右手)

    坐标轴映射 (世界):
        GL_x =  UE_y
        GL_y =  UE_z
        GL_z = -UE_x

    对 4x4 矩阵同时做行列变换:
        M_gl = S @ M_ue @ S^-1
    其中 S 是上述坐标轴置换矩阵。

    同时需要翻转相机 look-at 方向:
    OpenGL 相机默认看向 -Z, UE 相机默认看向 +X,
    经过世界轴变换后 UE +X -> GL -Z, 恰好一致, 无需额外翻转。
    """
    # mat[row][col] 行主序: 列0=forward, 列1=right, 列2=up, 列3=pos
    col0 = [mat[0][0], mat[1][0], mat[2][0]]  # forward 轴
    col1 = [mat[0][1], mat[1][1], mat[2][1]]  # right 轴
    col2 = [mat[0][2], mat[1][2], mat[2][2]]  # up 轴
    pos  = [mat[0][3], mat[1][3], mat[2][3]]  # 位置

    def remap_vec(v):
        return [v[1], v[2], -v[0]]

    # 对每个列向量的分量做世界轴重映射
    new_col0 = remap_world_vec_ue_to_gl(col0)  # UE forward
    new_col1 = remap_world_vec_ue_to_gl(col1)  # UE right
    new_col2 = remap_world_vec_ue_to_gl(col2)  # UE up
    new_pos  = remap_world_vec_ue_to_gl(pos)

    # GL 列排列: col0_gl=right(UE_col1), col1_gl=up(UE_col2), col2_gl=-forward(-UE_col0)
    gl_col0 = new_col1                                    # right
    gl_col1 = new_col2                                    # up
    gl_col2 = [-new_col0[0], -new_col0[1], -new_col0[2]] # -forward

    return [
        [gl_col0[0], gl_col1[0], gl_col2[0], new_pos[0]],
        [gl_col0[1], gl_col1[1], gl_col2[1], new_pos[1]],
        [gl_col0[2], gl_col1[2], gl_col2[2], new_pos[2]],
        [0,          0,          0,          1           ]
    ]


def convert_sun_direction(sun_dir_ue):
    """
    将 UE5 坐标系下的太阳方向向量转换为 OpenGL 坐标系下的归一化向量
    使用与相机世界轴一致的映射：GL_x = UE_y, GL_y = UE_z, GL_z = -UE_x
    """
    gl = remap_world_vec_ue_to_gl(sun_dir_ue)
    length = math.sqrt(gl[0] * gl[0] + gl[1] * gl[1] + gl[2] * gl[2])
    if length > 1e-8:
        gl = [gl[0] / length, gl[1] / length, gl[2] / length]
    return gl


def convert_transforms(input_path, output_path=None):
    input_path = Path(input_path)
    if output_path is None:
        # 缺省直接写 loader/split 读取的训练文件: generator -> convert -> split
        output_path = input_path.parent / "transforms_train.json"
    else:
        output_path = Path(output_path)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted = copy.deepcopy(data)

    for frame in converted["frames"]:
        frame["transform_matrix"] = ue_to_opengl(frame["transform_matrix"])

        # 太阳方向：UE5 -> OpenGL
        if "sun_direction_ue" in frame:
            frame["sun_direction"] = convert_sun_direction(frame["sun_direction_ue"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2)

    print(f"转换完成: {len(converted['frames'])} 帧")
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")

    # 写完新的完整训练集后删除过期切分: 否则 split_test_set.py 的幂等重切分会
    # 一直沿用已有的 transforms_train_full.json, 忽略这份新数据。
    if output_path.name == "transforms_train.json":
        for stale in ("transforms_train_full.json", "transforms_test.json"):
            p = output_path.parent / stale
            if p.exists():
                try:
                    p.unlink()
                    print(f"已清除过期切分: {p}")
                except OSError as e:
                    print(f"警告: 无法删除 {p}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python convert_transforms.py <transforms.json> [output.json]")
        print("  缺省输出到同目录 transforms_train.json（直接供 split_test_set.py）")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    convert_transforms(in_path, out_path)
