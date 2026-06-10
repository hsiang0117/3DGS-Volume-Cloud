"""
决定性测量: 从多视角"云轮廓质心"三角化出渲染图中云的真实 3D 中心。
每个相机一条射线 (相机中心 -> 该视角云轮廓质心像素), 求与所有射线最近的 3D 点。
- 若该点 ≈ 原点 -> 相机看向点和云一致, 问题主要是尺度/形状
- 若该点远离原点 -> 存在系统性偏移 (点云recenter到原点是错的)
同时输出: 该真实中心 vs 点云中心(原点) 的差 = 应施加给点云的平移修正。
"""
import os, json, math
import numpy as np
from PIL import Image

DS = r"D:\3DGS-Volume-Cloud\data\CloudDataset"
TJ = os.path.join(DS, "transforms_train.json")
LUM_TH = 0.25


def main():
    T = json.load(open(TJ))
    fovx = T["camera_angle_x"]; fx = 1.0 / math.tan(fovx / 2)
    cams = {}
    for fr in T["frames"]:
        ci = fr["camera_index"]
        cams.setdefault(ci, np.array(fr["transform_matrix"], np.float64))

    rays = []  # (origin, dir)
    centroids = []
    for ci, c2w in sorted(cams.items()):
        ip = os.path.join(DS, f"cam{ci:02d}", "images", "0030.png")
        if not os.path.exists(ip):
            d = os.path.join(DS, f"cam{ci:02d}", "images")
            cs = sorted(os.listdir(d));
            if not cs: continue
            ip = os.path.join(d, cs[len(cs)//2])
        im = np.asarray(Image.open(ip).convert("RGB"), np.float32) / 255.0
        H, W = im.shape[:2]
        lum = im.max(2); cloud = lum > LUM_TH
        if cloud.sum() < 50:
            continue
        ys, xs = np.where(cloud)
        # 亮度加权质心 (更贴近质量中心)
        w = lum[ys, xs]
        cx = (xs * w).sum() / w.sum(); cy = (ys * w).sum() / w.sum()
        # 像素 -> 归一化相机射线方向 (OpenGL: x右 y上 z后, 看向-z)
        ndc_x = (cx / W) * 2 - 1
        ndc_y = 1 - (cy / H) * 2
        d_cam = np.array([ndc_x / fx, ndc_y / fx, -1.0])
        d_cam /= np.linalg.norm(d_cam)
        R = c2w[:3, :3]; o = c2w[:3, 3]
        d_world = R @ d_cam
        rays.append((o, d_world))
        centroids.append((ci, cx, cy))

    # 最小二乘最近点: min Σ |(I - dd^T)(x - o)|^2
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ o
    center = np.linalg.solve(A, b)
    # 残差 (各射线到该点的距离)
    res = []
    for o, d in rays:
        v = center - o
        perp = v - (v @ d) * d
        res.append(np.linalg.norm(perp))
    res = np.array(res)

    print(f"用于三角化的相机数: {len(rays)}")
    print(f"[渲染图中云的真实3D中心] = {center.round(3)}  (训练/GL 坐标系, 单位 m)")
    print(f"  |center| 距原点 = {np.linalg.norm(center):.3f} m")
    print(f"  射线交汇残差: mean={res.mean():.3f} max={res.max():.3f} (越小越可信)")
    print()
    print(f"[结论] 点云当前 recenter 到原点 (0,0,0)。")
    print(f"  -> 应给点云施加平移修正 Δ = {center.round(3)}  (把点云从原点搬到云真实中心)")
    print(f"  -> 或等价地: 重新 recenter 点云时, 不要 recenter 到 bbox 中心, 而是对齐到此点。")

    # 顺带: 相机半径与该中心的关系
    cc = np.array([c2w[:3,3] for c2w in cams.values()])
    print()
    print(f"[参考] 相机中心均值={cc.mean(0).round(3)}  到真实云中心的平均距离={np.linalg.norm(cc-center,axis=1).mean():.3f} m")


if __name__ == "__main__":
    main()
