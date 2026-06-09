"""
诊断: 初始化点云 (points3d.ply) 与数据集相机/图像在尺度+位置上的对齐情况。

输出:
  1) 点云 bbox / center / 等效半径
  2) 从 transforms_train.json 反算的相机中心云: 中心, 平均半径, 真实 look-at 收敛点
  3) nerf_normalization 的 translate/radius (= cameras_extent, 驱动 densify 的 percent_dense*extent)
  4) 把点云投影到若干 (cam,frame) 图像, 量 "点落在云像素内的比例(hit)" 与
     "云像素被点覆盖的比例(cover)" —— 错位/尺度不匹配会同时压低这两个数
  5) 点云投影包围框 vs 云轮廓包围框 的中心偏移 & 尺寸比 (定量尺度误差)
"""
import os, json, math
import numpy as np
from PIL import Image

DS = r"D:\3DGS-Volume-Cloud\data\CloudDataset"
PLY = os.path.join(DS, "points3d.ply")
TJ = os.path.join(DS, "transforms_train.json")
LUM_TH = 0.25  # 云像素亮度阈值, 与 _ue_make_ply.py 一致


def read_ply_xyz(path):
    with open(path, "rb") as f:
        # header
        line = f.readline().decode("ascii").strip()
        assert line == "ply"
        fmt = f.readline().decode("ascii").strip()
        n = None
        props = []
        while True:
            l = f.readline().decode("ascii").strip()
            if l.startswith("element vertex"):
                n = int(l.split()[-1])
            elif l.startswith("property"):
                props.append(l.split()[-1])
            elif l == "end_header":
                break
        assert "binary_little_endian" in fmt, fmt
        # 假定 float x y z [+ nx ny nz float] [+ r g b uchar]
        # 逐顶点解析
        import struct
        # 计算每顶点字节: 统计 float 数与 uchar 数
        nf = sum(1 for p in props if p in ("x", "y", "z", "nx", "ny", "nz"))
        nb = sum(1 for p in props if p in ("red", "green", "blue"))
        rec = struct.Struct("<" + "f" * nf + "B" * nb)
        data = f.read(rec.size * n)
        xyz = np.empty((n, 3), np.float64)
        off = 0
        for i in range(n):
            vals = rec.unpack_from(data, off); off += rec.size
            xyz[i, 0] = vals[0]; xyz[i, 1] = vals[1]; xyz[i, 2] = vals[2]
        return xyz


def main():
    print("=" * 70)
    xyz = read_ply_xyz(PLY)
    mn = xyz.min(0); mx = xyz.max(0); c = (mn + mx) / 2; sz = mx - mn
    r = np.linalg.norm(xyz - c, axis=1)
    print(f"[PLY] n={len(xyz)}")
    print(f"[PLY] bbox min={mn.round(3)} max={mx.round(3)}")
    print(f"[PLY] center={c.round(3)} size={sz.round(3)}")
    print(f"[PLY] max_radius(from center)={r.max():.3f}  mean_radius={r.mean():.3f}")
    print(f"[PLY] max_radius(from origin)={np.linalg.norm(xyz,axis=1).max():.3f}")

    print("=" * 70)
    T = json.load(open(TJ))
    fovx = T["camera_angle_x"]
    frames = T["frames"]
    # 相机中心 = c2w[:3,3]
    cams = {}
    fwd_dirs = []
    cam_centers = []
    for fr in frames:
        ci = fr["camera_index"]
        if ci in cams:
            continue
        c2w = np.array(fr["transform_matrix"], np.float64)
        cen = c2w[:3, 3]
        cams[ci] = c2w
        cam_centers.append(cen)
        # OpenGL 相机看向 -Z (c2w 第三列的负向)
        fwd = -c2w[:3, 2]
        fwd_dirs.append((cen, fwd))
    cam_centers = np.array(cam_centers)
    cc_mean = cam_centers.mean(0)
    cc_r = np.linalg.norm(cam_centers - cc_mean, axis=1)
    print(f"[CAM] unique cams={len(cams)}")
    print(f"[CAM] center(mean)={cc_mean.round(3)}")
    print(f"[CAM] radius mean={cc_r.mean():.3f} min={cc_r.min():.3f} max={cc_r.max():.3f}")
    print(f"[CAM] |center| from origin={np.linalg.norm(cc_mean):.3f}")

    # 最小二乘求所有相机视线的最近交汇点 (真实 look-at 收敛点)
    # min Σ |(I - d d^T)(x - o)|^2
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in fwd_dirs:
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ o
    conv = np.linalg.solve(A, b)
    print(f"[CAM] 视线收敛点(真实look-at)={conv.round(3)}")
    print(f"[CAM] 收敛点 vs 点云中心 偏移={np.linalg.norm(conv - c):.3f}")
    print(f"[CAM] 收敛点 vs 原点 偏移={np.linalg.norm(conv):.3f}")

    # nerf normalization (训练里的 cameras_extent)
    center = cam_centers.mean(0)
    diag = np.linalg.norm(cam_centers - center, axis=1).max()
    radius = diag * 1.1
    print(f"[NORM] translate={(-center).round(3)}  radius(cameras_extent)={radius:.3f}")
    print(f"[NORM] percent_dense(0.01)*extent = split尺度阈值 = {0.01*radius:.4f}")

    print("=" * 70)
    # 反投影覆盖率 + 包围框对比
    fx = 1.0 / math.tan(fovx / 2)

    def analyze(ci, frame_idx=30):
        c2w = cams.get(ci)
        if c2w is None:
            return None
        ip = os.path.join(DS, f"cam{ci:02d}", "images", f"{frame_idx:04d}.png")
        if not os.path.exists(ip):
            # 退一步找该相机任意一张
            d = os.path.join(DS, f"cam{ci:02d}", "images")
            cands = sorted(os.listdir(d)) if os.path.isdir(d) else []
            if not cands:
                return None
            ip = os.path.join(d, cands[len(cands)//2])
        im = np.asarray(Image.open(ip).convert("RGB"), np.float32) / 255.0
        H, W = im.shape[:2]
        lum = im.max(2)
        cloud = lum > LUM_TH
        w2c = np.linalg.inv(c2w); R = w2c[:3, :3]; Tt = w2c[:3, 3]
        pc = (R @ xyz.T).T + Tt
        z = pc[:, 2]; front = z < -1e-6
        u = (pc[:, 0] / (-z)) * fx
        v = (pc[:, 1] / (-z)) * fx
        px = ((u + 1) * 0.5 * W).astype(np.int32)
        py = ((1 - (v + 1) * 0.5) * H).astype(np.int32)
        va = front & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        if va.sum() == 0:
            return dict(ci=ci, hit=0, cover=0, valid=0, note="no valid proj")
        hit = (lum[py[va], px[va]] > LUM_TH).mean()
        ptmask = np.zeros((H, W), bool); ptmask[py[va], px[va]] = True
        cover = (cloud & ptmask).sum() / max(cloud.sum(), 1)
        # 包围框对比
        ys, xs = np.where(cloud)
        ppx = px[va]; ppy = py[va]
        res = dict(ci=ci, hit=round(float(hit), 3), cover=round(float(cover), 3),
                   valid=int(va.sum()), HW=(H, W))
        if len(xs) > 0:
            cloud_box = (xs.min(), xs.max(), ys.min(), ys.max())
            cloud_c = ((xs.min()+xs.max())/2, (ys.min()+ys.max())/2)
            cloud_sz = (xs.max()-xs.min(), ys.max()-ys.min())
            pt_box = (ppx.min(), ppx.max(), ppy.min(), ppy.max())
            pt_c = ((ppx.min()+ppx.max())/2, (ppy.min()+ppy.max())/2)
            pt_sz = (ppx.max()-ppx.min(), ppy.max()-ppy.min())
            res["cloud_center_px"] = (round(cloud_c[0]), round(cloud_c[1]))
            res["pt_center_px"] = (round(pt_c[0]), round(pt_c[1]))
            res["center_off_px"] = (round(pt_c[0]-cloud_c[0]), round(pt_c[1]-cloud_c[1]))
            res["cloud_size_px"] = (int(cloud_sz[0]), int(cloud_sz[1]))
            res["pt_size_px"] = (int(pt_sz[0]), int(pt_sz[1]))
            sx = pt_sz[0]/max(cloud_sz[0],1); sy = pt_sz[1]/max(cloud_sz[1],1)
            res["size_ratio(pt/cloud)"] = (round(sx,3), round(sy,3))
        return res

    print("反投影分析 (hit=点落白率, cover=云被覆盖率; 理想都接近1):")
    print("  size_ratio>1 => 点云投影比云大(点云尺度偏大/太近); <1 => 偏小")
    print("  center_off_px 非0 => 点云中心与云轮廓中心在像素上错位")
    for ci in [0, 4, 16, 28, 40, 48]:
        r_ = analyze(ci)
        if r_:
            print("  ", r_)


if __name__ == "__main__":
    main()
