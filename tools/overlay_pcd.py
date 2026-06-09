"""
把 points3d.ply 投影叠加到数据集图像上, 直观判断错配类型 (平移/尺度/形状)。
另外估计初始高斯尺度(最近邻距离) vs densify split 阈值(percent_dense*extent)。
输出 PNG 到 tools/_overlay/。
"""
import os, json, math
import numpy as np
from PIL import Image

DS = r"D:\3DGS-Volume-Cloud\data\CloudDataset"
PLY = os.path.join(DS, "points3d.ply")
TJ = os.path.join(DS, "transforms_train.json")
OUT = r"D:\3DGS-Volume-Cloud\tools\_overlay"
os.makedirs(OUT, exist_ok=True)
LUM_TH = 0.25


def read_ply_xyz(path):
    with open(path, "rb") as f:
        assert f.readline().decode().strip() == "ply"
        f.readline()  # format
        n = None; props = []
        while True:
            l = f.readline().decode().strip()
            if l.startswith("element vertex"):
                n = int(l.split()[-1])
            elif l.startswith("property"):
                props.append(l.split()[-1])
            elif l == "end_header":
                break
        fl = [p for p in props if p in ("x","y","z","nx","ny","nz")]
        by = [p for p in props if p in ("red","green","blue")]
        dt = np.dtype([(p, "<f4") for p in fl] + [(p, "u1") for p in by])
        arr = np.frombuffer(f.read(dt.itemsize * n), dtype=dt, count=n)
        return np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)


def nn_dist_estimate(xyz, sample=3000, seed=0):
    """随机抽样估计最近邻距离分布 (代表初始高斯尺度)。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xyz), min(sample, len(xyz)), replace=False)
    Q = xyz[idx]
    # 分块算到全集的最近邻 (排除自身)
    nn = np.empty(len(Q))
    B = 500
    for i in range(0, len(Q), B):
        q = Q[i:i+B]
        d2 = ((q[:, None, :] - xyz[None, :, :]) ** 2).sum(-1)  # (b, N)
        # 自身距离=0, 取第二小
        d2.sort(1)
        nn[i:i+B] = np.sqrt(d2[:, 1])
    return nn


def main():
    xyz = read_ply_xyz(PLY)
    T = json.load(open(TJ))
    fovx = T["camera_angle_x"]; fx = 1.0 / math.tan(fovx / 2)
    cams = {}
    for fr in T["frames"]:
        ci = fr["camera_index"]
        cams.setdefault(ci, np.array(fr["transform_matrix"], np.float64))

    # 初始高斯尺度 vs split 阈值
    cc = np.array([cams[c][:3, 3] for c in cams])
    extent = np.linalg.norm(cc - cc.mean(0), axis=1).max() * 1.1
    nn = nn_dist_estimate(xyz)
    print(f"[SCALE] cameras_extent={extent:.3f}")
    print(f"[SCALE] percent_dense(0.01)*extent (split尺度阈值) = {0.01*extent:.4f}")
    print(f"[SCALE] 初始高斯尺度≈最近邻距离: p50={np.median(nn):.4f} p90={np.percentile(nn,90):.4f} mean={nn.mean():.4f}")
    print(f"[SCALE] 比值 (split阈值 / 初始尺度p50) = {0.01*extent/np.median(nn):.1f}x")
    print(f"        >>1: 初始高斯远小于split阈值 -> 几乎只clone不split; <1: 立刻被split")

    def overlay(ci, frame_idx=30, down=2):
        c2w = cams[ci]
        ip = os.path.join(DS, f"cam{ci:02d}", "images", f"{frame_idx:04d}.png")
        if not os.path.exists(ip):
            d = os.path.join(DS, f"cam{ci:02d}", "images")
            cs = sorted(os.listdir(d)); ip = os.path.join(d, cs[len(cs)//2])
        im = np.asarray(Image.open(ip).convert("RGB"), np.float32) / 255.0
        H, W = im.shape[:2]
        w2c = np.linalg.inv(c2w); R = w2c[:3, :3]; Tt = w2c[:3, 3]
        pc = (R @ xyz.T).T + Tt
        z = pc[:, 2]; front = z < -1e-6
        u = (pc[:, 0] / (-z)) * fx; v = (pc[:, 1] / (-z)) * fx
        px = ((u + 1) * 0.5 * W).astype(np.int32)
        py = ((1 - (v + 1) * 0.5) * H).astype(np.int32)
        va = front & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        # 底图变灰 + 点画红
        out = (im * 0.5 * 255).astype(np.uint8)
        out[py[va], px[va]] = [255, 0, 0]
        # 云轮廓中心(绿十字) & 点云投影中心(蓝十字)
        lum = im.max(2); cloud = lum > LUM_TH
        def cross(arr, cx, cy, col, s=12):
            cx, cy = int(cx), int(cy)
            arr[max(0,cy-s):cy+s, max(0,cx-1):cx+2] = col
            arr[max(0,cy-1):cy+2, max(0,cx-s):cx+s] = col
        if cloud.sum() > 0:
            ys, xs = np.where(cloud)
            cross(out, (xs.min()+xs.max())/2, (ys.min()+ys.max())/2, [0,255,0])
        if va.sum() > 0:
            cross(out, (px[va].min()+px[va].max())/2, (py[va].min()+py[va].max())/2, [0,128,255])
        img = Image.fromarray(out)
        if down > 1:
            img = img.resize((W//down, H//down))
        p = os.path.join(OUT, f"overlay_cam{ci:02d}.png")
        img.save(p)
        return p

    for ci in [0, 16, 28, 48]:
        p = overlay(ci)
        print("saved", p)


if __name__ == "__main__":
    main()
