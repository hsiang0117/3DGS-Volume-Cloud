"""
精确测准"渲染图里云"在世界系(GL/训练坐标)的轴对齐 bbox 中心。
方法: 空间雕刻 (visual hull) —— 用全部相机位姿把云剪影雕进 3D 网格,
取 occupied voxel 的 per-axis bbox 中心。与点云的 bbox 中心定义一致。
粗(1m)->细(0.25m) 两级。输出 GL bbox 中心/size + 映射回 UE 的平移补偿。
"""
import os, json, math
import numpy as np
from PIL import Image

DS = r"D:\3DGS-Volume-Cloud\data\CloudDataset"
TJ = os.path.join(DS, "transforms_train.json")
LUM_TH = 0.25
FRAME = 30
OCC_FRAC = 0.85   # voxel 投影落在云内的视角占比阈值 -> 判定 occupied


def load_views():
    T = json.load(open(TJ)); fovx = T["camera_angle_x"]; fx = 1.0/math.tan(fovx/2)
    cams = {}
    for fr in T["frames"]:
        cams.setdefault(fr["camera_index"], np.array(fr["transform_matrix"], np.float64))
    views = []
    for ci in sorted(cams):
        ip = os.path.join(DS, f"cam{ci:02d}", "images", f"{FRAME:04d}.png")
        if not os.path.exists(ip):
            d=os.path.join(DS,f"cam{ci:02d}","images"); cs=sorted(os.listdir(d))
            if not cs: continue
            ip=os.path.join(d, cs[len(cs)//2])
        im = np.asarray(Image.open(ip).convert("RGB"), np.float32)/255.0
        H, W = im.shape[:2]
        cloud = (im.max(2) > LUM_TH)
        w2c = np.linalg.inv(cams[ci]); R = w2c[:3,:3]; t = w2c[:3,3]
        views.append((R, t, W, H, cloud, fx))
    return views


def carve(views, lo, hi, step, occ_frac=OCC_FRAC):
    xs = np.arange(lo[0], hi[0]+1e-6, step)
    ys = np.arange(lo[1], hi[1]+1e-6, step)
    zs = np.arange(lo[2], hi[2]+1e-6, step)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)  # (M,3)
    hitcnt = np.zeros(len(pts), np.int32)
    valcnt = np.zeros(len(pts), np.int32)
    for (R, t, W, H, cloud, fx) in views:
        pc = (R @ pts.T).T + t
        z = pc[:,2]; frn = z < -1e-6
        u = (pc[:,0]/(-z))*fx; v = (pc[:,1]/(-z))*fx
        px = ((u+1)*0.5*W).astype(np.int32)
        py = ((1-(v+1)*0.5)*H).astype(np.int32)
        inb = frn & (px>=0)&(px<W)&(py>=0)&(py<H)
        valcnt += inb
        idx = np.where(inb)[0]
        on = cloud[py[idx], px[idx]]
        hitcnt[idx[on]] += 1
    frac = hitcnt / np.maximum(valcnt, 1)
    occ = (frac >= occ_frac) & (valcnt >= 0.5*len(views))
    return pts, occ


def bbox_center(pts, occ):
    P = pts[occ]
    if len(P) == 0:
        return None
    mn = P.min(0); mx = P.max(0)
    return mn, mx, (mn+mx)/2, (mx-mn), len(P)


def main():
    views = load_views()
    print(f"视角数={len(views)}, frame={FRAME}, occ_frac>={OCC_FRAC}")

    # 粗网格 (1m): 覆盖点云可能区 (点云 size~60x33x47, 像素配准心~(3.9,4.25,0.6))
    lo = (-35., -20., -30.); hi = (45., 30., 35.)
    pts, occ = carve(views, lo, hi, 1.0)
    r = bbox_center(pts, occ)
    print(f"\n[粗 1m] occupied voxels={r[4]}")
    print(f"  bbox min={np.round(r[0],2)} max={np.round(r[1],2)}")
    print(f"  CENTER={np.round(r[2],3)} size={np.round(r[3],2)}")
    c = r[2]

    # 细网格 (0.25m): 在粗中心附近聚焦 ±半size+裕度
    half = r[3]/2 + 3.0
    lo2 = c - half; hi2 = c + half
    pts2, occ2 = carve(views, lo2, hi2, 0.25)
    r2 = bbox_center(pts2, occ2)
    print(f"\n[细 0.25m] occupied voxels={r2[4]}")
    print(f"  bbox min={np.round(r2[0],3)} max={np.round(r2[1],3)}")
    print(f"  CENTER={np.round(r2[2],3)} size={np.round(r2[3],3)}")
    C = r2[2]

    print("\n" + "="*64)
    print(f"[精确云心 GL系] = {np.round(C,3)} m   |C|={np.linalg.norm(C):.3f} m")
    # GL -> UE: UE_x=-GL_z, UE_y=GL_x, UE_z=GL_y ; *100 cm
    C_ue = np.array([-C[2], C[0], C[1]]) * 100.0
    print(f"[精确云心 UE系] = {np.round(C_ue,1)} cm")
    print(f"\n[对齐操作-方案A] 把 UE HeterogeneousVolume 从(0,0,0)平移到 (cm):")
    print(f"   {np.round(-C_ue,1)}   (= 让云的bbox中心落到世界原点)")
    print(f"\n[交叉验证] get_actor_bounds 给的 UE AABB 中心 ≈ (-100, 427, 733) cm")
    print(f"           visual hull 给的 UE 云心 = {np.round(C_ue,1)} cm")
    # 稳定性: 不同 occ_frac 复算细网格
    print(f"\n[稳定性] 不同 occ_frac 下的 GL 云心:")
    for thr in (0.75, 0.80, 0.90, 0.95):
        _, o = carve(views, lo2, hi2, 0.25, occ_frac=thr)
        rr = bbox_center(pts2, o)
        if rr:
            print(f"   occ>={thr}: center={np.round(rr[2],3)} size={np.round(rr[3],2)} nvox={rr[4]}")


if __name__ == "__main__":
    main()
