"""
针对 UE 数据集: 求"把 points3d.ply 平移多少(GL 米)能与渲染图云轮廓最佳重叠"。
该平移 T = 云相对世界原点的真实偏移(GL系)。
-> 要让重渲图对齐到 origin 处的点云, 就把 UE 里的云平移 -T(映射回UE) 使其居中到原点。

方法: 下采样剪影 mask, 坐标下降/粗到细网格搜索 T 最大化 mean IoU(跨多视角)。
不依赖 UE-vs-Blender 变换细节, 纯像素证据。
"""
import os, json, math
import numpy as np
from PIL import Image

DS = r"D:\3DGS-Volume-Cloud\data\CloudDataset"
PLY = os.path.join(DS, "points3d.ply")
TJ = os.path.join(DS, "transforms_train.json")
LUM_TH = 0.25
GRID = 96            # 下采样 mask 分辨率
N_VIEWS = 16         # 参与配准的视角数(均匀取)
N_PTS = 60000        # 点云子采样
FRAME = 30


def read_ply_xyz(path):
    with open(path, "rb") as f:
        assert f.readline().decode().strip() == "ply"; f.readline()
        n=None; props=[]
        while True:
            l=f.readline().decode().strip()
            if l.startswith("element vertex"): n=int(l.split()[-1])
            elif l.startswith("property"): props.append(l.split()[-1])
            elif l=="end_header": break
        fl=[p for p in props if p in ("x","y","z","nx","ny","nz")]
        by=[p for p in props if p in ("red","green","blue")]
        dt=np.dtype([(p,"<f4") for p in fl]+[(p,"u1") for p in by])
        a=np.frombuffer(f.read(dt.itemsize*n),dtype=dt,count=n)
        return np.stack([a["x"],a["y"],a["z"]],1).astype(np.float64)


def main():
    xyz = read_ply_xyz(PLY)
    if len(xyz) > N_PTS:
        idx = np.linspace(0, len(xyz)-1, N_PTS).astype(int); xyz = xyz[idx]
    T = json.load(open(TJ)); fovx = T["camera_angle_x"]; fx = 1.0/math.tan(fovx/2)
    cams = {}
    for fr in T["frames"]:
        cams.setdefault(fr["camera_index"], np.array(fr["transform_matrix"], np.float64))
    cam_ids = sorted(cams)
    sel = [cam_ids[i] for i in np.linspace(0, len(cam_ids)-1, N_VIEWS).astype(int)]

    # 预载每个视角的 cloud mask(下采样) + 相机 R,T
    views = []
    for ci in sel:
        ip = os.path.join(DS, f"cam{ci:02d}", "images", f"{FRAME:04d}.png")
        if not os.path.exists(ip):
            d=os.path.join(DS,f"cam{ci:02d}","images"); cs=sorted(os.listdir(d))
            if not cs: continue
            ip=os.path.join(d, cs[len(cs)//2])
        im = np.asarray(Image.open(ip).convert("RGB"), np.float32)/255.0
        H, W = im.shape[:2]
        cloud = (im.max(2) > LUM_TH)
        # 下采样 cloud mask 到 GRID
        ys = (np.linspace(0, H-1, GRID)).astype(int)
        xs = (np.linspace(0, W-1, GRID)).astype(int)
        cmask = cloud[np.ix_(ys, xs)]
        w2c = np.linalg.inv(cams[ci]); R = w2c[:3,:3]; t = w2c[:3,3]
        views.append((R, t, W, H, cmask))

    def mean_iou(Tvec):
        ious = []
        for (R, t, W, H, cmask) in views:
            pc = (R @ (xyz + Tvec).T).T + t
            z = pc[:,2]; frn = z < -1e-6
            u = (pc[:,0]/(-z))*fx; v = (pc[:,1]/(-z))*fx
            px = ((u+1)*0.5*GRID).astype(np.int32)
            py = ((1-(v+1)*0.5)*GRID).astype(np.int32)
            ok = frn & (px>=0)&(px<GRID)&(py>=0)&(py<GRID)
            pm = np.zeros((GRID, GRID), bool)
            pm[py[ok], px[ok]] = True
            inter = (pm & cmask).sum(); union = (pm | cmask).sum()
            ious.append(inter/max(union,1))
        return float(np.mean(ious))

    # 粗到细搜索, init = 0
    best = (mean_iou(np.zeros(3)), np.zeros(3))
    print(f"IoU @ T=0 (现状, 点云在原点) = {best[0]:.4f}")
    for (lo, step) in [(8.0, 2.0), (2.0, 0.5), (0.5, 0.125)]:
        c = best[1].copy()
        rng = np.arange(-lo, lo+1e-9, step)
        for dx in rng:
            for dy in rng:
                for dz in rng:
                    Tv = c + np.array([dx, dy, dz])
                    s = mean_iou(Tv)
                    if s > best[0]:
                        best = (s, Tv)
        print(f"  refine lo={lo} step={step}: best IoU={best[0]:.4f} T={np.round(best[1],3)}")

    Topt = best[1]
    print("\n" + "="*60)
    print(f"[最佳平移 T] (GL米, 施加到点云能贴合云轮廓) = {np.round(Topt,3)}")
    print(f"  => 云在GL系相对原点偏移 ≈ {np.round(Topt,3)}, |T|={np.linalg.norm(Topt):.3f} m")
    print(f"  IoU: 0->{best[0]:.4f} (越高越贴合)")
    # GL -> UE 位移映射: UE_x=-GL_z, UE_y=GL_x, UE_z=GL_y ; *100 cm
    # 要把云移到原点 => 云平移 -Topt(GL)
    sGL = -Topt
    ue_dx = -sGL[2]*100; ue_dy = sGL[0]*100; ue_dz = sGL[1]*100
    print(f"\n[UE 操作] 把 HeterogeneousVolume 平移 (cm): "
          f"({ue_dx:.1f}, {ue_dy:.1f}, {ue_dz:.1f})")
    print(f"  当前 actor location=(0,0,0) -> 新 location=({ue_dx:.1f}, {ue_dy:.1f}, {ue_dz:.1f}) cm")
    print(f"  (等价: 云的GL中心 {np.round(Topt,2)} 映射回UE并取负)")


if __name__ == "__main__":
    main()
