"""
精确平移配准: 原点点云 vs 现有图(云在原 actor 0,0,0)。
用膨胀IoU(避开点稀疏假象)做精细坐标下降, 求最佳 T_gl。
T_gl = 现有图中云(密度bbox)中心在GL的位置。
-> 正确的UE actor位置 = -map_gl_to_ue(T_gl)。
"""
import os, json, math
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

DS = r"D:\3DGS-Volume-Cloud\data\CloudDataset"
PLY = os.path.join(DS, "points3d.ply")
TJ = os.path.join(DS, "transforms_train.json")
LUM_TH = 0.25; FRAME = 30; N_VIEWS = 16; N_PTS = 40000; DOWN = 2; DIL = 3


def read_ply(path):
    with open(path,"rb") as f:
        f.readline(); f.readline(); n=None; props=[]
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


xyz = read_ply(PLY)
if len(xyz) > N_PTS: xyz = xyz[np.linspace(0,len(xyz)-1,N_PTS).astype(int)]
T = json.load(open(TJ)); fovx=T["camera_angle_x"]; fx=1.0/math.tan(fovx/2)
cams={}
for fr in T["frames"]: cams.setdefault(fr["camera_index"], np.array(fr["transform_matrix"],np.float64))
ids=sorted(cams); sel=[ids[i] for i in np.linspace(0,len(ids)-1,N_VIEWS).astype(int)]
views=[]
for ci in sel:
    ip=os.path.join(DS,f"cam{ci:02d}","images",f"{FRAME:04d}.png")
    if not os.path.exists(ip): continue
    im=np.asarray(Image.open(ip).convert("RGB"),np.float32)/255.0
    H,W=im.shape[:2]
    cloud=(im.max(2)>LUM_TH)[::DOWN,::DOWN]
    h,w=cloud.shape
    w2c=np.linalg.inv(cams[ci]); views.append((w2c[:3,:3],w2c[:3,3],w,h,cloud))

def iou(Tv):
    acc=[]
    for (R,t,w,h,cloud) in views:
        pc=(R@(xyz+Tv).T).T+t; z=pc[:,2]; frn=z<-1e-6; zz=np.where(frn,-z,1.0)
        u=(pc[:,0]/zz)*fx; v=(pc[:,1]/zz)*fx
        px=((u+1)*0.5*w).astype(np.int32); py=((1-(v+1)*0.5)*h).astype(np.int32)
        ok=frn&(px>=0)&(px<w)&(py>=0)&(py<h)
        pm=np.zeros((h,w),bool); pm[py[ok],px[ok]]=True
        pm=binary_dilation(pm,iterations=DIL)
        inter=(pm&cloud).sum(); union=(pm|cloud).sum(); acc.append(inter/max(union,1))
    return float(np.mean(acc))

best=(iou(np.zeros(3)), np.zeros(3))
print(f"IoU@T=0 = {best[0]:.4f}")
for st in [2.0,0.5,0.125,0.0625]:
    improved=True
    while improved:
        improved=False
        for ax in range(3):
            for dd in (st,-st):
                cand=best[1].copy(); cand[ax]+=dd; v=iou(cand)
                if v>best[0]+1e-4: best=(v,cand); improved=True
    print(f"  step={st}: IoU={best[0]:.4f} T={np.round(best[1],3)}")

Tg=best[1]
ue_center = np.array([-Tg[2], Tg[0], Tg[1]])*100   # GL->UE *100
new_actor = -ue_center
print("\n"+"="*60)
print(f"[精确 T_gl] = {np.round(Tg,3)} m  (膨胀IoU={best[0]:.4f})")
print(f"[云密度中心 UE] = {np.round(ue_center,1)} cm")
print(f"[正确 actor 位置] = {np.round(new_actor,1)} cm  (= -云中心, 使其落到原点)")
print(f"[对照] 我当前移到了 (137.5, -337.5, -175) cm")
print(f"[对照] UE bounds 中心(移云前) = (-100, 427, 733) cm -> 该方案actor = (100,-427,-733)")
