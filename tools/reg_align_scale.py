"""
决定性测量(快速版, 坐标下降): 拟合 平移T + 均匀缩放s (绕点云自身中心缩放),
最大化多视角 mean IoU。回答唯一问题: 最佳 s 是否 ≈1.0。
"""
import os, json, math
import numpy as np
from PIL import Image

DS = r"D:\3DGS-Volume-Cloud\data\CloudDataset"
PLY = os.path.join(DS, "points3d.ply")
TJ = os.path.join(DS, "transforms_train.json")
LUM_TH = 0.25; GRID = 80; N_VIEWS = 12; N_PTS = 25000; FRAME = 30


def read_ply_xyz(path):
    with open(path,"rb") as f:
        assert f.readline().decode().strip()=="ply"; f.readline()
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


xyz = read_ply_xyz(PLY)
ctr0 = (xyz.min(0)+xyz.max(0))/2
if len(xyz) > N_PTS:
    xyz = xyz[np.linspace(0,len(xyz)-1,N_PTS).astype(int)]
T = json.load(open(TJ)); fovx=T["camera_angle_x"]; fx=1.0/math.tan(fovx/2)
cams={}
for fr in T["frames"]: cams.setdefault(fr["camera_index"], np.array(fr["transform_matrix"],np.float64))
ids=sorted(cams); sel=[ids[i] for i in np.linspace(0,len(ids)-1,N_VIEWS).astype(int)]
views=[]
for ci in sel:
    ip=os.path.join(DS,f"cam{ci:02d}","images",f"{FRAME:04d}.png")
    if not os.path.exists(ip):
        d=os.path.join(DS,f"cam{ci:02d}","images"); cs=sorted(os.listdir(d))
        if not cs: continue
        ip=os.path.join(d,cs[len(cs)//2])
    im=np.asarray(Image.open(ip).convert("RGB"),np.float32)/255.0
    H,W=im.shape[:2]; cloud=im.max(2)>LUM_TH
    ys=np.linspace(0,H-1,GRID).astype(int); xs=np.linspace(0,W-1,GRID).astype(int)
    cmask=cloud[np.ix_(ys,xs)]
    w2c=np.linalg.inv(cams[ci]); views.append((w2c[:3,:3], w2c[:3,3], cmask))

Xc = xyz - ctr0   # 居中, 缩放绕此

def iou(s, Tv):
    P = Xc*s + ctr0 + Tv
    acc=[]
    for (R,t,cmask) in views:
        pc=(R@P.T).T+t; z=pc[:,2]; frn=z<-1e-6
        zz=np.where(frn,-z,1.0)
        u=(pc[:,0]/zz)*fx; v=(pc[:,1]/zz)*fx
        px=((u+1)*0.5*GRID).astype(np.int32); py=((1-(v+1)*0.5)*GRID).astype(np.int32)
        ok=frn&(px>=0)&(px<GRID)&(py>=0)&(py<GRID)
        pm=np.zeros((GRID,GRID),bool); pm[py[ok],px[ok]]=True
        inter=(pm&cmask).sum(); union=(pm|cmask).sum(); acc.append(inter/max(union,1))
    return float(np.mean(acc))

def line_search_T(s, Tv, steps):
    best=(iou(s,Tv), Tv.copy())
    for st in steps:
        improved=True
        while improved:
            improved=False
            for ax in range(3):
                for dd in (st,-st):
                    cand=best[1].copy(); cand[ax]+=dd; v=iou(s,cand)
                    if v>best[0]+1e-5: best=(v,cand); improved=True
    return best

print(f"points={len(xyz)} views={len(views)} GRID={GRID}")
print(f"IoU @ s=1,T=0 (现状) = {iou(1.0,np.zeros(3)):.4f}")

# 仅平移
bT = line_search_T(1.0, np.zeros(3), [4,1,0.25])
print(f"[仅平移]    IoU={bT[0]:.4f} T={np.round(bT[1],3)}")

# 平移+缩放交替
best=(bT[0],1.0,bT[1].copy())
for rnd in range(5):
    s0=best[1]
    for ss in np.arange(max(0.6,s0-0.25), s0+0.2501, 0.025):
        r=line_search_T(ss, best[2], [1,0.25])
        if r[0]>best[0]+1e-5: best=(r[0],ss,r[1])
    # 精修
    r=line_search_T(best[1], best[2], [0.5,0.125])
    if r[0]>best[0]: best=(r[0],best[1],r[1])
print(f"[平移+缩放] IoU={best[0]:.4f} s={best[1]:.3f} T={np.round(best[2],3)}")

print("\n"+"="*60)
s=best[1]
print(f"最佳均匀缩放 s = {s:.3f}")
if abs(s-1.0) < 0.05:
    print("结论: s≈1 => 点云尺度已正确, 错配主要是平移")
else:
    print(f"结论: 存在 {s:.3f}x 尺度失配 => _ue_make_ply.py 的 SCALE 0.109 -> {0.109*s:.4f}")
print(f"IoU: 现状{iou(1.0,np.zeros(3)):.3f} -> 仅平移{bT[0]:.3f} -> 平移+缩放{best[0]:.3f}")
