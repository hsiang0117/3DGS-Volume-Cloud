"""
验证: 把现有点云整体平移 +C_gl(=visual hull 云心) 后重投影到现有 UE 图,
IoU 应从 ~0.58 跳到高位 => 证实"错配是纯平移 且 云心测值准确"。
等价于: 重渲时把云移到原点(方案A)后, 原点处点云与新图的对齐效果预演。
同时输出平移后的逐视角 hit/cover + 叠加图(几张)。
"""
import os, json, math
import numpy as np
from PIL import Image

DS = r"D:\3DGS-Volume-Cloud\data\CloudDataset"
PLY = os.path.join(DS, "points3d.ply")
TJ = os.path.join(DS, "transforms_train.json")
OUT = r"D:\3DGS-Volume-Cloud\tools\_overlay_shift"
os.makedirs(OUT, exist_ok=True)
LUM_TH = 0.25
FRAME = 30
C_GL = np.array([3.375, 1.75, 1.375])   # visual hull 测出的云心(GL)


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


def metrics(xyz, cams, fovx, ids):
    fx=1.0/math.tan(fovx/2); ious=[]; hits=[]; covers=[]
    for ci in ids:
        ip=os.path.join(DS,f"cam{ci:02d}","images",f"{FRAME:04d}.png")
        if not os.path.exists(ip): continue
        im=np.asarray(Image.open(ip).convert("RGB"),np.float32)/255.0
        H,W=im.shape[:2]; cloud=im.max(2)>LUM_TH
        w2c=np.linalg.inv(cams[ci]); R=w2c[:3,:3]; t=w2c[:3,3]
        pc=(R@xyz.T).T+t; z=pc[:,2]; frn=z<-1e-6
        u=(pc[:,0]/(-z))*fx; v=(pc[:,1]/(-z))*fx
        px=((u+1)*0.5*W).astype(np.int32); py=((1-(v+1)*0.5)*H).astype(np.int32)
        ok=frn&(px>=0)&(px<W)&(py>=0)&(py<H)
        pm=np.zeros((H,W),bool); pm[py[ok],px[ok]]=True
        inter=(pm&cloud).sum(); union=(pm|cloud).sum()
        ious.append(inter/max(union,1))
        hits.append((cloud[py[ok],px[ok]]).mean() if ok.sum() else 0)
        covers.append(inter/max(cloud.sum(),1))
    return np.mean(ious), np.mean(hits), np.mean(covers)


def overlay(xyz, cams, fovx, ci, tag):
    fx=1.0/math.tan(fovx/2)
    ip=os.path.join(DS,f"cam{ci:02d}","images",f"{FRAME:04d}.png")
    im=np.asarray(Image.open(ip).convert("RGB"),np.float32)/255.0
    H,W=im.shape[:2]
    w2c=np.linalg.inv(cams[ci]); R=w2c[:3,:3]; t=w2c[:3,3]
    pc=(R@xyz.T).T+t; z=pc[:,2]; frn=z<-1e-6
    u=(pc[:,0]/(-z))*fx; v=(pc[:,1]/(-z))*fx
    px=((u+1)*0.5*W).astype(np.int32); py=((1-(v+1)*0.5)*H).astype(np.int32)
    ok=frn&(px>=0)&(px<W)&(py>=0)&(py<H)
    out=(im*0.5*255).astype(np.uint8); out[py[ok],px[ok]]=[255,0,0]
    Image.fromarray(out).resize((W//2,H//2)).save(os.path.join(OUT,f"shift_{tag}_cam{ci:02d}.png"))


def main():
    xyz=read_ply_xyz(PLY)
    T=json.load(open(TJ)); fovx=T["camera_angle_x"]
    cams={}
    for fr in T["frames"]: cams.setdefault(fr["camera_index"], np.array(fr["transform_matrix"],np.float64))
    ids=sorted(cams)

    print("="*60)
    i0,h0,c0=metrics(xyz, cams, fovx, ids)
    print(f"[平移前] 全49视角  IoU={i0:.4f}  hit={h0:.4f}  cover={c0:.4f}")
    xyz2=xyz+C_GL
    i1,h1,c1=metrics(xyz2, cams, fovx, ids)
    print(f"[平移后] +C_gl={C_GL}  IoU={i1:.4f}  hit={h1:.4f}  cover={c1:.4f}")
    print(f"  IoU 提升 {i0:.3f} -> {i1:.3f}  (cover 尤其关键: {c0:.3f}->{c1:.3f})")

    # 也试 IoU 配准给的 (3.875,4.25,0.625) 作对比
    alt=np.array([3.875,4.25,0.625])
    ia,ha,ca=metrics(xyz+alt, cams, fovx, ids)
    print(f"[对比] +IoU配准{alt}  IoU={ia:.4f} cover={ca:.4f}")

    for ci in [0,28,48]:
        overlay(xyz, cams, fovx, ci, "before")
        overlay(xyz2, cams, fovx, ci, "after")
    print(f"叠加图: {OUT}")


if __name__ == "__main__":
    main()
