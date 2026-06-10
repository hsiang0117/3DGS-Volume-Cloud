"""
重渲前预演: 新点云(原点,SCALE=0.098) 平移 +T 后投影到现有图,
模拟"UE云已移到原点、重渲后"的对齐效果。
- 逐视角 IoU/hit/cover (不只看均值)
- 几张叠加图(红=点云投影, 灰=现有渲染云)
T 来自 reg_align_scale 的仅平移最优解。
"""
import os, json, math
import numpy as np
from PIL import Image

DS = r"D:\3DGS-Volume-Cloud\data\CloudDataset"
PLY = os.path.join(DS, "points3d.ply")
TJ = os.path.join(DS, "transforms_train.json")
OUT = r"D:\3DGS-Volume-Cloud\tools\_preview_rerender"
os.makedirs(OUT, exist_ok=True)
LUM_TH = 0.25; FRAME = 30
T_SHIFT = np.array([4.75, 7.5, 1.0])   # 仅平移最优(模拟重渲后云回到原点)


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


def proj(xyz, c2w, fovx, W, H):
    fx=1.0/math.tan(fovx/2)
    w2c=np.linalg.inv(c2w); R=w2c[:3,:3]; t=w2c[:3,3]
    pc=(R@xyz.T).T+t; z=pc[:,2]; frn=z<-1e-6
    zz=np.where(frn,-z,1.0)
    u=(pc[:,0]/zz)*fx; v=(pc[:,1]/zz)*fx
    px=((u+1)*0.5*W).astype(np.int32); py=((1-(v+1)*0.5)*H).astype(np.int32)
    ok=frn&(px>=0)&(px<W)&(py>=0)&(py<H)
    return px,py,ok


def main():
    xyz0 = read_ply_xyz(PLY)
    xyz = xyz0 + T_SHIFT
    T = json.load(open(TJ)); fovx=T["camera_angle_x"]
    cams={}
    for fr in T["frames"]: cams.setdefault(fr["camera_index"], np.array(fr["transform_matrix"],np.float64))
    ids=sorted(cams)

    def metrics(P, ci):
        ip=os.path.join(DS,f"cam{ci:02d}","images",f"{FRAME:04d}.png")
        if not os.path.exists(ip):
            d=os.path.join(DS,f"cam{ci:02d}","images"); cs=sorted(os.listdir(d))
            if not cs: return None
            ip=os.path.join(d,cs[len(cs)//2])
        im=np.asarray(Image.open(ip).convert("RGB"),np.float32)/255.0
        H,W=im.shape[:2]; cloud=im.max(2)>LUM_TH
        px,py,ok=proj(P,cams[ci],fovx,W,H)
        pm=np.zeros((H,W),bool); pm[py[ok],px[ok]]=True
        inter=(pm&cloud).sum(); union=(pm|cloud).sum()
        return (inter/max(union,1), (cloud[py[ok],px[ok]]).mean() if ok.sum() else 0,
                inter/max(cloud.sum(),1))

    print("="*64)
    print(f"预演: 新点云 +T{T_SHIFT} (模拟重渲后云在原点)")
    print(f"{'cam':>5} | {'IoU_before':>10} {'IoU_after':>10} | {'hit_aft':>8} {'cover_aft':>9}")
    iob=[]; ioa=[]
    for ci in ids:
        mb=metrics(xyz0, ci); ma=metrics(xyz, ci)
        if mb and ma:
            iob.append(mb[0]); ioa.append(ma[0])
            if ci % 6 == 0 or ci in (0,16,28,48):
                print(f"{ci:>5} | {mb[0]:>10.3f} {ma[0]:>10.3f} | {ma[1]:>8.3f} {ma[2]:>9.3f}")
    iob=np.array(iob); ioa=np.array(ioa)
    print("-"*64)
    print(f"全{len(ioa)}视角 IoU: before {iob.mean():.3f} -> after {ioa.mean():.3f}")
    print(f"  after IoU: min={ioa.min():.3f} p25={np.percentile(ioa,25):.3f} "
          f"median={np.median(ioa):.3f} max={ioa.max():.3f}")
    print(f"  after IoU<0.85 的视角数: {(ioa<0.85).sum()}/{len(ioa)}")

    # 叠加图
    def overlay(ci):
        ip=os.path.join(DS,f"cam{ci:02d}","images",f"{FRAME:04d}.png")
        im=np.asarray(Image.open(ip).convert("RGB"),np.float32)/255.0
        H,W=im.shape[:2]
        px,py,ok=proj(xyz,cams[ci],fovx,W,H)
        out=(im*0.5*255).astype(np.uint8); out[py[ok],px[ok]]=[255,0,0]
        Image.fromarray(out).resize((W//2,H//2)).save(os.path.join(OUT,f"preview_cam{ci:02d}.png"))
    for ci in [0,16,28,48]:
        overlay(ci)
    print(f"叠加图: {OUT}")


if __name__ == "__main__":
    main()
