"""
对齐人工审核图生成: 把 points3d.ply 投影叠加到【新重渲图】D:\CloudDataset\cam##\images\0000.png 上。
- 相机位姿来源: data/CloudDataset/transforms_train.json (与重渲同一套半球采样位姿)
- 点云: data/CloudDataset/points3d.ply (原点, SCALE=0.098)
- 叠加: 底图变暗50%, 点云投影画红点; 绿十字=云轮廓中心, 蓝十字=点云投影中心
- 输出: D:\CloudDataset\_align_review\cam##.png  (每视角一张, 全分辨率)
只处理已存在 0000.png 的视角。
"""
import os, json, math
import numpy as np
from PIL import Image

POSE = r"D:\3DGS-Volume-Cloud\data\CloudDataset\transforms_train.json"
PLY  = r"D:\3DGS-Volume-Cloud\data\CloudDataset\points3d.ply"
NEW  = r"D:\CloudDataset"
OUT  = r"D:\CloudDataset\_align_review"
os.makedirs(OUT, exist_ok=True)
LUM_TH = 0.25


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


def cross(arr, cx, cy, col, s=14):
    H,W = arr.shape[:2]; cx,cy=int(cx),int(cy)
    y0,y1=max(0,cy-s),min(H,cy+s); x0,x1=max(0,cx-s),min(W,cx+s)
    arr[y0:y1, max(0,cx-1):min(W,cx+2)] = col
    arr[max(0,cy-1):min(H,cy+2), x0:x1] = col


def main():
    xyz = read_ply(PLY)
    T = json.load(open(POSE)); fovx=T["camera_angle_x"]; fx=1.0/math.tan(fovx/2)
    cams={}
    for fr in T["frames"]: cams.setdefault(fr["camera_index"], np.array(fr["transform_matrix"],np.float64))

    rows=[]
    done=0
    for ci in sorted(cams):
        ip = os.path.join(NEW, f"cam{ci:02d}", "images", "0000.png")
        if not os.path.exists(ip) or os.path.getsize(ip)==0:
            continue
        im = np.asarray(Image.open(ip).convert("RGB"), np.float32)/255.0
        H,W = im.shape[:2]
        cloud = im.max(2)>LUM_TH
        c2w = cams[ci]; w2c=np.linalg.inv(c2w); R=w2c[:3,:3]; t=w2c[:3,3]
        pc=(R@xyz.T).T+t; z=pc[:,2]; frn=z<-1e-6; zz=np.where(frn,-z,1.0)
        u=(pc[:,0]/zz)*fx; v=(pc[:,1]/zz)*fx
        px=((u+1)*0.5*W).astype(np.int32); py=((1-(v+1)*0.5)*H).astype(np.int32)
        ok=frn&(px>=0)&(px<W)&(py>=0)&(py<H)
        # 底图变暗 + 红点
        out=(im*0.45*255).astype(np.uint8)
        out[py[ok],px[ok]]=[255,40,40]
        # 中心十字
        if cloud.sum()>50:
            ys,xs=np.where(cloud)
            ccx,ccy=(xs.min()+xs.max())/2,(ys.min()+ys.max())/2
            cross(out,ccx,ccy,[0,255,0])   # 绿=云轮廓中心
        if ok.sum()>50:
            pcx,pcy=(px[ok].min()+px[ok].max())/2,(py[ok].min()+py[ok].max())/2
            cross(out,pcx,pcy,[40,140,255]) # 蓝=点云投影中心
        # 指标
        pm=np.zeros((H,W),bool); pm[py[ok],px[ok]]=True
        hit=(cloud[py[ok],px[ok]]).mean() if ok.sum() else 0
        cover=(pm&cloud).sum()/max(cloud.sum(),1)
        off=(round(float(pcx-ccx)),round(float(pcy-ccy))) if (cloud.sum()>50 and ok.sum()>50) else None
        Image.fromarray(out).save(os.path.join(OUT,f"cam{ci:02d}.png"))
        rows.append((ci,hit,cover,off))
        done+=1

    print(f"生成 {done} 张审核图 -> {OUT}")
    print(f"{'cam':>4} {'hit':>6} {'cover':>6}  center_off(px)")
    for ci,hit,cover,off in rows:
        print(f"{ci:>4} {hit:>6.3f} {cover:>6.3f}  {off}")


if __name__ == "__main__":
    main()
