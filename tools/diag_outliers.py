"""
诊断 _align_review 异常视角(cam10/22/34/46): 残留平移 vs 稀薄采样不足。
对每个视角:
 1) 单视角最优2D平移(在图像平面平移点云投影)能把IoU拉到多高 -> 残留平移可救性
 2) "云有但点没有"的缺失区 的平均亮度 vs "云亮核"亮度 -> 缺失区是否=暗部稀薄区
 3) 点云投影范围 vs 云轮廓范围 的尺寸比 -> 是否整体偏小(采样不足)还是错位
对照组: cam30(对齐好的)同样测, 作基线。
"""
import os, json, math
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

POSE = r"D:\3DGS-Volume-Cloud\data\CloudDataset\transforms_train.json"
PLY  = r"D:\3DGS-Volume-Cloud\data\CloudDataset\points3d.ply"
NEW  = r"D:\CloudDataset"
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


xyz = read_ply(PLY)
T = json.load(open(POSE)); fovx=T["camera_angle_x"]; fx=1.0/math.tan(fovx/2)
cams={}
for fr in T["frames"]: cams.setdefault(fr["camera_index"], np.array(fr["transform_matrix"],np.float64))


def proj(ci):
    c2w=cams[ci]; w2c=np.linalg.inv(c2w); R=w2c[:3,:3]; t=w2c[:3,3]
    pc=(R@xyz.T).T+t; z=pc[:,2]; frn=z<-1e-6; zz=np.where(frn,-z,1.0)
    u=(pc[:,0]/zz)*fx; v=(pc[:,1]/zz)*fx
    return u,v,frn


def analyze(ci):
    ip=os.path.join(NEW,f"cam{ci:02d}","images","0000.png")
    im=np.asarray(Image.open(ip).convert("RGB"),np.float32)/255.0
    H,W=im.shape[:2]; lum=im.max(2); cloud=lum>LUM_TH
    u,v,frn=proj(ci)
    px=((u+1)*0.5*W).astype(np.int32); py=((1-(v+1)*0.5)*H).astype(np.int32)
    ok=frn&(px>=0)&(px<W)&(py>=0)&(py<H)
    pm=np.zeros((H,W),bool); pm[py[ok],px[ok]]=True
    pmd=binary_dilation(pm,iterations=3)

    base_iou=(pmd&cloud).sum()/max((pmd|cloud).sum(),1)

    # 1) 单视角2D平移搜索 (像素平移点云mask, 看IoU上限)
    best=(base_iou,0,0)
    ys,xs=np.where(pm)
    for dx in range(-90,91,6):
        for dy in range(-90,91,6):
            sh=np.zeros((H,W),bool)
            ny=ys+dy; nx=xs+dx
            m=(ny>=0)&(ny<H)&(nx>=0)&(nx<W)
            sh[ny[m],nx[m]]=True
            sh=binary_dilation(sh,iterations=3)
            i=(sh&cloud).sum()/max((sh|cloud).sum(),1)
            if i>best[0]: best=(i,dx,dy)
    iou_shift=best[0]

    # 2) 缺失区(云有,膨胀点云没有) 的平均亮度 vs 已覆盖区亮度
    missing = cloud & (~pmd)
    covered = cloud & pmd
    miss_lum = lum[missing].mean() if missing.sum() else 0
    cov_lum  = lum[covered].mean() if covered.sum() else 0
    miss_frac = missing.sum()/max(cloud.sum(),1)

    # 3) 尺寸比
    if ok.sum() and cloud.sum():
        cw=xs.max()-xs.min(); ch=ys.max()-ys.min()
        cyy,cxx=np.where(cloud); clw=cxx.max()-cxx.min(); clh=cyy.max()-cyy.min()
        szr=( (px[ok].max()-px[ok].min())/max(clw,1), (py[ok].max()-py[ok].min())/max(clh,1) )
    else: szr=(0,0)

    return dict(ci=ci, base_iou=round(base_iou,3), iou_shift=round(iou_shift,3),
                best_shift=(best[1],best[2]), miss_frac=round(float(miss_frac),3),
                miss_lum=round(float(miss_lum),3), cov_lum=round(float(cov_lum),3),
                size_ratio=(round(szr[0],3),round(szr[1],3)))


print("视角  base_IoU  shift_IoU  best_shift(px)  miss_frac  miss_lum  cov_lum  size_ratio")
print("(shift_IoU>>base 且 best_shift大 => 残留平移可救; miss_lum<<cov_lum => 缺失区是暗部稀薄)")
for ci in [30, 10, 22, 34, 46, 26, 48]:
    r=analyze(ci)
    tag = "<-基线好" if ci==30 else ("<-异常" if ci in (10,22,34,46) else "")
    print(f"{r['ci']:>4}  {r['base_iou']:>7}  {r['iou_shift']:>8}  {str(r['best_shift']):>14}  "
          f"{r['miss_frac']:>8}  {r['miss_lum']:>7}  {r['cov_lum']:>7}  {str(r['size_ratio']):>12}  {tag}")
