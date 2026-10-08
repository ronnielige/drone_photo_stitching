# -*- coding: utf-8 -*-
"""诊断：单视图 vs 多视融合 vs 大疆，同一建筑区域对比。"""
import os, sys, json, math
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, OUT, load_json, imread_any, imwrite_any
from step6_ortho import _render_tile, image_ortho_bbox
from common import Camera

meta = load_json(os.path.join(WORK, "meta.json"))
cj = load_json(os.path.join(WORK, "camera.json"))
cam = Camera(cj["w"], cj["h"], *cj["init"])
cam_p = (cj["w"], cj["h"], *cj["init"])
paths = [m["path"] for m in meta]
grid = load_json(os.path.join(WORK, "grid.json"))
dmeta = load_json(os.path.join(WORK, "dsm_meta.json"))
Rs = np.load(os.path.join(WORK, "ba_R.npy"))
Cs = np.load(os.path.join(WORK, "ba_C.npy"))
gains = np.load(os.path.join(WORK, "gains.npy"))
dsm = np.load(os.path.join(WORK, "dsm.npy"))
n = len(meta)

# ---- 1. 找最大建筑群（Z > ground+4 的最大连通域，且在大疆覆盖交集内）----
import rasterio as _rio
_dd = _rio.open(r"D:/lr/project/无人机航拍拼接/xiangjia1/map/result.tif")
dji_bounds = _dd.bounds  # left bottom right top

def _dji_valid_around(E, N, half=16.0):
    from rasterio.windows import from_bounds as _fb
    w = _fb(E - half, N - half, E + half, N + half, _dd.transform)
    a = _dd.read(indexes=4, window=w)
    return a.size > 0 and (a > 128).mean() > 0.98

ground = float(np.nanmedian(dsm))
lab, nl = ndimage.label(~np.isnan(dsm) & (dsm > ground + 4.0))
sizes = ndimage.sum(np.ones_like(lab), lab, range(1, nl + 1))
order = np.argsort(-sizes)
dg = dmeta["gsd"]
z_sat = 156.5   # DSM 搜索层顶部饱和值，排除匹配失败的假高点
found = False
for k0 in order[:400]:
    k = int(k0) + 1
    if sizes[k - 1] * dg * dg < 100:
        break
    ys, xs = np.nonzero(lab == k)
    zmax_k = float(np.nanmax(dsm[lab == k]))
    if zmax_k >= z_sat:
        continue
    E_k = dmeta["x0"] + (xs.mean() + 0.5) * dg
    N_k = dmeta["y0"] - (ys.mean() + 0.5) * dg
    if _dji_valid_around(E_k, N_k):
        E_build, N_build = E_k, N_k
        z_build = zmax_k
        area = sizes[k - 1] * dg * dg
        found = True
        break
assert found, "没有找到交集内建筑"
print(f"建筑群: E={E_build:.1f} N={N_build:.1f} 面积={area:.0f}m2 顶高={z_build:.1f} 地面={ground:.1f}")

# ---- 2. 选覆盖该点的视图（用真实投影，而非 bbox）----
from step6_ortho import project_world, sample_dsm, ortho_pixel_xyz
ortho_ids = [i for i in range(n) if -Rs[i][2, 2] > 0.85]
Xp, Yp = ortho_pixel_xyz(E_build - 0.5, N_build + 0.5, 1.0, 3, 3)
Zp = sample_dsm(dsm, dmeta, Xp, Yp)
Xw_c = np.stack([Xp.ravel(), Yp.ravel(), Zp.ravel()], 1)
cover = []
for i in ortho_ids:
    u, v, nadir, front = project_world(Xw_c, Rs[i], Cs[i], cam)
    if front.all() and (u > 100).all() and (u < cam.w - 100).all() and \
       (v > 100).all() and (v < cam.h - 100).all():
        dv = Xw_c - Cs[i][None, :]
        cosn = -dv[:, 2] / np.linalg.norm(dv, axis=1)
        cover.append((float(cosn.min()), i))
cover.sort(reverse=True)
print("覆盖视图 (cosn, idx):")
for c, i in cover[:8]:
    print(f"  {c:.3f} #{i}")

best = cover[0][1]
top8 = [i for _, i in cover[:8]]

# ---- 3. 渲染 600x600 @5cm (30m) ----
def render(views, gsd=0.05, size=600):
    gx0 = E_build - size / 2 * gsd
    gy0 = N_build + size / 2 * gsd
    args = (0, 0, gx0, gy0, gsd, size, size, views, cam_p, Rs, Cs, paths, gains,
            os.path.join(WORK, "dsm.npy"), dmeta, os.path.join(WORK, "dsm_conf.npy"))
    _, _, blk, vmask = _render_tile(args)
    return blk, vmask

single, _ = render([best])
multi, _ = render(top8)

# ---- 4. 大疆同区域 ----
half_m = 15.0
with rasterio.open(r"D:/lr/project/无人机航拍拼接/xiangjia1/map/result.tif") as dd:
    gsd_d = dd.transform.a; x0d = dd.transform.c; y0d = dd.transform.f
    win = from_bounds(E_build - half_m, N_build - half_m, E_build + half_m, N_build + half_m, dd.transform)
    a = dd.read(window=win)
    if a.shape[0] == 4:
        al = a[3]
    else:
        al = np.full(a.shape[1:], 255, np.uint8)
    dji_rgb = np.moveaxis(a[:3], 0, 2)
dji_bgr = dji_rgb[:, :, ::-1].copy()  # RGB -> BGR
wpx = dji_bgr.shape[1]
print("dji crop", dji_bgr.shape, "gsd %.4f" % gsd_d)

# ---- 5. 指标 ----
def lap(a):
    g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.Laplacian(g, cv2.CV_32F).var()

def edge_strength(a):
    g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3).var() + cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3).var()) / 2

print()
print(f"拉普拉斯方差  单视图: {lap(single):.0f}   多视融合: {lap(multi):.0f}   大疆@5.62cm: {lap(dji_bgr):.0f}")
print(f"Sobel梯度方差 单视图: {edge_strength(single):.0f}   多视融合: {edge_strength(multi):.0f}   大疆: {edge_strength(dji_bgr):.0f}")

# 大疆缩到 5cm 供公平数值对比
dji_r = cv2.resize(dji_bgr, (600, 600), interpolation=cv2.INTER_AREA)
print(f"大疆重采样到5cm后 拉普拉斯: {lap(dji_r):.0f}")

# ---- 6. 拼图保存（上：单视/多视/大疆，下：各放大2倍中心300x300）----
dji_show = cv2.resize(dji_bgr, (600, 600), interpolation=cv2.INTER_AREA)
h, w = 600, 600
sep = np.full((h, 4, 3), 240, np.uint8)
top = np.concatenate([single, sep, multi, sep, dji_show], axis=1)
def zoom2(a):
    c = a[150:450, 150:450]
    return cv2.resize(c, (600, 600), interpolation=cv2.INTER_NEAREST)
bot = np.concatenate([zoom2(single), sep, zoom2(multi), sep, zoom2(dji_show)], axis=1)
canvas = np.concatenate([top, np.full((4, top.shape[1], 3), 240, np.uint8), bot], axis=0)
imwrite_any(os.path.join(OUT, "diag_building.jpg"), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
print("已保存 diag_building.jpg  (上: 单视|多视|大疆  下:中心2x放大)")

# ---- 7. 拼接痕迹检测：多视融合里的水平/垂直梯度异常线 ----
g = cv2.cvtColor(multi, cv2.COLOR_BGR2GRAY).astype(np.float32)
gx = np.abs(np.diff(g, axis=1)).mean(axis=0)   # 每列平均梯度
gy = np.abs(np.diff(g, axis=0)).mean(axis=1)
print(f"列梯度: p50={np.percentile(gx,50):.2f} p99={np.percentile(gx,99):.2f} max={gx.max():.2f} @col{gx.argmax()}")
print(f"行梯度: p50={np.percentile(gy,50):.2f} p99={np.percentile(gy,99):.2f} max={gy.max():.2f} @row{gy.argmax()}")
