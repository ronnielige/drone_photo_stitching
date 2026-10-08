# -*- coding: utf-8 -*-
"""批量渲染候选建筑点：单视 vs 多视 vs 大疆，contact sheet。"""
import os, sys, json, math
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, OUT, load_json, imwrite_any, Camera
from step6_ortho import _render_tile, project_world, sample_dsm, ortho_pixel_xyz

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
ortho_ids = [i for i in range(n) if -Rs[i][2, 2] > 0.85]

dd = rasterio.open(r"D:/lr/project/无人机航拍拼接/xiangjia1/map/result.tif")
dji_bounds = dd.bounds

def dji_valid(E, N, half=16.0):
    w = from_bounds(E - half, N - half, E + half, N + half, dd.transform)
    a = dd.read(indexes=4, window=w)
    return a.size > 0 and (a > 128).mean() > 0.98

def cover_views(E, N):
    Xp, Yp = ortho_pixel_xyz(E - 0.5, N + 0.5, 1.0, 3, 3)
    Zp = sample_dsm(dsm, dmeta, Xp, Yp)
    Xw_c = np.stack([Xp.ravel(), Yp.ravel(), Zp.ravel()], 1)
    out = []
    for i in ortho_ids:
        u, v, nadir, front = project_world(Xw_c, Rs[i], Cs[i], cam)
        if front.all() and (u > 100).all() and (u < cam.w - 100).all() and \
           (v > 100).all() and (v < cam.h - 100).all():
            dv = Xw_c - Cs[i][None, :]
            cosn = -dv[:, 2] / np.linalg.norm(dv, axis=1)
            out.append((float(cosn.min()), i))
    out.sort(reverse=True)
    return [i for _, i in out[:8]], (out[0][1] if out else None)

def render(E, N, views, gsd=0.05, size=400):
    gx0 = E - size / 2 * gsd
    gy0 = N + size / 2 * gsd
    args = (0, 0, gx0, gy0, gsd, size, size, views, cam_p, Rs, Cs, paths, gains,
            os.path.join(WORK, "dsm.npy"), dmeta, os.path.join(WORK, "dsm_conf.npy"))
    _, _, blk, vmask = _render_tile(args)
    return blk

ground = float(np.nanmedian(dsm))
lab, nl = ndimage.label(~np.isnan(dsm) & (dsm > ground + 4.0))
sizes = ndimage.sum(np.ones_like(lab), lab, range(1, nl + 1))
dg = dmeta["gsd"]

# 候选：真建筑(非饱和)、大疆有效、彼此距离>80m、面积 80-2000 m²
cands = []
for k0 in np.argsort(-sizes)[:800]:
    k = int(k0) + 1
    a_m2 = sizes[k - 1] * dg * dg
    if a_m2 < 80:
        break
    ys, xs = np.nonzero(lab == k)
    zmax_k = float(np.nanmax(dsm[lab == k]))
    if zmax_k >= 156.5:
        continue
    E_k = dmeta["x0"] + (xs.mean() + 0.5) * dg
    N_k = dmeta["y0"] - (ys.mean() + 0.5) * dg
    if not dji_valid(E_k, N_k):
        continue
    if any(np.hypot(E_k - e, N_k - nn) < 80 for e, nn in cands):
        continue
    cands.append((E_k, N_k))
    if len(cands) >= 6:
        break
print("候选点:", [(round(e), round(nn)) for e, nn in cands])

tiles = []
for E_k, N_k in cands:
    views, best = cover_views(E_k, N_k)
    single = render(E_k, N_k, [best])
    multi = render(E_k, N_k, views)
    w = from_bounds(E_k - 10, N_k - 10, E_k + 10, N_k + 10, dd.transform)
    a = dd.read(window=w)
    dji = np.moveaxis(a[:3], 0, 2)[:, :, ::-1]
    dji = cv2.resize(dji, (400, 400), interpolation=cv2.INTER_AREA)
    row = np.concatenate([single, np.full((400, 3, 3), 240, np.uint8), multi,
                          np.full((400, 3, 3), 240, np.uint8), dji], axis=1)
    tiles.append(row)
sheet = np.concatenate(tiles, axis=0)
imwrite_any(os.path.join(OUT, "diag_candidates.jpg"), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
print("saved diag_candidates.jpg", sheet.shape)
