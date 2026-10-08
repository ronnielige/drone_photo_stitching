# -*- coding: utf-8 -*-
"""在大疆 DSM 有效区内找显著建筑，对比两边 DSM 高程形态。"""
import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy import ndimage
from common import WORK, load_json

DJI_DSM = r"D:/lr/project/无人机航拍拼接/xiangjia1/map/dsm.tif"
OUT = {}

with rasterio.open(DJI_DSM) as ds:
    b = ds.bounds
    gd = ds.transform.a
    # 概览
    a = ds.read(1, out_shape=(2000, 2000), resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
    a[a <= 0] = np.nan
print("大疆 DSM bounds E[%.0f,%.0f] N[%.0f,%.0f] GSD %.3f" % (b.left, b.right, b.bottom, b.top, gd))
print("有效率 %.3f" % (np.isfinite(a).mean()))
ground = float(np.nanmedian(a))
print("地面中位 %.2f (正高)" % ground)

lab, nl = ndimage.label(np.isfinite(a) & (a > ground + 3.5))
sz = ndimage.sum(np.ones_like(lab), lab, range(1, nl + 1))
sx = (b.right - b.left) / 2000
sy = (b.top - b.bottom) / 2000
print("建筑候选簇 %d 个，前 10:" % nl)
order = np.argsort(-sz)[:10]
for k0 in order:
    ys, xs = np.nonzero(lab == k0 + 1)
    cy, cx = ys.mean(), xs.mean()
    E = b.left + (cx + 0.5) * sx
    N = b.top - (cy + 0.5) * sy
    zmax = np.nanmax(a[lab == k0 + 1])
    print("  簇%d: 面积 %.0f m2  E%.1f N%.1f  顶高 %.2f (高出地面 %.2f m)" % (
        k0 + 1, sz[k0] * sx * sy, E, N, zmax, zmax - ground))
    OUT[k0 + 1] = (E, N, sz[k0] * sx * sy, zmax)

# 选最大簇做逐格对比
k = order[0] + 1
E, N, area, zmax = OUT[k]
print("\n选定建筑: E=%.1f N=%.1f 面积 %.0f m2 顶高 %.2f" % (E, N, area, zmax))

half = 25.0
with rasterio.open(DJI_DSM) as ds:
    dji = ds.read(1, window=from_bounds(E - half, N - half, E + half, N + half, ds.transform),
                  out_shape=(150, 150), resampling=rasterio.enums.Resampling.bilinear).astype(np.float64)
dji[dji <= 0] = np.nan

dsm = np.load(os.path.join(WORK, "dsm.npy"))
dm = load_json(os.path.join(WORK, "dsm_meta.json"))
g = dm["gsd"]
X, Y = np.meshgrid(np.linspace(E - half, E + half, 150), np.linspace(N + half, N - half, 150))
fx = (X - dm["x0"]) / g - 0.5
fy = (dm["y0"] - Y) / g - 0.5
from scipy.ndimage import map_coordinates
mine = map_coordinates(dsm, [fy, fx], order=1, mode="nearest")

m = np.isfinite(dji) & np.isfinite(mine)
off = np.median(mine[m]) - np.median(dji[m])
print("基准偏移 (我的-大疆) = %.2f m" % off)
print("大疆: 地面 %.2f 顶 %.2f 高差 %.2f" % (
    np.nanpercentile(dji, 10), np.nanpercentile(dji, 99),
    np.nanpercentile(dji, 99) - np.nanpercentile(dji, 10)))
print("我的: 地面 %.2f 顶 %.2f 高差 %.2f" % (
    np.nanpercentile(mine - off, 10), np.nanpercentile(mine - off, 99),
    np.nanpercentile(mine - off, 99) - np.nanpercentile(mine - off, 10)))
d = (mine - off) - dji
print("逐格差值: 有效格 %d  平均 %.2f  std %.2f  p95|d| %.2f  max|d| %.2f" % (
    m.sum(), np.nanmean(d[m]), np.nanstd(d[m]),
    np.percentile(np.abs(d[m]), 95), np.abs(d[m]).max()))

np.save(os.path.join(WORK, "check_building_xy.npy"), np.array([E, N]))
print("\n建筑坐标已存 check_building_xy.npy: E=%.1f N=%.1f" % (E, N))
