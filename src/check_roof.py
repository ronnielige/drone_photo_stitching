# -*- coding: utf-8 -*-
"""对照大疆 DSM 检查建筑屋顶真实高程（基准差 = 常数偏移）。"""
import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from common import WORK, load_json

DJI_DSM = r"D:/lr/project/无人机航拍拼接/xiangjia1/map/dsm.tif"
E, N = 562543.0, 3324779.0

dsm = np.load(os.path.join(WORK, "dsm.npy"))
dm = load_json(os.path.join(WORK, "dsm_meta.json"))
g = dm["gsd"]
xs = np.arange(dsm.shape[1]) * g + dm["x0"]
ys = dm["y0"] - np.arange(dsm.shape[0]) * g
m = (np.abs(xs - E) < 20)[None, :] & (np.abs(ys - N) < 20)[:, None]
mine = dsm[m]

with rasterio.open(DJI_DSM) as ds:
    print("DJI DSM: %.3fm GSD, dtype %s, count %d" % (ds.transform.a, ds.dtypes[0], ds.count))
    a = ds.read(1, window=from_bounds(E - 20, N - 20, E + 20, N + 20, ds.transform),
                out_shape=(200, 200), resampling=rasterio.enums.Resampling.bilinear).astype(np.float64)
print("DJI DSM 窗口: p50 %.2f p90 %.2f p99 %.2f max %.2f" % (
    np.percentile(a, 50), np.percentile(a, 90), np.percentile(a, 99), a.max()))
print("我的 DSM 窗口: p50 %.2f p90 %.2f p99 %.2f max %.2f" % (
    np.percentile(mine, 50), np.percentile(mine, 90), np.percentile(mine, 99), mine.max()))

off = np.median(mine) - np.median(a)
print("基准偏移(我的-大疆) = %.2f m  -> 抵消后对比" % off)
print("我的 (减偏移): p90 %.2f p99 %.2f max %.2f" % (
    np.percentile(mine, 90) - off, np.percentile(mine, 99) - off, mine.max() - off))

# 地面（低分位）与屋顶（高分位）高差
print("\n地面/屋顶高差（相对各自地面）:")
print("  大疆 DOM: 地面 p10 %.2f 屋顶 p99 %.2f 高差 %.2f m" % (
    np.percentile(a, 10), np.percentile(a, 99), np.percentile(a, 99) - np.percentile(a, 10)))
print("  我的:     地面 p10 %.2f 屋顶 p99 %.2f 高差 %.2f m" % (
    np.percentile(mine, 10), np.percentile(mine, 99),
    np.percentile(mine, 99) - np.percentile(mine, 10)))

# 同窗口逐格比较（重采样到 100x100）
from scipy.ndimage import zoom
mm = zoom(mine.reshape(int(m.sum() / m.shape[1]), m.shape[1]), (100 / (m.sum() / m.shape[1]), 100 / m.shape[1]), order=1)
aa = zoom(a, (100 / a.shape[0], 100 / a.shape[1]), order=1)
d = mm - aa - off
print("\n逐格差值: mean %.2f std %.2f p95|d| %.2f max|d| %.2f" % (
    d.mean(), d.std(), np.percentile(np.abs(d), 95), np.abs(d).max()))
