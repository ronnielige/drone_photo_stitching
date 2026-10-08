# -*- coding: utf-8 -*-
"""Step5d: DSM v2 去尖刺 + 精度验证（对照大疆 DSM）。

去尖刺：与 9x9 局部中值偏离超过 max(2m, 6*MAD) 的格子用局部中值替换，
消除边缘弱支撑区的高程跳变（会造成正射局部"拉扯"）。
"""
import os
import sys
import numpy as np
from scipy.ndimage import median_filter, map_coordinates
import rasterio
from rasterio.windows import from_bounds
from common import WORK, DJI_DIR, load_json, save_json, log

DJI_DSM = os.path.join(DJI_DIR, "map", "dsm.tif")  # 可选：仅用于精度对照，不存在则自动跳过


def despike(z, size=9, thr=2.0, sig=6.0):
    med = median_filter(z, size=size)
    resid = z - med
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826
    t = max(thr, sig * mad)
    bad = np.abs(resid) > t
    out = z.copy()
    out[bad] = med[bad]
    return out, int(bad.sum()), float(t), float(mad)


def main():
    p = os.path.join(WORK, "dsm_v2.npy")
    z = np.load(p).astype(np.float32)
    m = load_json(os.path.join(WORK, "dsm_meta_v2.json"))
    g = m["gsd"]
    log(f"DSM v2 {z.shape} @ {g}m, 高程 [{np.nanmin(z):.2f},{np.nanmax(z):.2f}] "
        f"中位 {np.nanmedian(z):.2f}")
    q = np.nanpercentile(z, [0.1, 1, 50, 99, 99.9])
    log(f"分位 0.1/1/50/99/99.9%: {np.round(q,2)}")

    z2, nbad, t, mad = despike(z)
    log(f"去尖刺: MAD {mad:.3f}m 阈值 {t:.2f}m -> 替换 {nbad} 格 "
        f"({nbad/z.size*100:.4f}%)")
    z3, nbad2, _, _ = despike(z2)
    log(f"第二遍: 替换 {nbad2} 格")
    log(f"去尖刺后 高程 [{np.nanmin(z3):.2f},{np.nanmax(z3):.2f}] 中位 {np.nanmedian(z3):.2f}")
    np.save(p, z3.astype(np.float32))
    m["z_min"] = float(np.nanmin(z3))
    m["z_max"] = float(np.nanmax(z3))
    m["z_median"] = float(np.nanmedian(z3))
    m["despiked"] = True
    save_json(os.path.join(WORK, "dsm_meta_v2.json"), m)
    log("已保存 dsm_v2.npy (去尖刺)")

    # ---------------- 与大疆 DSM 对比精度 ----------------
    if not os.path.exists(DJI_DSM):
        return
    with rasterio.open(DJI_DSM) as dd:
        b = dd.bounds
        dgsd = dd.transform.a
        a = dd.read(1, out_shape=(1600, 1600),
                    resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
        a[a <= 0] = np.nan
    log(f"\n大疆 DSM {dgsd*100:.1f}cm, 有效 {np.isfinite(a).mean()*100:.1f}%, "
        f"中位 {np.nanmedian(a):.2f}(正高)")

    from scipy import ndimage
    # 在大疆有效区内找高起伏建筑块，逐块做形态对比
    gr = float(np.nanmedian(a))
    lab, nl = ndimage.label(np.isfinite(a) & (a > gr + 3.5))
    if nl == 0:
        return
    sz = ndimage.sum(np.ones_like(lab), lab, range(1, nl + 1))
    sx = (b.right - b.left) / 1600
    sy = (b.top - b.bottom) / 1600
    order = np.argsort(-sz)
    log("大疆有效区内显著建筑（前 6）:")
    zs = np.arange(z.shape[0]) * g
    xs = np.arange(z.shape[1]) * g
    done = 0
    for k0 in order:
        if sz[k0] * sx * sy < 300:
            break
        if done >= 6:
            break
        ys_l, xs_l = np.nonzero(lab == k0 + 1)
        E = b.left + (xs_l.mean() + 0.5) * sx
        N = b.top - (ys_l.mean() + 0.5) * sy
        half = 30.0
        if not (m["x0"] + 40 < E < m["x0"] + z.shape[1] * g - 40 and
                m["y0"] - z.shape[0] * g + 40 < N < m["y0"] - 40):
            continue
        n = 120
        gx = np.linspace(E - half, E + half, n)
        gy = np.linspace(N + half, N - half, n)
        X, Y = np.meshgrid(gx, gy)
        fx = (X - m["x0"]) / g - 0.5
        fy = (m["y0"] - Y) / g - 0.5
        mine = map_coordinates(z3, [fy, fx], order=1, mode="nearest")
        with rasterio.open(DJI_DSM) as dd:
            dji = dd.read(1, window=from_bounds(E - half, N - half, E + half, N + half,
                                                dd.transform),
                          out_shape=(n, n),
                          resampling=rasterio.enums.Resampling.bilinear).astype(np.float64)
        dji[dji <= 0] = np.nan
        mm = np.isfinite(dji) & np.isfinite(mine)
        if mm.sum() < n * n * 0.9:
            continue
        off = np.median(mine[mm]) - np.median(dji[mm])
        d = (mine - off) - dji
        # 平坦子区（局部起伏小）的噪声
        sm = median_filter(mine, 5)
        flat = mm & (np.abs(mine - sm) < 0.15)
        log(f"  E{E:.0f} N{N:.0f} 面积{sz[k0]*sx*sy:.0f}m2 | 基准偏移 {off:+.2f}m | "
            f"差值 std {np.nanstd(d[mm]):.2f}m p95|d| {np.nanpercentile(np.abs(d[mm]),95):.2f}m "
            f"| 平坦区 std {np.nanstd(d[flat]) if flat.sum()>100 else float('nan'):.2f}m")
        done += 1


if __name__ == "__main__":
    main()
