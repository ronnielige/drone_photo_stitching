# -*- coding: utf-8 -*-
"""DSM 可视化对照：我的 v2 / 我的 v1 / 大疆（同一窗口，去掉各自中位偏移）。"""
import os
import sys
import json
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, OUT, load_json, imwrite_any

DJI_DSM = r"D:/lr/project/无人机航拍拼接/xiangjia1/map/dsm.tif"


def sample(z, meta, X, Y):
    g = meta["gsd"]
    fx = (X - meta["x0"]) / g - 0.5
    fy = (meta["y0"] - Y) / g - 0.5
    return map_coordinates(z, [fy, fx], order=1, mode="nearest")


def colorize(a, vmin, vmax, mask=None):
    a = np.clip((np.nan_to_num(a, nan=vmin) - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    col = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    if mask is not None:
        col[~mask] = (0, 0, 0)
    return col


def hill(a):
    gy, gx = np.gradient(np.nan_to_num(a, nan=np.nanmedian(a)))
    hs = 0.5 + 0.5 * (gx * 3 - gy * 3) / (np.abs(gx * 3).max() + 1e-6)
    h = np.clip(hs, 0, 1)
    return cv2.cvtColor((h * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def label(img, txt, h=34):
    bar = np.full((h, img.shape[1], 3), 25, np.uint8)
    cv2.putText(bar, txt, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def main():
    E = float(sys.argv[1]) if len(sys.argv) > 1 else 562445.0
    N = float(sys.argv[2]) if len(sys.argv) > 2 else 3324746.0
    half = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
    n = 800

    z2 = np.load(os.path.join(WORK, "dsm_v2.npy"))
    m2 = load_json(os.path.join(WORK, "dsm_meta_v2.json"))
    z1 = np.load(os.path.join(WORK, "dsm.npy"))
    m1 = load_json(os.path.join(WORK, "dsm_meta.json"))
    X, Y = np.meshgrid(np.linspace(E - half, E + half, n), np.linspace(N + half, N - half, n))
    v2 = sample(z2, m2, X, Y)
    v1 = sample(z1, m1, X, Y)
    with rasterio.open(DJI_DSM) as dd:
        dj = dd.read(1, window=from_bounds(E - half, N - half, E + half, N + half, dd.transform),
                     out_shape=(n, n), resampling=rasterio.enums.Resampling.bilinear).astype(np.float64)
    dj[dj <= 0] = np.nan

    for nm, v in (("v2", v2), ("v1", v1), ("dji", dj)):
        print(f"{nm}: p1 {np.nanpercentile(v,1):.2f} p50 {np.nanmedian(v):.2f} "
              f"p99 {np.nanpercentile(v,99):.2f} std {np.nanstd(v):.2f} nan% {np.isnan(v).mean()*100:.1f}")

    # 去掉各自中位（基准差），统一色标
    o2 = np.nanmedian(v2); o1 = np.nanmedian(v1); od = np.nanmedian(dj)
    a2, a1, ad = v2 - o2, v1 - o1, dj - od
    vmin, vmax = -1.0, 14.0
    m = np.isfinite(a2)
    p1_ = label(colorize(a2, vmin, vmax, m), f"MINE v2 DSM(0.1m, despeckled) E{E:.0f} N{N:.0f}")
    p2_ = label(colorize(a1, vmin, vmax, np.isfinite(a1)), "MINE v1 DSM(0.25m)")
    p3_ = label(colorize(ad, vmin, vmax, np.isfinite(ad)), "DJI DSM(0.113m)")
    p4_ = label(hill(a1), "v1 hillshade")
    p5_ = label(hill(a2), "v2 hillshade")
    p6_ = label(hill(ad), "DJI hillshade")
    top = np.concatenate([p1_, p2_, p3_], axis=1)
    bot = np.concatenate([p4_, p5_, p6_], axis=1)
    canvas = np.concatenate([top, np.full((8, top.shape[1], 3), 250, np.uint8), bot], axis=0)
    out = os.path.join(OUT, f"dsm_view_{E:.0f}_{N:.0f}.jpg")
    imwrite_any(out, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print("saved", out, canvas.shape)

    # 剖面线（穿过建筑）
    r = n // 2
    pr = np.stack([a1[r], a2[r], ad[r]], 1)
    print("剖面(row=mid) 前20列 v1/v2/dji:", np.round(pr[:20], 2).tolist())


if __name__ == "__main__":
    main()
