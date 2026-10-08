# -*- coding: utf-8 -*-
"""局部窗口锐度/色块量化：对若干地理窗口，比较 v3 / v2pure / v1 / DJI。
用法: python eval_windows.py
"""
import os
import sys
import json
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step8c_evaluate import edge_sharpness, gray, lp_block_metric, seam_bands

ROOT = r"D:/lr/project/无人机航拍拼接"
OUT = os.path.join(ROOT, "mysolution", "out")
PATHS = {"v3": os.path.join(OUT, "xiangjia_ortho_mine.tif"),
         "v2pure": os.path.join(OUT, "xiangjia_ortho_v2pure.tif"),
         "v1": os.path.join(OUT, "xiangjia_ortho_v1.tif"),
         "DJI": os.path.join(ROOT, "xiangjia1", "map", "result.tif")}
GSD = 0.055


def grab(path, E0, N1, half):
    E1, N0 = E0 + 2 * half, N1 - 2 * half
    with rasterio.open(path) as d:
        win = from_bounds(E0, N0, E1, N1, d.transform)
        nb = min(3, d.count)
        a = d.read(indexes=tuple(range(1, nb + 1)), window=win)
        av = d.read(indexes=4, window=win) > 128 if d.count >= 4 else None
    n = int(round(2 * half / GSD))
    it = cv2.INTER_AREA if half > 25 else cv2.INTER_LINEAR
    bgr = np.stack([cv2.resize(a[i], (n, n), interpolation=it) for i in (2, 1, 0)], axis=2)
    if av is not None:
        av = cv2.resize(av.astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST) > 0
    return bgr, av


def main():
    wins = [("瓦屋面#2", 562501, 3324402, 20),
            ("村庄#4", 562008, 3324441, 20),
            ("耕地#1", 562336, 3324559, 20),
            ("梯田顶#3", 562446, 3324756, 20),
            ("长厂房", 562360, 3324330, 25)]
    print(f"{'窗口':10s} {'方案':8s} {'边缘锐度':>8s} {'LP梯度p95':>9s} {'灰度':>7s}")
    rows = {}
    for tag, E, N, half in wins:
        for nm, p in PATHS.items():
            if not os.path.exists(p):
                continue
            im, av = grab(p, E, N, half)
            m = av if av is not None else np.ones(im.shape[:2], bool)
            g = gray(im)
            es = edge_sharpness(g, m)
            lap = cv2.Laplacian(g, cv2.CV_32F)
            lp = np.hypot(cv2.Sobel(cv2.GaussianBlur(g, (0, 0), 1.5 / GSD), cv2.CV_32F, 1, 0, 3),
                          cv2.Sobel(cv2.GaussianBlur(g, (0, 0), 1.5 / GSD), cv2.CV_32F, 0, 1, 3))
            rows.setdefault(nm, []).append(es)
            print(f"{tag:10s} {nm:8s} {es:8.3f} {np.percentile(lp[m],95):9.2f} {g[m].mean():7.1f}"
                  f"   lapvar {lap[m].var():7.0f}")
        print()
    print("窗口平均边缘锐度:", {k: round(float(np.mean(v)), 3) for k, v in rows.items()})


if __name__ == "__main__":
    main()
