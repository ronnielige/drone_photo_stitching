# -*- coding: utf-8 -*-
"""比较不同低频尺度 LF_M 的混合结果（色块/锐度权衡）。
用法: python cmp_lf.py E N half_m
"""
import os
import sys
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds

ROOT = r"D:/lr/project/无人机航拍拼接"
OUT = os.path.join(ROOT, "mysolution", "out")
ITEMS = [("v3 LF1.3", os.path.join(OUT, "xiangjia_ortho_mine.tif")),
         ("v3 LF2.5", os.path.join(OUT, "test_lf25.tif")),
         ("v3 LF4.0", os.path.join(OUT, "test_lf40.tif")),
         ("v2 pure", os.path.join(OUT, "xiangjia_ortho_v2pure.tif")),
         ("DJI", os.path.join(ROOT, "xiangjia1", "map", "result.tif"))]


def lab(img, t, h=30):
    bar = np.full((h, img.shape[1], 3), 22, np.uint8)
    cv2.putText(bar, t, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2,
                cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def grab(p, E0, N0, E1, N1, n):
    with rasterio.open(p) as d:
        w = from_bounds(E0, N0, E1, N1, d.transform)
        nb = min(3, d.count)
        a = d.read(indexes=tuple(range(1, nb + 1)), window=w, out_shape=(nb, n, n),
                   resampling=rasterio.enums.Resampling.bilinear)
    return np.stack([a[i] for i in (2, 1, 0)], 2)


def main():
    E, N, half = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    n = 780
    pan = [lab(grab(p, E - half, N - half, E + half, N + half, n), t)
           for t, p in ITEMS]
    sep = np.full((pan[0].shape[0], 5, 3), 245, np.uint8)
    row = pan[0]
    for p in pan[1:]:
        row = np.concatenate([row, sep, p], axis=1)
    row = cv2.resize(row, (row.shape[1], row.shape[0]), interpolation=cv2.INTER_AREA)
    p = os.path.join(OUT, "cmp_lf.jpg")
    cv2.imencode(".jpg", row, [int(cv2.IMWRITE_JPEG_QUALITY), 93])[1].tofile(p)
    print("已写", p, row.shape)


if __name__ == "__main__":
    main()
