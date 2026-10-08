# -*- coding: utf-8 -*-
"""按地理范围裁剪对照：我的 v2 / 我的 v1 / 大疆，三联输出。
用法: python crop_cmp.py E0 N0 E1 N1 [outname] [maxpx]
注意 E0<E1, N0<N1。
"""
import os
import sys
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds

ROOT = r"D:/lr/project/无人机航拍拼接"
OUT = os.path.join(ROOT, "mysolution", "out")
DJI = os.path.join(ROOT, "xiangjia1", "map", "result.tif")
MINE = os.path.join(OUT, "xiangjia_ortho_mine.tif")
V2P = os.path.join(OUT, "xiangjia_ortho_v2pure.tif")
V1 = os.path.join(OUT, "xiangjia_ortho_v1.tif")


def grab(path, E0, N0, E1, N1, maxpx=1400):
    with rasterio.open(path) as d:
        win = from_bounds(E0, N0, E1, N1, d.transform)
        nb = min(3, d.count)
        a = d.read(indexes=tuple(range(1, nb + 1)), window=win)
        av = None
        if d.count >= 4:
            av = d.read(indexes=4, window=win) > 128
    bgr = np.stack([a[i] for i in (2, 1, 0)], axis=2)
    h, w = bgr.shape[:2]
    if max(h, w) > maxpx:
        s = maxpx / float(max(h, w))
        bgr = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
        if av is not None:
            av = cv2.resize(av.astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
                            interpolation=cv2.INTER_NEAREST) > 0
    return bgr, av


def lab(img, t, h=30):
    bar = np.full((h, img.shape[1], 3), 25, np.uint8)
    cv2.putText(bar, t, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def pad(a, H, W):
    o = np.zeros((H, W, 3), np.uint8)
    h = min(H, a.shape[0]); w = min(W, a.shape[1])
    o[:h, :w] = a[:h, :w]
    return o


def main():
    E0, N0, E1, N1 = (float(sys.argv[i]) for i in range(1, 5))
    name = sys.argv[5] if len(sys.argv) > 5 else "crop_cmp"
    maxpx = int(sys.argv[6]) if len(sys.argv) > 6 else 1400
    mine, _ = grab(MINE, E0, N0, E1, N1, maxpx)
    dji, dv = grab(DJI, E0, N0, E1, N1, maxpx)
    v1, _ = grab(V1, E0, N0, E1, N1, maxpx)
    v2p, _ = grab(V2P, E0, N0, E1, N1, maxpx)
    H = max(mine.shape[0], dji.shape[0], v1.shape[0], v2p.shape[0])
    W = max(mine.shape[1], dji.shape[1], v1.shape[1], v2p.shape[1])
    mine = pad(mine, H, W); dji = pad(dji, H, W)
    v1 = pad(v1, H, W); v2p = pad(v2p, H, W)
    row = np.concatenate([lab(mine, "MINE v3 = single-frame detail + LF tone consensus"),
                          lab(v2p, "MINE v2 = pure hard single-frame"),
                          lab(v1, "MINE v1 = multi-view average"),
                          lab(dji, "DJI Terra")], axis=1)
    p = os.path.join(OUT, name + ".jpg")
    cv2.imencode(".jpg", row, [int(cv2.IMWRITE_JPEG_QUALITY), 93])[1].tofile(p)
    print(f"{name}.jpg  {row.shape[1]}x{row.shape[0]}  "
          f"({(E1-E0):.0f}x{(N1-N0):.0f} m, 每联 {W}x{H}px)")


if __name__ == "__main__":
    main()
