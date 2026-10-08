# -*- coding: utf-8 -*-
"""在 DJI 覆盖区内找视图切换线最密处，输出 4x 放大对照，检验残留"发丝线"是否可见。
用法: python seam_probe.py [topN]
"""
import os
import sys
import json
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds

ROOT = r"D:/lr/project/无人机航拍拼接"
OUT = os.path.join(ROOT, "mysolution", "out")
DJI = os.path.join(ROOT, "xiangjia1", "map", "result.tif")
V3 = os.path.join(OUT, "xiangjia_ortho_mine.tif")
V2P = os.path.join(OUT, "xiangjia_ortho_v2pure.tif")
V1 = os.path.join(OUT, "xiangjia_ortho_v1.tif")
GSD = 0.055


def lab(img, t, h=32):
    bar = np.full((h, img.shape[1], 3), 22, np.uint8)
    cv2.putText(bar, t, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def grab(p, E0, N0, E1, N1, n, bilinear=True):
    with rasterio.open(p) as d:
        w = from_bounds(E0, N0, E1, N1, d.transform)
        nb = min(3, d.count)
        rs = rasterio.enums.Resampling.bilinear if bilinear else rasterio.enums.Resampling.average
        a = d.read(indexes=tuple(range(1, nb + 1)), window=w, out_shape=(nb, n, n),
                   resampling=rs)
    return np.stack([a[i] for i in (2, 1, 0)], 2)


def main():
    topn = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    om = json.load(open(os.path.join(OUT, "ortho_meta.json"), encoding="utf-8"))
    gx0, gy0 = om["x0"], om["y0"]
    vidx = np.load(os.path.join(OUT, "ortho_vidx.npy"), mmap_mode="r")
    vmask = np.load(os.path.join(OUT, "ortho_valid.npy"), mmap_mode="r")
    with rasterio.open(DJI) as d:
        db = d.bounds
    # DJI 覆盖区（内缩 120m，避开测区边缘候选视图稀少导致的碎片化选区）
    mg = 120.0
    c0 = int((db.left + mg - gx0) / GSD); c1 = int((db.right - mg - gx0) / GSD)
    r0 = int((gy0 - (db.top - mg)) / GSD);  r1 = int((gy0 - (db.bottom + mg)) / GSD)
    sub = np.array(vidx[r0:r1, c0:c1])
    subv = np.array(vmask[r0:r1, c0:c1])
    bx = np.zeros_like(subv)
    bx[:, 1:] |= (sub[:, 1:] != sub[:, :-1]) & subv[:, 1:] & subv[:, :-1]
    bx[1:, :] |= (sub[1:, :] != sub[:-1, :]) & subv[1:, :] & subv[:-1, :]
    # 用 41x41 窗口统计密度
    dens = cv2.boxFilter(bx.astype(np.float32), -1, (81, 81), normalize=True)
    print(f"切换线总长 {bx.sum()} px; 密度峰值 {dens.max()*81*81:.0f} px / 81x81 窗")

    picks = []
    d2 = dens.copy()
    for _ in range(topn):
        iy, ix = np.unravel_index(np.argmax(d2), d2.shape)
        E = gx0 + (c0 + ix) * GSD; N = gy0 - (r0 + iy) * GSD
        picks.append((E, N, float(d2[iy, ix])))
        y0 = max(0, iy - 400); y1 = min(d2.shape[0], iy + 400)
        x0 = max(0, ix - 400); x1 = min(d2.shape[1], ix + 400)
        d2[y0:y1, x0:x1] = 0
    rows = []
    for k, (E, N, dv) in enumerate(picks):
        print(f"#{k+1} E={E:.0f} N={N:.0f} 密度 {dv*81*81:.0f}px")
        half = 7.0
        n = 840
        imgs = [("MINE v3", V3), ("MINE v2 pure", V2P), ("MINE v1 avg", V1), ("DJI", DJI)]
        pan = []
        for nm, p in imgs:
            a = grab(p, E - half, N - half, E + half, N + half, n)
            a = cv2.resize(a, (n, n), interpolation=cv2.INTER_CUBIC)
            pan.append(lab(a, f"#{k+1} {nm}  x{n/(2*half/GSD):.1f}"))
        sep = np.full((pan[0].shape[0], 5, 3), 245, np.uint8)
        row = pan[0]
        for p in pan[1:]:
            row = np.concatenate([row, sep, p], axis=1)
        rows.append(row)
    gap = np.full((8, rows[0].shape[1], 3), 245, np.uint8)
    stack = rows[0]
    for r in rows[1:]:
        stack = np.concatenate([stack, gap, r], axis=0)
    if stack.shape[0] > 2500:
        s = 2500 / stack.shape[0]
        stack = cv2.resize(stack, (int(stack.shape[1] * s), 2500), interpolation=cv2.INTER_AREA)
    p = os.path.join(OUT, "fig_seam_probe.jpg")
    cv2.imencode(".jpg", stack, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(p)
    print("已写", p, stack.shape)


if __name__ == "__main__":
    main()
