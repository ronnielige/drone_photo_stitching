# -*- coding: utf-8 -*-
"""生成最终交付对照图：
  fig_final_full.jpg      全图 v3 vs 大疆（含各自的覆盖范围）
  fig_blocks_beforeafter  「色块伪影」前后对照（v2纯单帧 -> v3）+ 大疆参考
  fig_seam_zoom.jpg       视图切换线处放大：v3 / v2纯单帧 / 大疆（检验残留"发丝线"）
用法: python make_figs.py
"""
import os
import json
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

ROOT = r"D:/lr/project/无人机航拍拼接"
OUT = os.path.join(ROOT, "mysolution", "out")
DJI = os.path.join(ROOT, "xiangjia1", "map", "result.tif")
V3 = os.path.join(OUT, "xiangjia_ortho_mine.tif")
V2P = os.path.join(OUT, "xiangjia_ortho_v2pure.tif")
V1 = os.path.join(OUT, "xiangjia_ortho_v1.tif")
GSD = 0.055


def lab(img, t, h=34, fs=0.78):
    bar = np.full((h, img.shape[1], 3), 22, np.uint8)
    cv2.putText(bar, t, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2,
                cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def row_of(panels, gap=6, val=245):
    sep = np.full((panels[0].shape[0], gap, 3), val, np.uint8)
    r = panels[0]
    for p in panels[1:]:
        r = np.concatenate([r, sep, p], axis=1)
    return r


def read_common(paths, E0, N0, E1, N1, maxpx):
    out = []
    for p in paths:
        with rasterio.open(p) as d:
            win = from_bounds(E0, N0, E1, N1, d.transform)
            nb = min(3, d.count)
            a = d.read(indexes=tuple(range(1, nb + 1)), window=win,
                       out_shape=(nb, maxpx[1], maxpx[0]), resampling=Resampling.bilinear)
            av = None
            if d.count >= 4:
                av = d.read(indexes=4, window=win, out_shape=(maxpx[1], maxpx[0]),
                            resampling=Resampling.nearest) > 128
        bgr = np.stack([a[i] for i in (2, 1, 0)], axis=2)
        if av is not None:
            bgr = bgr.copy()
            bgr[~av] = 255
        out.append(bgr)
    return out


def main():
    with rasterio.open(V3) as d:
        mb = d.bounds
    with rasterio.open(DJI) as d:
        db = d.bounds
    E0, E1 = max(mb.left, db.left), min(mb.right, db.right)
    N0, N1 = max(mb.bottom, db.bottom), min(mb.top, db.top)
    w, h = 1500, int(1500 * (N1 - N0) / (E1 - E0))
    v3, dji = read_common([V3, DJI], E0, N0, E1, N1, (w, h))
    fig = row_of([lab(v3, "MINE v3  (5.50 cm, single-frame detail + LF tone consensus)"),
                  lab(dji, "DJI Terra  (5.62 cm, reference)")], gap=8)
    cv2.imencode(".jpg", fig, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(
        os.path.join(OUT, "fig_final_full.jpg"))
    print("fig_final_full.jpg", fig.shape)

    # ---- 色块伪影 前后对照 ----
    E0b, N1b, E1b, N0b = 562100, 3324890, 562450, 3324580
    n = 1350
    p2, p3, pd = read_common([V2P, V3, DJI], E0b, N0b, E1b, N1b, (n, n))
    fig2 = row_of([lab(p2, "BEFORE  MINE v2: pure hard single-frame"),
                   lab(p3, "AFTER   MINE v3: single-frame detail + LF tone consensus"),
                   lab(pd, "DJI Terra (reference)")])
    cv2.imencode(".jpg", fig2, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(
        os.path.join(OUT, "fig_blocks_beforeafter.jpg"))
    print("fig_blocks_beforeafter.jpg", fig2.shape)

    # ---- 切换线放大检查 ----
    vidx = np.load(os.path.join(OUT, "ortho_vidx.npy"), mmap_mode="r")
    meta = json.load(open(os.path.join(OUT, "ortho_meta.json"), encoding="utf-8"))
    gx0, gy0 = meta["x0"], meta["y0"]
    # 在雾区范围内找一条切换线最密集的行（限制在大疆覆盖区内部）
    c0, c1 = int((562050 - gx0) / GSD), int((562500 - gx0) / GSD)
    r0, r1 = int((gy0 - 3324790) / GSD), int((gy0 - 3324310) / GSD)
    sub = np.array(vidx[r0:r1:2, c0:c1:2])
    ch = ((sub[:, 1:] != sub[:, :-1]) & (sub[:, 1:] >= 0) & (sub[:, :-1] >= 0)).sum(1)
    by = int(np.argmax(ch))
    Ey = 3324790 - (r0 + by * 2) * GSD
    chc = ((sub[1:, :] != sub[:-1, :]) & (sub[1:, :] >= 0) & (sub[:-1, :] >= 0)).sum(0)
    bx = int(np.argmax(chc))
    Ex = 562050 + (c0 + bx * 2) * GSD
    print(f"切换线最密处 E={Ex:.0f} N={Ey:.0f} (横 {ch.max()} 竖 {chc.max()} 个变化点)")
    half = 9.0
    n2 = 900
    Ec = Ex
    p2b, p3b, pdb = read_common([V2P, V3, DJI], Ec - half, Ey - half, Ec + half, Ey + half,
                                (n2, n2))
    fig3 = row_of([lab(p3b, f"AFTER  v3  x{n2/(2*half/GSD):.1f}"),
                   lab(p2b, "BEFORE  v2 pure single-frame"),
                   lab(pdb, "DJI")])
    cv2.imencode(".jpg", fig3, [int(cv2.IMWRITE_JPEG_QUALITY), 93])[1].tofile(
        os.path.join(OUT, "fig_seam_zoom.jpg"))
    print("fig_seam_zoom.jpg", fig3.shape)


if __name__ == "__main__":
    main()
