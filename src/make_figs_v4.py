# -*- coding: utf-8 -*-
"""最终交付对照图（v4 = 接缝线优化版）。

产出：
  fig_final_v4_full.jpg     全图 v3 vs v4 vs 大疆
  fig_seam_hotspots_v4.jpg  缝合线热点放大：v3 / v4 / 大疆（三栏 x 4 处）
  fig_seam_zoom_v4.jpg      单点极限放大
支持直接读管线 npy（省去导出 GeoTIFF）。
用法: python make_figs_v4.py
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
V4 = os.environ.get("FIG_V4", os.path.join(OUT, "ortho_rgb3_cut.npy"))
V3 = os.environ.get("FIG_V3", os.path.join(OUT, "ortho_rgb3.npy"))
V2P = os.environ.get("FIG_V2P", os.path.join(OUT, "ortho_rgb.npy"))
OM = json.load(open(os.path.join(OUT, "ortho_meta.json"), encoding="utf-8"))

HOT = [(562181.6, 3324720.8), (562180.5, 3324560.2),
       (562299.5, 3324715.9), (561955.7, 3324570.1)]


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


def read_at(path, E0, N0, E1, N1, w, h):
    """按地理窗口取图并缩放到 (w,h)。.npy 走直接切片，.tif 走 rasterio。"""
    if path.lower().endswith(".npy"):
        a = np.load(path, mmap_mode="r")
        g0, x0, y0 = OM["gsd"], OM["x0"], OM["y0"]
        c0 = int(round((E0 - x0) / g0)); r0 = int(round((y0 - N1) / g0))
        wpx = max(1, int(round((E1 - E0) / g0))); hpx = max(1, int(round((N1 - N0) / g0)))
        sub = np.ascontiguousarray(a[r0:r0 + hpx, c0:c0 + wpx])
        it = cv2.INTER_AREA if wpx > w else cv2.INTER_LINEAR
        sub = cv2.resize(sub, (w, h), interpolation=it)
        return np.stack([sub[:, :, 2], sub[:, :, 1], sub[:, :, 0]], 2)
    with rasterio.open(path) as d:
        win = from_bounds(E0, N0, E1, N1, d.transform)
        nb = min(3, d.count)
        a = d.read(indexes=tuple(range(1, nb + 1)), window=win,
                   out_shape=(nb, h, w), resampling=Resampling.bilinear)
        av = None
        if d.count >= 4:
            av = d.read(indexes=4, window=win, out_shape=(h, w),
                        resampling=Resampling.nearest) > 128
    bgr = np.stack([a[i] for i in (2, 1, 0)], 2)
    if av is not None:
        bgr = bgr.copy(); bgr[~av] = 255
    return bgr


def main():
    with rasterio.open(DJI) as d:
        db = d.bounds
    mw = OM["w"] * OM["gsd"]; mh = OM["h"] * OM["gsd"]
    E0, E1 = max(OM["x0"], db.left), min(OM["x0"] + mw, db.right)
    N1, N0 = min(OM["y0"], db.top), max(OM["y0"] - mh, db.bottom)
    print(f"重叠区 E {E0:.0f}~{E1:.0f}  N {N0:.0f}~{N1:.0f}")

    # ---- 1) 全图 ----
    w = 1300
    h = int(round(w * (N1 - N0) / (E1 - E0)))
    v3 = read_at(V3, E0, N0, E1, N1, w, h)
    v4 = read_at(V4, E0, N0, E1, N1, w, h)
    dj = read_at(DJI, E0, N0, E1, N1, w, h)
    fig = row_of([lab(v3, "v3  single-frame + LF tone consensus"),
                  lab(v4, "v4  + seam-line optimisation (this round)"),
                  lab(dj, "DJI Terra (reference)")], gap=8)
    cv2.imencode(".jpg", fig, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(
        os.path.join(OUT, "fig_final_v4_full.jpg"))
    print("fig_final_v4_full.jpg", fig.shape)

    # ---- 2) 热点放大 ----
    N = 560
    half = 30.0
    rows = []
    for k, (E, Nc) in enumerate(HOT):
        p3 = read_at(V3, E - half, Nc - half, E + half, Nc + half, N, N)
        p4 = read_at(V4, E - half, Nc - half, E + half, Nc + half, N, N)
        pd = read_at(DJI, E - half, Nc - half, E + half, Nc + half, N, N)
        rows.append(row_of([lab(p3, f"#{k+1} v3 BEFORE  E{E:.0f} N{Nc:.0f}"),
                            lab(p4, f"#{k+1} v4 AFTER  seam re-routed"),
                            lab(pd, f"#{k+1} DJI Terra")], gap=5))
    gap = np.full((8, rows[0].shape[1], 3), 245, np.uint8)
    st = rows[0]
    for r in rows[1:]:
        st = np.concatenate([st, gap, r], 0)
    cv2.imencode(".jpg", st, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(
        os.path.join(OUT, "fig_seam_hotspots_v4.jpg"))
    print("fig_seam_hotspots_v4.jpg", st.shape)

    # ---- 3) 单点极限放大（热点 #1，缝合线曾横切房屋） ----
    E, Nc = HOT[0]
    half = 13.0
    n2 = 700
    p3 = read_at(V3, E - half, Nc - half, E + half, Nc + half, n2, n2)
    p4 = read_at(V4, E - half, Nc - half, E + half, Nc + half, n2, n2)
    pd = read_at(DJI, E - half, Nc - half, E + half, Nc + half, n2, n2)
    z = row_of([lab(p3, f"BEFORE  v3  x{n2/(2*half/OM['gsd']):.1f}"),
                lab(p4, "AFTER   v4 (seam moved off the roof)"),
                lab(pd, "DJI")], gap=5)
    cv2.imencode(".jpg", z, [int(cv2.IMWRITE_JPEG_QUALITY), 93])[1].tofile(
        os.path.join(OUT, "fig_seam_zoom_v4.jpg"))
    print("fig_seam_zoom_v4.jpg", z.shape)


if __name__ == "__main__":
    main()
