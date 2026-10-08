# -*- coding: utf-8 -*-
"""接缝优化窗口级评估：以某个 vidx 的缝合带为统一采样位置，
比较 v3（优化前）/ v4（优化后）/ DJI 在同一位置的：
  cross_step  跨接缝两侧各 1.1m 的灰度落差（捕捉"平滑色块"与"细线"）
  seam_p95    原始梯度在缝合带上的 p95 超出量（= 带内 p95 - 远场对照带 p95）
用法: python cut_eval.py E0 N1 size [vidx_file]
"""
import os
import sys
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
import rasterio
from rasterio.windows import from_bounds

ROOT = r"D:/lr/project/无人机航拍拼接"
DJI = os.path.join(ROOT, "xiangjia1", "map", "result.tif")


def gray(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)


def cross_step(g, bx, w=20):
    H, W = g.shape
    ys, xs = np.nonzero(bx)
    sel = (ys >= w) & (ys < H - w) & (xs >= w) & (xs < W - w)
    ys, xs = ys[sel], xs[sel]
    if len(ys) == 0:
        return np.array([0.0])
    st = max(1, len(ys) // 400000)
    ys, xs = ys[::st], xs[::st]
    a = g[ys, np.clip(xs + w, 0, W - 1)] - g[ys, np.clip(xs - w, 0, W - 1)]
    b = g[np.clip(ys + w, 0, H - 1), xs] - g[np.clip(ys - w, 0, H - 1), xs]
    return np.maximum(np.abs(a), np.abs(b))


def band_metrics(g, bx, mv):
    k = np.ones((5, 5), np.uint8)
    bd = cv2.dilate(bx.astype(np.uint8), k).astype(bool) & mv
    dist = cv2.distanceTransform((~bx).astype(np.uint8), cv2.DIST_L2, 5)
    ctrl = (dist > 26) & mv
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3), cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
    if bd.sum() < 100 or ctrl.sum() < 100:
        return float("nan"), float("nan")
    return float(np.percentile(mag[bd], 95) - np.percentile(mag[ctrl], 95)), \
        float(np.percentile(mag[bd], 95))


def main():
    E0, N1, size = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    meta = json.load(open(os.path.join(OUT, "ortho_meta.json"), encoding="utf-8"))
    gsd, gx0, gy0 = meta["gsd"], meta["x0"], meta["y0"]
    c0 = int(round((E0 - gx0) / gsd)); r0 = int(round((gy0 - N1) / gsd))
    n = int(round(size / gsd))
    mv = np.ascontiguousarray(np.load(os.path.join(OUT, "ortho_valid.npy"))[r0:r0 + n, c0:c0 + n])

    with rasterio.open(DJI) as d:
        w = from_bounds(E0, N1 - size, E0 + size, N1, d.transform)
        nb = min(3, d.count)
        a = d.read(indexes=tuple(range(1, nb + 1)), window=w, out_shape=(nb, n, n),
                   resampling=rasterio.enums.Resampling.average)
    dj = np.stack([a[i] for i in (2, 1, 0)], 2)
    gdji = gray(dj)

    # 每个方案用「自己的 vidx」定义自己的缝合带 —— 这才公平
    items = [("v3 优化前", "ortho_rgb3.npy", "ortho_vidx.npy"),
             ("v4 接缝优化", "ortho_rgb3_cut.npy", "ortho_vidx_cut.npy")]
    print(f"窗口 {E0:.0f},{N1:.0f} {size:.0f}m @{gsd*100:.2f}cm")
    print(f"{'方案':14s} {'带px':>8s} {'落差中位':>9s} {'DJI同位置':>10s} {'超出中位':>9s} "
          f"{'超出p90':>9s} {'seam超出':>9s} {'能量密度':>9s} {'>10占比':>8s}")
    rows = {}
    for nm, ip, vp in items:
        if not os.path.exists(os.path.join(OUT, ip)):
            continue
        a2 = np.load(os.path.join(OUT, ip), mmap_mode="r")
        sub = np.ascontiguousarray(a2[r0:r0 + n, c0:c0 + n])
        im = np.stack([sub[:, :, 2], sub[:, :, 1], sub[:, :, 0]], 2)
        vi = np.ascontiguousarray(np.load(os.path.join(OUT, vp))[r0:r0 + n, c0:c0 + n])
        bx = np.zeros_like(mv)
        bx[:, 1:] |= (vi[:, 1:] != vi[:, :-1]) & mv[:, 1:] & mv[:, :-1]
        bx[1:, :] |= (vi[1:, :] != vi[:-1, :]) & mv[1:, :] & mv[:-1, :]
        g = gray(im)
        v = cross_step(g, bx)
        r = cross_step(gdji, bx)
        ex = v - r
        e2, p2 = band_metrics(g, bx, mv)
        e2d, _ = band_metrics(gdji, bx, mv)
        dens = float(np.sum(np.maximum(ex, 0))) / mv.size * 100
        frac = float((ex > 10).mean() * 100)
        rows[nm] = (bx.sum(), v, ex, e2 - e2d)
        print(f"{nm:14s} {bx.sum():8d} {np.median(v):9.2f} {np.median(r):10.2f} "
              f"{np.median(ex):+9.2f} {np.percentile(ex,90):+9.2f} {e2-e2d:+9.2f} "
              f"{dens:9.2f} {frac:7.1f}%")

    # DJI 自身
    print(f"{'DJI 基准':14s} {'-':>8s} {'-':>9s} {'-':>10s} {'0':>9s} {'0':>9s} {'0':>9s} "
          f"{'0':>9s} {'0':>8s}")
    # 对照图：原图 + 差异放大
    panes = []
    for nm, ip, vp in items:
        p = os.path.join(OUT, ip)
        if not os.path.exists(p):
            continue
        a2 = np.load(p, mmap_mode="r")
        sub = np.ascontiguousarray(a2[r0:r0 + n, c0:c0 + n])
        panes.append((nm, np.stack([sub[:, :, 2], sub[:, :, 1], sub[:, :, 0]], 2)))
    panes.append(("DJI", dj))
    if len(panes) >= 2:
        g3 = gray(panes[0][1]); g4 = gray(panes[1][1])
        dmap = np.clip(np.abs(g4 - g3) * 6, 0, 255).astype(np.uint8)
        dmap = cv2.applyColorMap(dmap, cv2.COLORMAP_TURBO)
        Wp = 620
        ps = []
        for nm, im in panes:
            ps.append(label(cv2.resize(im, (Wp, Wp), interpolation=cv2.INTER_AREA), nm))
        ps.append(label(cv2.resize(dmap, (Wp, Wp), interpolation=cv2.INTER_NEAREST), "|v4-v3| x6"))
        sep = np.full((ps[0].shape[0], 5, 3), 240, np.uint8)
        row = ps[0]
        for p in ps[1:]:
            row = np.concatenate([row, sep, p], 1)
        from common import imwrite_any
        imwrite_any(os.path.join(OUT, "fig_cut_eval.jpg"), row, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print("已写 fig_cut_eval.jpg")


def label(img, txt, h=28, sc=0.62):
    bar = np.full((h, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, txt, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, sc,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


if __name__ == "__main__":
    main()
