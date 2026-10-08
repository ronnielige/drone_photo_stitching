# -*- coding: utf-8 -*-
"""热点巡查：把 seam_scan 定位出的缝合线热点逐个放大，三栏对照
[MINE v3 | 视图标签 vidx | DJI Terra]，用于肉眼判断接缝是否真的可见。
用法: python hotspot_view.py [half_m]
产出 out/fig_hotspots.jpg
"""
import os
import sys
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, imwrite_any

ROOT = r"D:/lr/project/无人机航拍拼接"
DJI = os.path.join(ROOT, "xiangjia1", "map", "result.tif")

HOT = [(562181.6, 3324720.8), (562180.5, 3324560.2), (562299.5, 3324715.9),
       (561955.7, 3324570.1), (562439.7, 3324233.7), (562467.8, 3324836.3)]


def label(img, txt, h=28, sc=0.62):
    bar = np.full((h, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, txt, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, sc,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def main():
    half = float(sys.argv[1]) if len(sys.argv) > 1 else 35.0
    meta = json.load(open(os.path.join(OUT, "ortho_meta.json"), encoding="utf-8"))
    gsd, gx0, gy0 = meta["gsd"], meta["x0"], meta["y0"]
    rgb = np.load(os.path.join(OUT, "ortho_rgb3.npy"), mmap_mode="r")
    cutp = os.path.join(OUT, "ortho_rgb3_cut.npy")
    cut = np.load(cutp, mmap_mode="r") if os.path.exists(cutp) else None
    vidx = np.load(os.path.join(OUT, "ortho_vidx.npy"), mmap_mode="r")
    vcutp = os.path.join(OUT, "ortho_vidx_cut.npy")
    vcut = np.load(vcutp, mmap_mode="r") if os.path.exists(vcutp) else None
    valid = np.load(os.path.join(OUT, "ortho_valid.npy"), mmap_mode="r")

    import rasterio
    from rasterio.windows import from_bounds
    dp = rasterio.open(DJI)
    print(f"DJI {dp.width}x{dp.height} @{dp.transform.a*100:.2f}cm bounds {[round(v) for v in dp.bounds]}")

    N = 620
    rows = []
    for k, (E, Nc) in enumerate(HOT):
        c = int((E - gx0) / gsd); r = int((gy0 - Nc) / gsd)
        h = int(half / gsd)
        mine = np.ascontiguousarray(rgb[r - h:r + h, c - h:c + h])
        vv = np.ascontiguousarray(vidx[r - h:r + h, c - h:c + h])
        mv = np.ascontiguousarray(valid[r - h:r + h, c - h:c + h])
        rng = np.random.RandomState(7)
        lut = rng.randint(40, 255, (int(vv.max()) + 2, 3)).astype(np.uint8)
        lut[0] = 0
        lab = lut[np.clip(vv, 0, int(vv.max()) + 1)]
        lab[~mv] = 255
        with rasterio.open(DJI) as d:
            w = from_bounds(E - half, Nc - half, E + half, Nc + half, d.transform)
            nb = min(3, d.count)
            on = max(8, min(N, int(round(2 * half / d.transform.a))))
            try:
                a = d.read(indexes=tuple(range(1, nb + 1)), window=w, out_shape=(nb, on, on),
                           resampling=rasterio.enums.Resampling.average)
            except Exception:
                a = np.zeros((nb, on, on), np.uint8)
        dj = np.stack([a[i] for i in (2, 1, 0)], 2)
        if on != N:
            dj = cv2.resize(dj, (N, N), interpolation=cv2.INTER_CUBIC)
        pn = [label(cv2.resize(mine, (N, N), interpolation=cv2.INTER_AREA), f"#{k+1} v3  E{E:.0f} N{Nc:.0f}")]
        if cut is not None:
            cu = np.ascontiguousarray(cut[r - h:r + h, c - h:c + h])
            pn.append(label(cv2.resize(cu, (N, N), interpolation=cv2.INTER_AREA), f"#{k+1} v4 seam-cut"))
        if vcut is not None:
            vc = np.ascontiguousarray(vcut[r - h:r + h, c - h:c + h])
            rng2 = np.random.RandomState(7)
            lut2 = rng2.randint(40, 255, (int(vc.max()) + 2, 3)).astype(np.uint8)
            lut2[0] = 0
            lb = lut2[np.clip(vc, 0, int(vc.max()) + 1)]
            lb[~mv] = 255
            pn.append(label(cv2.resize(lb, (N, N), interpolation=cv2.INTER_NEAREST), f"#{k+1} cut labels"))
        else:
            pn.append(label(cv2.resize(lab, (N, N), interpolation=cv2.INTER_NEAREST), f"#{k+1} view labels"))
        pn.append(label(dj, f"#{k+1} DJI Terra"))
        sep = np.full((N + 28, 5, 3), 240, np.uint8)
        row = pn[0]
        for p in pn[1:]:
            row = np.concatenate([row, sep, p], 1)
        rows.append(row)
    gap = np.full((8, rows[0].shape[1], 3), 240, np.uint8)
    stack = rows[0]
    for rr in rows[1:]:
        stack = np.concatenate([stack, gap, rr], 0)
    imwrite_any(os.path.join(OUT, "fig_hotspots_cut.jpg"), stack, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"已写 fig_hotspots_cut.jpg {stack.shape[1]}x{stack.shape[0]}")


if __name__ == "__main__":
    main()
