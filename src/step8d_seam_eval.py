# -*- coding: utf-8 -*-
"""Step8d: 接缝可见度的**公平**全图评估。

要点：不能用同一套缝线位置评测两个版本 ——
  · v3 在 v4 的新缝线位置上本来就没有缝线，测出来的是"自然地物梯度"；
  · 反之亦然。
所以每个版本都用**自己的 vidx** 定义自己的缝合带，并各自减去
**大疆在同一位置的跨线落差**（这才是真正由"我的接缝"引入的量）。

指标：
  cross_step  缝合线两侧各 1.1m 的灰度落差
  超出中位/p90 = 我方落差 − 大疆在同位置落差
  >8 灰阶占比  肉眼可辨的接缝像素比例
产出 out/metrics_seam.json + 控制台表
"""
import os
import sys
import json
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT

ROOT = r"D:/lr/project/无人机航拍拼接"
DJI = os.path.join(ROOT, "xiangjia1", "map", "result.tif")
DS = 2                      # 下采样倍数
WSPAN = 20                  # 跨线半距（原分辨率像素，=1.1m）


def main():
    om = json.load(open(os.path.join(OUT, "ortho_meta.json"), encoding="utf-8"))
    gsd, x0, y0, W, H = om["gsd"], om["x0"], om["y0"], om["w"], om["h"]
    w = W // DS
    h = H // DS
    print(f"网格 {W}x{H} -> {w}x{h} (DS={DS}); 跨线半距 {WSPAN}px={WSPAN*gsd*100:.1f}cm -> {WSPAN//DS}px(下采样)")

    with rasterio.open(DJI) as d:
        win = from_bounds(x0, y0 - h * DS * gsd, x0 + w * DS * gsd, y0, d.transform)
        a = d.read(indexes=(1, 2, 3), window=win, out_shape=(3, h, w),
                   resampling=Resampling.average)
        av = d.read(4, window=win, out_shape=(h, w), resampling=Resampling.nearest) > 128
    dj = np.stack([a[i] for i in (2, 1, 0)], 2)
    gdji = cv2.cvtColor(dj, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mv_full = np.load(os.path.join(OUT, "ortho_valid.npy"), mmap_mode="r")
    SL = (slice(0, h * DS, DS), slice(0, w * DS, DS))
    mv = np.ascontiguousarray(mv_full[SL])

    items = [("v3 优化前", "ortho_rgb3.npy", "ortho_vidx.npy"),
             ("v4 接缝优化", "ortho_rgb3_cut.npy", "ortho_vidx_cut.npy")]
    wq = max(1, WSPAN // DS)
    res = {}
    print(f"\n{'方案':14s} {'缝线px':>9s} {'落差中位':>9s} {'DJI同位置':>10s} {'超出中位':>9s} "
          f"{'超出p90':>9s} {'>8灰阶占比':>11s} {'缝线Sobel':>10s} {'刀口超出':>9s}")
    for nm, ip, vp in items:
        arr = np.load(os.path.join(OUT, ip), mmap_mode="r")
        sub = np.ascontiguousarray(arr[SL])
        g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float32)
        vi = np.ascontiguousarray(np.load(os.path.join(OUT, vp))[SL])
        bx = np.zeros_like(mv)
        bx[:, 1:] |= (vi[:, 1:] != vi[:, :-1]) & mv[:, 1:] & mv[:, :-1]
        bx[1:, :] |= (vi[1:, :] != vi[:-1, :]) & mv[1:, :] & mv[:-1, :]
        m2 = mv & av
        bx = bx & m2
        ys, xs = np.nonzero(bx)
        sel = (ys >= wq) & (ys < h - wq) & (xs >= wq) & (xs < w - wq)
        ys, xs = ys[sel], xs[sel]
        st = max(1, len(ys) // 500000)
        ys, xs = ys[::st], xs[::st]
        dn = np.maximum(np.abs(g[ys, xs + wq] - g[ys, xs - wq]),
                        np.abs(g[ys + wq, xs] - g[ys - wq, xs]))
        dr = np.maximum(np.abs(gdji[ys, xs + wq] - gdji[ys, xs - wq]),
                        np.abs(gdji[ys + wq, xs] - gdji[ys - wq, xs]))
        ex = dn - dr
        # 缝线处「锐利刀口」超出量：Sobel p95 带内值 − 大疆在同一带的 p95
        mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3), cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
        magd = np.hypot(cv2.Sobel(gdji, cv2.CV_32F, 1, 0, 3), cv2.Sobel(gdji, cv2.CV_32F, 0, 1, 3))
        bd = cv2.dilate(bx.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & m2
        se_m = float(np.percentile(mag[bd], 95)) if bd.sum() > 100 else float("nan")
        se_d = float(np.percentile(magd[bd], 95)) if bd.sum() > 100 else float("nan")
        d = dict(seam_px=int(bx.sum()), n=int(len(ys)),
                 step_median=float(np.median(dn)), step_p90=float(np.percentile(dn, 90)),
                 dji_median=float(np.median(dr)),
                 excess_median=float(np.median(ex)),
                 excess_p90=float(np.percentile(ex, 90)),
                 frac_gt8=float((ex > 8).mean() * 100),
                 sobel_p95=se_m, sobel_excess=se_m - se_d)
        res[nm] = d
        print(f"{nm:14s} {bx.sum():9d} {d['step_median']:9.2f} {d['dji_median']:10.2f} "
              f"{d['excess_median']:+9.2f} {d['excess_p90']:+9.2f} {d['frac_gt8']:10.1f}% "
              f"{se_m:9.1f} {d['sobel_excess']:+9.2f}")

    json.dump(res, open(os.path.join(OUT, "metrics_seam.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n已写 metrics_seam.json")


if __name__ == "__main__":
    main()
