# -*- coding: utf-8 -*-
"""Step9a: 全图扫描「缝合线可见度」，定位肉眼最可能看到的接缝热点。

可见度 = 跨线落差 × 该处纹理强度
  - 跨线落差：在视图切换线上，沿法向跨 w 像素取两点灰度差 |I(+w)-I(-w)|
    （梯度类指标抓不到"平滑大色块"，这种跨距差能抓到）
  - 纹理强度：缝合线处的局部梯度幅值。平坦农田上的落差不易被注意，
    但若缝隙切过屋檐/田埂，1px 错位立刻变成可见"断口"。
产出 out/seam_scan.npz（4x 下采样热度图）+ out/fig_seam_heat.jpg + 热点坐标表。
"""
import os
import sys
import json
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, imwrite_any

DS = 4
WSPAN = 20              # 跨线取样半距（像素，1.1m）
BLK = 2048
HALO = 32


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "ortho_rgb3.npy")
    meta = json.load(open(os.path.join(OUT, "ortho_meta.json"), encoding="utf-8"))
    gsd, gx0, gy0, W, H = meta["gsd"], meta["x0"], meta["y0"], meta["w"], meta["h"]
    rgb = np.load(src, mmap_mode="r")
    vidx = np.load(os.path.join(OUT, "ortho_vidx.npy"), mmap_mode="r")
    valid = np.load(os.path.join(OUT, "ortho_valid.npy"), mmap_mode="r")
    print(f"扫描 {src}  {W}x{H} @{gsd*100:.2f}cm")

    hw, hh = (W + DS - 1) // DS, (H + DS - 1) // DS
    heat = np.zeros((hh, hw), np.float32)      # 落差×纹理 的热度
    stepm = np.zeros((hh, hw), np.float32)     # 纯落差
    seamd = np.zeros((hh, hw), np.uint8)       # 缝合线密度
    t0 = time.time()
    ny = (H + BLK - 1) // BLK
    for iy, y0 in enumerate(range(0, H, BLK)):
        for x0 in range(0, W, BLK):
            y1, x1 = min(y0 + BLK, H), min(x0 + BLK, W)
            ry0, ry1 = max(0, y0 - HALO), min(H, y1 + HALO)
            rx0, rx1 = max(0, x0 - HALO), min(W, x1 + HALO)
            g = cv2.cvtColor(np.ascontiguousarray(rgb[ry0:ry1, rx0:rx1]),
                             cv2.COLOR_BGR2GRAY).astype(np.float32)
            v = np.ascontiguousarray(vidx[ry0:ry1, rx0:rx1])
            m = np.ascontiguousarray(valid[ry0:ry1, rx0:rx1])
            bx = np.zeros_like(m)
            bx[:, 1:] |= (v[:, 1:] != v[:, :-1]) & m[:, 1:] & m[:, :-1]
            bx[1:, :] |= (v[1:, :] != v[:-1, :]) & m[1:, :] & m[:-1, :]
            # 落差（水平/垂直各取，注意 halo 内计算避免边界效应）
            hi, wi = g.shape
            dh = np.zeros_like(g)
            dh[:, WSPAN:wi - WSPAN] = np.abs(g[:, 2 * WSPAN:] - g[:, :wi - 2 * WSPAN])
            dv = np.zeros_like(g)
            dv[WSPAN:hi - WSPAN, :] = np.abs(g[2 * WSPAN:, :] - g[:hi - 2 * WSPAN, :])
            step = np.maximum(dh, dv)
            # 纹理强度（3x3 梯度幅值）
            tex = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3),
                           cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
            vis = step * np.clip(tex / 40.0, 0, 2.0)
            bxf = bx.astype(np.float32)
            # 块内 4x4 池化：求和
            def pool(a):
                a = a[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0]
                h2, w2 = ((a.shape[0] + DS - 1) // DS) * DS, ((a.shape[1] + DS - 1) // DS) * DS
                p = np.zeros((h2, w2), np.float32)
                p[:a.shape[0], :a.shape[1]] = a
                return p.reshape(h2 // DS, DS, w2 // DS, DS).mean((1, 3))
            pl = pool(vis * bxf)
            ps = pool(step * bxf)
            pc = pool(bxf)
            yy, xx = y0 // DS, x0 // DS
            heat[yy:yy + pl.shape[0], xx:xx + pl.shape[1]] = pl
            stepm[yy:yy + pl.shape[0], xx:xx + pl.shape[1]] = ps
            seamd[yy:yy + pl.shape[0], xx:xx + pl.shape[1]] = np.clip(pc * 255, 0, 255).astype(np.uint8)
        print(f"  行 {iy+1}/{ny} ({time.time()-t0:.0f}s)")
    np.savez_compressed(os.path.join(OUT, "seam_scan.npz"),
                        heat=heat, step=stepm, seam=seamd, ds=DS,
                        gsd=gsd, gx0=gx0, gy0=gy0)

    # 排除「有效区边界」附近：那里的标签变化只是覆盖边缘，不是真接缝
    vms = np.ascontiguousarray(valid[::DS, ::DS])
    inside = cv2.distanceTransform(vms.astype(np.uint8), cv2.DIST_L2, 5) > (120.0 / DS)
    heat = heat * inside
    stepm = stepm * inside

    # 热点：8 单元滑动平均取局部极大
    k = 8
    sm = cv2.blur(heat, (k, k))
    hits = []
    tmp = sm.copy()
    for _ in range(30):
        _, mx, _, ml = cv2.minMaxLoc(tmp)
        if mx <= 0:
            break
        x, y = ml
        E = gx0 + x * DS * gsd
        N = gy0 - y * DS * gsd
        hits.append((float(mx), float(E), float(N), x, y))
        cv2.circle(tmp, (x, y), 60, 0, -1)
    hits.sort(reverse=True)
    print("\n缝合线可见度热点（热度 / 东坐标 / 北坐标 / 与最高点距离）:")
    for i, (h, E, N, x, y) in enumerate(hits[:16]):
        print(f"  #{i+1:<2d} {h:8.2f}   E {E:9.1f}  N {N:9.1f}")

    # 在原图上标热点 + vidx 伪彩
    small = cv2.resize(np.ascontiguousarray(rgb[::DS, ::DS]),
                       (hw, hh), interpolation=cv2.INTER_AREA)
    marks = small.copy()
    for i, (h, E, N, x, y) in enumerate(hits[:10]):
        cv2.circle(marks, (x, y), 46, (0, 0, 255), 3)
        cv2.putText(marks, f"#{i+1}", (x - 22, y - 52), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 3, cv2.LINE_AA)
    vv = np.ascontiguousarray(vidx[::DS, ::DS])
    rng = np.random.RandomState(7)
    lut = rng.randint(40, 255, (int(vv.max()) + 2, 3)).astype(np.uint8)
    lut[0] = 0
    lab = lut[np.clip(vv, 0, int(vv.max()) + 1)]
    lab[~vms] = 255
    cv2.putText(marks, "seam hotspots (boundary excluded)", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(lab, "view label map (vidx)", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    imwrite_any(os.path.join(OUT, "fig_seam_heat.jpg"), marks, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    imwrite_any(os.path.join(OUT, "fig_seam_vidx.jpg"), lab, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"已写 fig_seam_heat.jpg / fig_seam_vidx.jpg {marks.shape[1]}x{marks.shape[0]}")


if __name__ == "__main__":
    main()
