# -*- coding: utf-8 -*-
"""直接检验：我的正射与大疆正射在同一地理窗口下是否对齐。
用相位相关(phase correlation)求全局平移，避免肉眼误判。
用法: python check_align.py [E中心] [N中心] [半宽m]
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
MINE = os.path.join(OUT, "xiangjia_ortho_mine.tif")
V1 = os.path.join(OUT, "xiangjia_ortho_v1.tif")


def grab(path, E0, N1, half, gsd, use_alpha=False):
    E1, N0 = E0 + 2 * half, N1 - 2 * half
    with rasterio.open(path) as ds:
        win = from_bounds(E0, N0, E1, N1, ds.transform)
        nb = min(3, ds.count)
        a = ds.read(indexes=tuple(range(1, nb + 1)), window=win)
        av = None
        if use_alpha and ds.count >= 4:
            av = ds.read(indexes=4, window=win) > 128
    n = int(round(2 * half / gsd))
    bgr = np.stack([cv2.resize(a[i], (n, n), interpolation=cv2.INTER_AREA)
                    for i in (2, 1, 0)], axis=2)
    if av is not None:
        av = cv2.resize(av.astype(np.uint8), (n, n),
                        interpolation=cv2.INTER_NEAREST) > 0
    return bgr, av


def phase_shift(a, b, m=None):
    """返回 (dx, dy)，单位 px，含义：b 相对 a 的平移（b 右移正 dx）"""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if m is not None:
        ga = ga * m
        gb = gb * m
    hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 3.0)
    ha, hb = hp(ga), hp(gb)
    win = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ha, hb, win)
    return dx, dy, resp


def main():
    E0 = float(sys.argv[1]) if len(sys.argv) > 1 else 562230.0
    N1 = float(sys.argv[2]) if len(sys.argv) > 2 else 3324560.0
    half = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    gsd = 0.055
    om = json.load(open(os.path.join(OUT, "ortho_meta.json"), encoding="utf-8"))

    mine, mv = grab(MINE, E0, N1, half, gsd)
    dji, dv = grab(DJI, E0, N1, half, gsd, use_alpha=True)
    v1, _ = grab(V1, E0, N1, half, gsd)
    print(f"窗口 中心E={E0} N={N1} 半宽{half}m  -> {mine.shape[1]}x{mine.shape[0]} @5.5cm")
    mvstat = "n/a" if mv is None else f"{mv.mean()*100:.1f}%"
    dvstat = "n/a" if dv is None else f"{dv.mean()*100:.1f}%"
    print(f"大疆有效 {dvstat}  我的有效 {mvstat}")

    for nm, im in (("v2 single", mine), ("v1 average", v1)):
        dx, dy, r = phase_shift(im, dji, dv)
        print(f"  {nm:11s} vs DJI: 平移 dx={dx:+.2f}px dy={dy:+.2f}px "
              f"= ({dx*gsd*100:+.1f}cm, {dy*gsd*100:+.1f}cm) 响应 {r:.3f}")
    dx, dy, r = phase_shift(v1, mine, mv)
    print(f"  v1 vs v2  : 平移 dx={dx:+.2f}px dy={dy:+.2f}px 响应 {r:.3f}")

    # 目视三联图
    def lab(img, t, h=30):
        bar = np.full((h, img.shape[1], 3), 25, np.uint8)
        cv2.putText(bar, t, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return np.concatenate([bar, img], axis=0)
    row = np.concatenate([lab(mine, "MINE v2"), lab(v1, "MINE v1"), lab(dji, "DJI")], axis=1)
    p = os.path.join(OUT, "check_align.jpg")
    cv2.imencode(".jpg", row, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(p)
    print("已写", p)


if __name__ == "__main__":
    main()
