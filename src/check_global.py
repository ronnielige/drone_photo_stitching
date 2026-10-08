# -*- coding: utf-8 -*-
"""全局配准检验：在共同地理范围内降采样，用模板匹配求我的正射相对大疆的整体平移。
输出绝对偏移 (m) 与是否只是"网格原点不同"能解释的量级。
用法: python check_global.py
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
MINE = os.path.join(OUT, "xiangjia_ortho_mine.tif")

DS = 8          # 降采样倍数 -> 0.44 m/px
os.makedirs(os.path.join(OUT, "align"), exist_ok=True)


def read_common(path, E0, N1, E1, N0, ds):
    with rasterio.open(path) as d:
        win = from_bounds(E0, N0, E1, N1, d.transform)
        h = max(1, int(round((N1 - N0) / (d.transform.a * ds))))
        w = max(1, int(round((E1 - E0) / (d.transform.a * ds))))
        nb = min(3, d.count)
        a = d.read(indexes=tuple(range(1, nb + 1)), window=win,
                   out_shape=(nb, h, w), resampling=Resampling.average)
        av = None
        if d.count >= 4:
            av = d.read(indexes=4, window=win, out_shape=(1, h, w),
                        resampling=Resampling.nearest)[0] > 128
    bgr = np.stack([a[i] for i in (2, 1, 0)], axis=2)
    return bgr, av, (h, w)


def main():
    with rasterio.open(MINE) as m:
        mb = m.bounds
        mgsd = m.transform.a
        mp = json.load(open(os.path.join(OUT, "ortho_meta.json"), encoding="utf-8"))
    with rasterio.open(DJI) as d:
        db = d.bounds
        dgsd = d.transform.a
    print(f"我的: E[{mb.left:.1f},{mb.right:.1f}] N[{mb.bottom:.1f},{mb.top:.1f}] "
          f"{(mb.right-mb.left):.1f}x{(mb.top-mb.bottom):.1f}m @{mgsd*100:.2f}cm")
    print(f"大疆: E[{db.left:.1f},{db.right:.1f}] N[{db.bottom:.1f},{db.top:.1f}] "
          f"{(db.right-db.left):.1f}x{(db.top-db.bottom):.1f}m @{dgsd*100:.2f}cm")

    E0 = max(mb.left, db.left); E1 = min(mb.right, db.right)
    N0 = max(mb.bottom, db.bottom); N1 = min(mb.top, db.top)
    print(f"共同范围: {(E1-E0):.1f} x {(N1-N0):.1f} m")

    mine, mv, shp = read_common(MINE, E0, N1, E1, N0, DS)
    dji, dv, _ = read_common(DJI, E0, N1, E1, N0, DS)
    print(f"读取形状 mine {mine.shape} dji {dji.shape} 缩放 {DS}x -> {mgsd*DS:.3f} m/px")

    gm = cv2.cvtColor(mine, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gd = cv2.cvtColor(dji, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for nm, im in (("mine_small", mine), ("dji_small", dji)):
        s = cv2.resize(im, (1200, int(1200 * im.shape[0] / im.shape[1])),
                       interpolation=cv2.INTER_AREA)
        cv2.imencode(".jpg", s, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(
            os.path.join(OUT, "align", nm + ".jpg"))
    hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 2.0)
    hm, hd = hp(gm), hp(gd)
    # 以我的图像中心大块为模板，在大疆全图内搜索
    H, W = hm.shape
    th, tw = int(H * 0.5), int(W * 0.5)
    y0, x0 = (H - th) // 2, (W - tw) // 2
    tpl = hm[y0:y0 + th, x0:x0 + tw]
    res = cv2.matchTemplate(hd, tpl, cv2.TM_CCOEFF_NORMED)
    _, mv_, _, ml = cv2.minMaxLoc(res)
    dx = ml[0] - x0
    dy = ml[1] - y0
    ppm = mgsd * DS
    print(f"\n[全局] 模板匹配峰值 {mv_:.3f}")
    print(f"  我的内容相对大疆在地理网格上偏移 dx={dx}px dy={dy}px "
          f"= ({dx*ppm:+.2f} m, {dy*ppm:+.2f} m)")
    print("  (dx>0 表示我的内容出现在更东侧的地理位置，即我的地理参考偏西)")

    # 目视：把 mine 平移 -d 后与 dji 叠半边对比
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    sh = cv2.warpAffine(mine, M, (W, H), borderValue=(0, 0, 0))
    split = np.concatenate([dji[:, :W // 2], sh[:, W // 2:]], axis=1)
    small = cv2.resize(split, (1400, int(1400 * H / W)), interpolation=cv2.INTER_AREA)
    cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(
        os.path.join(OUT, "align", "split_after_shift.jpg"))
    # 未平移的对照
    split0 = np.concatenate([dji[:, :W // 2], mine[:, W // 2:]], axis=1)
    small0 = cv2.resize(split0, (1400, int(1400 * H / W)), interpolation=cv2.INTER_AREA)
    cv2.imencode(".jpg", small0, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(
        os.path.join(OUT, "align", "split_raw.jpg"))
    print("已写 align/split_raw.jpg 与 align/split_after_shift.jpg (左=大疆, 右=我的)")


if __name__ == "__main__":
    main()
