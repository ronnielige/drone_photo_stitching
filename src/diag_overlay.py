# -*- coding: utf-8 -*-
"""红青叠加(anaglyph)配准诊断：我的(红) vs 大疆(青)。
对齐 -> 灰度；错位 -> 红/青彩边。同时用模板匹配给出精确位移。
用法: python diag_overlay.py [E] [N] [half_m] ...
不带参数则检查 step8c 用的 4 个窗口。
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


def grab(path, E0, N1, half, alpha=False):
    E1, N0 = E0 + 2 * half, N1 - 2 * half
    with rasterio.open(path) as d:
        win = from_bounds(E0, N0, E1, N1, d.transform)
        nb = min(3, d.count)
        a = d.read(indexes=tuple(range(1, nb + 1)), window=win)
        av = d.read(indexes=4, window=win) > 128 if (alpha and d.count >= 4) else None
    n = int(round(2 * half / GSD))
    bgr = np.stack([cv2.resize(a[i], (n, n), interpolation=cv2.INTER_LINEAR)
                    for i in (2, 1, 0)], axis=2)
    if av is not None:
        av = cv2.resize(av.astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST) > 0
    return bgr, av


def gray(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)


def best_shift(a, b, m, rad):
    ga, gb = gray(a), gray(b)
    hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 2.0)
    ha, hb = hp(ga), hp(gb)
    H, W = ha.shape
    trim = rad + 4
    tpl = ha[trim:H - trim, trim:W - trim]
    reg = hb
    res = cv2.matchTemplate(reg, tpl, cv2.TM_CCOEFF_NORMED)
    _, mv, _, ml = cv2.minMaxLoc(res)
    dx, dy = ml[0] - trim, ml[1] - trim
    return dx, dy, mv


def lab(img, t, h=30):
    bar = np.full((h, img.shape[1], 3), 25, np.uint8)
    cv2.putText(bar, t, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2,
                cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def anaglyph(a, b, mask=None):
    """a->红, b->青；对齐则近似灰"""
    ga, gb = gray(a), gray(b)
    ga = (ga - ga.mean()) / (ga.std() + 1e-6) * 40 + 128
    gb = (gb - gb.mean()) / (gb.std() + 1e-6) * 40 + 128
    out = np.zeros((ga.shape[0], ga.shape[1], 3), np.uint8)
    out[:, :, 2] = np.clip(ga, 0, 255)          # R
    out[:, :, 1] = np.clip(gb, 0, 255)          # G
    out[:, :, 0] = np.clip(gb, 0, 255)          # B
    if mask is not None:
        out[~mask] = 0
    return out


def one(E, N, half, tag):
    m3, _ = grab(V3, E, N, half)
    m2, _ = grab(V2P, E, N, half)
    v1, _ = grab(V1, E, N, half)
    dj, dja = grab(DJI, E, N, half, alpha=True)
    ok = dja if dja is not None else np.ones(m3.shape[:2], bool)
    rad = int(half * 0.35 / GSD)
    print(f"\n[{tag}] 中心 E={E} N={N} 半宽{half}m  大疆有效 {ok.mean()*100:.0f}%")
    for nm, im in (("v3", m3), ("v2pure", m2), ("v1", v1)):
        dx, dy, mv = best_shift(im, dj, ok, rad)
        print(f"   {nm:6s} vs DJI 最佳位移 dx={dx:+d} dy={dy:+d} px "
              f"= ({dx*GSD*100:+.1f},{dy*GSD*100:+.1f}) cm  响应 {mv:.3f}")
    panels = [lab(m3, tag + " | MINE v3"), lab(m2, "MINE v2 pure"),
              lab(dj, "DJI"), lab(anaglyph(m3, dj, ok), "anaglyph R=mine C=DJI")]
    sep = np.full((panels[0].shape[0], 6, 3), 245, np.uint8)
    row = panels[0]
    for p in panels[1:]:
        row = np.concatenate([row, sep, p], axis=1)
    return row


def main():
    args = sys.argv[1:]
    if len(args) >= 3:
        wins = [(float(args[0]), float(args[1]), float(args[2]), "custom")]
    else:
        wins = [(562336, 3324559, 20, "#1"), (562501, 3324402, 20, "#2"),
                (562446, 3324756, 20, "#3"), (562008, 3324441, 20, "#4"),
                (562230, 3324560, 60, "wide")]
    rows = [one(E, N, half, tag) for E, N, half, tag in wins]
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=245) for r in rows]
    gap = np.full((8, W, 3), 245, np.uint8)
    stack = rows[0]
    for r in rows[1:]:
        stack = np.concatenate([stack, gap, r], axis=0)
    p = os.path.join(OUT, "diag_overlay.jpg")
    cv2.imencode(".jpg", stack, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(p)
    if stack.shape[0] > 2600:
        s = 2600 / stack.shape[0]
        small = cv2.resize(stack, (int(stack.shape[1] * s), 2600), interpolation=cv2.INTER_AREA)
        cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(
            os.path.join(OUT, "diag_overlay_small.jpg"))
        print("小图:", os.path.join(OUT, "diag_overlay_small.jpg"))
    print("已写", p, stack.shape)


if __name__ == "__main__":
    main()
