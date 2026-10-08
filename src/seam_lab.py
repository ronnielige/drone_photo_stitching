# -*- coding: utf-8 -*-
"""接缝过渡实验台（seam lab）。

目的：在不重跑全图的前提下，于一个窗口内比较多种「消除缝合线可见性」的策略。

背景：v3 的缝合线是几何评分 s_a=s_b 的等值线，两侧次优得分 s2 与 s1 本就接近。
渲染器里的 TRACK2 机制会记录 col2/s2，于是可以离线试各种混合：
  A) 固定 SEAM_EPS：过渡带 = {s1-s2 < eps}，a = clip(0.5+(s1-s2)/(2eps),0,1)
  B) 自适应 eps：按两视图颜色差放大带宽（颜色差大处才需要宽过渡）
  C) 带内改用「多视共识」avg 而非次优视图
所有结果再统一叠一层低频色调共识（LF_M 米）。

用法: python seam_lab.py E0 N1 size_m [tag]
产出 out/seam_lab_<tag>.jpg + 控制台指标
"""
import os
import sys
import math
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step6b_ortho as S
from common import WORK, OUT, load_json, imwrite_any

ROOT = r"D:/lr/project/无人机航拍拼接"
DJI = os.path.join(ROOT, "xiangjia1", "map", "result.tif")

GSD = float(os.environ.get("ORTHO_GSD", "0.055"))
LF_M = float(os.environ.get("LF_M", "2.5"))
TILE = 1024
MAX_VIEWS = S.MAX_VIEWS


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def blocks_render(gx0, gy0, W, H, views, cam_p, Rs, Cs, paths, gains,
                  dsm_v2, dm2, z_fb, radio):
    """窗口内分块渲染，返回 col1/col2/s1/s2/vidx/valid/avg 的拼接结果。"""
    c1 = np.zeros((H, W, 3), np.uint8)
    c2 = np.zeros((H, W, 3), np.uint8)
    s1 = np.full((H, W), -1e9, np.float32)
    s2 = np.full((H, W), -1e9, np.float32)
    vi = np.full((H, W), -1, np.int16)
    vm = np.zeros((H, W), bool)
    av = np.zeros((H, W, 3), np.uint8)
    t0 = time.time()
    nblk = 0
    for ty in range(0, H, TILE):
        for tx in range(0, W, TILE):
            tw, th = min(TILE, W - tx), min(TILE, H - ty)
            d = S.render_rect(gx0 + tx * GSD, gy0 - ty * GSD, GSD, tw, th, views,
                              cam_p, Rs, Cs, paths, gains, dsm_v2, dm2, z_fb,
                              also_avg=True, radio=radio, debug=True)
            c1[ty:ty + th, tx:tx + tw] = d["col1"]
            c2[ty:ty + th, tx:tx + tw] = d["col2"]
            s1[ty:ty + th, tx:tx + tw] = d["s1"]
            s2[ty:ty + th, tx:tx + tw] = d["s2"]
            vi[ty:ty + th, tx:tx + tw] = d["vidx"]
            vm[ty:ty + th, tx:tx + tw] = d["valid"]
            if d["avg"] is not None:
                av[ty:ty + th, tx:tx + tw] = d["avg"]
            nblk += 1
    print(f"  渲染 {nblk} 块 ({time.time()-t0:.0f}s)  有效 {vm.mean()*100:.1f}%")
    return c1, c2, s1, s2, vi, vm, av


def lf_consensus(sf, av, vm, sigma):
    """低频色调共识：out = sf + (LP(av) - LP(sf))，权重只在有效区做归一化。"""
    k = int(2 * round(3 * sigma) + 1)
    s = sf.astype(np.float32)
    a = av.astype(np.float32)
    w = vm.astype(np.float32)[..., None]
    ws = np.maximum(cv2.GaussianBlur(w, (k, k), sigma, borderType=cv2.BORDER_REPLICATE), 1e-6)
    if ws.ndim == 2:
        ws = ws[..., None]
    ls = cv2.GaussianBlur(s * w, (k, k), sigma, borderType=cv2.BORDER_REPLICATE) / ws
    la = cv2.GaussianBlur(a * w, (k, k), sigma, borderType=cv2.BORDER_REPLICATE) / ws
    return np.clip(s + (la - ls), 0, 255).astype(np.uint8)


def mix_eps(c1, c2, s1, s2, eps, adapt=0.0, use_avg=None, vm=None):
    """按 SEAM_EPS 混合；adapt>0 时按颜色差放大带宽；use_avg 非空则带内用多视共识。"""
    if eps <= 0:
        return c1.copy()
    e = np.float32(eps)
    if adapt > 0:
        dif = np.abs(c1.astype(np.float32) - c2.astype(np.float32)).mean(2)
        e = e * (1.0 + adapt * dif / 40.0)
    a = np.clip(0.5 + (s1 - s2) / (2.0 * np.maximum(e, 1e-9)), 0.0, 1.0)
    a[s2 < -1e8] = 1.0
    a[~vm] = 1.0
    a = a[..., None]
    if use_avg is not None:
        base = use_avg.astype(np.float32)
    else:
        base = c2.astype(np.float32)
    return np.clip(a * c1.astype(np.float32) + (1 - a) * base, 0, 255).astype(np.uint8)


def cross_step_metric(g, vidx, mv, w=20):
    """跨切换线落差（两侧各 w 像素的灰度差），无 DJI 基准，直接看绝对值分布。"""
    H, W = g.shape
    bx = np.zeros((H, W), bool)
    bx[:, 1:] |= (vidx[:, 1:] != vidx[:, :-1]) & mv[:, 1:] & mv[:, :-1]
    bx[1:, :] |= (vidx[1:, :] != vidx[:-1, :]) & mv[1:, :] & mv[:-1, :]
    ys, xs = np.nonzero(bx)
    sel = (ys >= w) & (ys < H - w) & (xs >= w) & (xs < W - w)
    ys, xs = ys[sel], xs[sel]
    if len(ys) == 0:
        return np.array([0.0]), 0
    st = max(1, len(ys) // 400000)
    ys, xs = ys[::st], xs[::st]
    v = np.full(len(ys), -1.0, np.float32)
    okh = mv[ys, xs - w] & mv[ys, xs + w] & (vidx[ys, xs - w] != vidx[ys, xs + w])
    v[okh] = np.abs(g[ys[okh], xs[okh] + w] - g[ys[okh], xs[okh] - w])
    okv = mv[ys - w, xs] & mv[ys + w, xs] & (vidx[ys - w, xs] != vidx[ys + w, xs])
    hv = np.abs(g[ys[okv] + w, xs[okv]] - g[ys[okv] - w, xs[okv]])
    v[okv] = np.maximum(v[okv], hv)
    good = v >= 0
    return v[good], int(bx.sum())


def grab_dji(E0, N0, E1, N1, n):
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(DJI) as d:
        w = from_bounds(E0, N0, E1, N1, d.transform)
        nb = min(3, d.count)
        a = d.read(indexes=tuple(range(1, nb + 1)), window=w,
                   out_shape=(nb, n, n), resampling=rasterio.enums.Resampling.average)
    return np.stack([a[i] for i in (2, 1, 0)], 2)


def label(img, txt, h=30, scale=0.6):
    bar = np.full((h, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, txt, (6, h - 9), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def main():
    E0 = float(sys.argv[1]); N1 = float(sys.argv[2]); size = float(sys.argv[3])
    tag = sys.argv[4] if len(sys.argv) > 4 else "w"
    N0, E1 = N1 - size, E0 + size
    W = int(round(size / GSD)); H = W

    meta = load_json(os.path.join(WORK, "meta.json"))
    cj = load_json(os.path.join(WORK, "camera.json"))
    cam_p = (cj["w"], cj["h"], *cj["init"])
    paths = [m["path"] for m in meta]
    n = len(meta)
    Rs = np.load(os.path.join(WORK, "ba_R.npy"))
    Cs = np.load(os.path.join(WORK, "ba_C.npy"))
    gains = np.load(os.path.join(WORK, "gains.npy"))
    radio = np.load(os.path.join(WORK, "radio.npy"))
    dm2 = load_json(os.path.join(WORK, "dsm_meta_v2.json"))
    dsm_v2 = os.path.join(WORK, "dsm_v2.npy")
    z_fb = float(dm2["z_median"])
    om = load_json(os.path.join(OUT, "ortho_meta.json"))
    gx0, gy0 = om["x0"], om["y0"]

    ortho_ids = [i for i in range(n) if -Rs[i][2, 2] > 0.85]
    bboxes = np.load(os.path.join(WORK, "ortho_bboxes_v2.npy"))
    c0 = int((E0 - gx0) / GSD); r0 = int((gy0 - N1) / GSD)
    cands = []
    for i in ortho_ids:
        b0, rr0, b1, rr1 = bboxes[i]
        ox = max(0, min(b1, c0 + W) - max(b0, c0))
        oy = max(0, min(rr1, r0 + H) - max(rr0, r0))
        if ox > 0 and oy > 0:
            cands.append(((ox * oy) / float(W * H) * (0.25 + 0.75 * (-Rs[i][2, 2])), i))
    cands.sort(key=lambda t: -t[0])
    views = [i for _, i in cands[:MAX_VIEWS]]
    print(f"窗口 {E0:.0f},{N1:.0f} {size:.0f}m -> {W}x{H}px; 候选 {len(cands)} 用 {len(views)}: {views}")

    S.TRACK2 = True          # 强制记录次优，便于离线试各种 eps
    S.SEAM_EPS = 1.0
    cache = os.path.join(WORK, "seamlab", tag)
    keys = ("c1", "c2", "s1", "s2", "vi", "vm", "av")
    if all(os.path.exists(os.path.join(cache, k + ".npy")) for k in keys):
        c1, c2, s1, s2, vi, vm, av = [np.load(os.path.join(cache, k + ".npy")) for k in keys]
        print(f"  命中缓存 {cache} (跳过渲染)")
    else:
        c1, c2, s1, s2, vi, vm, av = blocks_render(
            E0, gy0 - r0 * GSD, W, H, views, cam_p, Rs, Cs, paths, gains,
            dsm_v2, dm2, z_fb, radio)
        os.makedirs(cache, exist_ok=True)
        for k, arr in zip(keys, (c1, c2, s1, s2, vi, vm, av)):
            np.save(os.path.join(cache, k + ".npy"), arr)
        print(f"  已缓存到 {cache}")

    sigma = LF_M / GSD
    variants = []
    variants.append(("eps=0  (v3 现状)", c1.copy()))
    for eps in (0.002, 0.005, 0.012):
        variants.append((f"eps={eps}",
                         mix_eps(c1, c2, s1, s2, eps, vm=vm)))
    variants.append(("eps=0.012 自适应", mix_eps(c1, c2, s1, s2, 0.012, adapt=2.0, vm=vm)))
    variants.append(("eps=0.012 带内用共识", mix_eps(c1, c2, s1, s2, 0.012, use_avg=av, vm=vm)))

    # 统一叠低频色调共识
    outs = []
    for nm, im in variants:
        im2 = lf_consensus(im, av, vm, sigma)
        v, nbx = cross_step_metric(gray(im2), vi, vm)
        outs.append((nm, im2))
        print(f"  {nm:20s} 跨线落差 中位 {np.median(v):5.2f}  p90 {np.percentile(v,90):6.2f} "
              f" p99 {np.percentile(v,99):6.2f}  (线像素 {nbx})")

    dj = grab_dji(E0, N0, E1, N1, 1200)

    def make_row(items, w=760, h=760, ncol=4):
        ps = [label(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA), nm) for nm, im in items]
        while len(ps) < ncol:
            ps.append(np.full((h + 30, w, 3), 240, np.uint8))
        sep = lambda p: np.concatenate([p, np.full((p.shape[0], 5, 3), 240, np.uint8)], 1)
        return np.concatenate([sep(p) for p in ps[:-1]] + [ps[-1]], 1)

    rows = [make_row(outs[0:4]),
            make_row([outs[4], outs[5], ("DJI 对照", dj)])]
    gap = np.full((8, rows[0].shape[1], 3), 240, np.uint8)
    fig = np.concatenate([rows[0], gap, rows[1]], 0)

    # 放大窗（每个面板都覆盖同样地理范围，按其自身像素比例取中心 24m）
    def zoom_of(im, meters=24.0, out=500):
        hz = max(2, int(round(meters / size * im.shape[0])) // 2)
        cy, cx = im.shape[0] // 2, im.shape[1] // 2
        z = im[cy - hz:cy + hz, cx - hz:cx + hz]
        return cv2.resize(z, (out, out), interpolation=cv2.INTER_NEAREST)

    zooms = []
    for nm, im in [("eps=0", outs[0][1]), ("eps=0.002", outs[1][1]),
                   ("eps=0.005", outs[2][1]), ("eps=0.012", outs[3][1]),
                   ("eps=自适应", outs[4][1]), ("eps=共识", outs[5][1]),
                   ("DJI", dj)]:
        zooms.append(label(zoom_of(im), nm))
    zrow = np.concatenate([np.concatenate([p, np.full((p.shape[0], 5, 3), 240, np.uint8)], 1)
                           for p in zooms[:-1]] + [zooms[-1]], 1)
    np.save(os.path.join(OUT, f"seamlab_{tag}_s1.npy"), s1)
    np.save(os.path.join(OUT, f"seamlab_{tag}_s2.npy"), s2)
    imwrite_any(os.path.join(OUT, f"seam_lab_{tag}.jpg"), fig, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    imwrite_any(os.path.join(OUT, f"seam_lab_{tag}_zoom.jpg"), zrow, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    print(f"已写 seam_lab_{tag}.jpg 与 seam_lab_{tag}_zoom.jpg")
    d = s1 - s2
    print(f"  s1-s2 分布: p10 {np.percentile(d[vm],10):.5f} p50 {np.percentile(d[vm],50):.5f} "
          f"p90 {np.percentile(d[vm],90):.5f}")


if __name__ == "__main__":
    main()
