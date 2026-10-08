# -*- coding: utf-8 -*-
"""正射渲染 v2 局部诊断：硬性单帧 vs 多视平均 vs 大疆 vs 旧版。

用法: python diag_v2.py E N [half_m] [--dsm v2|v1]
"""
import os
import sys
import argparse
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, OUT, load_json, Camera, imread_any, imwrite_any
from step6b_ortho import render_rect

DJI_TIF = r"D:/lr/project/无人机航拍拼接/xiangjia1/map/result.tif"
V1_TIF = os.path.join(OUT, "xiangjia_ortho_mine.tif")
GSD = 0.055


def read_geo(path, E, N, half, gsd, tw, th, alpha=False):
    """按原生分辨率读出窗口，再重采样到目标网格（避免二次模糊）"""
    with rasterio.open(path) as ds:
        nb = min(3, ds.count)
        win = from_bounds(E - half, N - half, E + half, N + half, ds.transform)
        a = ds.read(indexes=tuple(range(1, nb + 1)), window=win)
        av = None
        if alpha and ds.count >= 4:
            av = ds.read(indexes=4, window=win) > 128
    h0, w0 = a.shape[1], a.shape[2]
    interp = cv2.INTER_AREA if (th < h0 and tw < w0) else cv2.INTER_LINEAR
    ch = [cv2.resize(a[i], (tw, th), interpolation=interp) for i in range(nb)]
    bgr = np.stack(ch[::-1], axis=2).copy()      # RGB -> BGR
    if av is not None:
        av = cv2.resize(av.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST) > 0
    return bgr, av


def lapvar(img, m=None):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    l = cv2.Laplacian(g, cv2.CV_32F)
    return float(l[m].var()) if m is not None else float(l.var())


def edge_sharpness(img, m=None, q=0.90):
    """感知型清晰度：强边缘处 |∇I| / 局部对比度 的中位数。
    模糊会拉宽边缘 -> 该比值下降；噪声抬升该比值。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    mag = np.hypot(gx, gy)
    k = 9
    loc_max = cv2.dilate(g, np.ones((k, k), np.uint8))
    loc_min = -cv2.dilate(-g, np.ones((k, k), np.uint8))
    contrast = np.maximum(loc_max - loc_min, 1.0)
    r = mag / contrast
    thr = np.percentile(mag[m] if m is not None else mag, q * 100)
    sel = mag >= thr
    if m is not None:
        sel &= m
    return float(np.median(r[sel])), int(sel.sum())


def seam_excess(img, bx, m, shift=7):
    """视图边界带的梯度超出量：边界带梯度均值 - 平移对照带梯度均值"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3), cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
    k = np.ones((5, 5), np.uint8)
    bd = cv2.dilate(bx.astype(np.uint8), k).astype(bool) & m
    ctrl = cv2.dilate(np.roll(bx, shift, axis=0).astype(np.uint8), k).astype(bool) & m & ~bd
    if bd.sum() < 20 or ctrl.sum() < 20:
        return None
    return float(grad[bd].mean() - grad[ctrl].mean()), int(bd.sum())


def label(img, txt, h=34):
    bar = np.full((h, img.shape[1], 3), 28, np.uint8)
    cv2.putText(bar, txt, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("E", type=float)
    ap.add_argument("N", type=float)
    ap.add_argument("half", type=float, nargs="?", default=15.0)
    ap.add_argument("--dsm", default="v2")
    ap.add_argument("--zoom", type=float, default=2.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    meta = load_json(os.path.join(WORK, "meta.json"))
    cj = load_json(os.path.join(WORK, "camera.json"))
    cam_p = (cj["w"], cj["h"], *cj["init"])
    paths = [m["path"] for m in meta]
    Rs = np.load(os.path.join(WORK, "ba_R.npy"))
    Cs = np.load(os.path.join(WORK, "ba_C.npy"))
    gains = np.load(os.path.join(WORK, "gains.npy"))
    if a.dsm == "v2":
        dsm_path = os.path.join(WORK, "dsm_v2.npy")
        dm = load_json(os.path.join(WORK, "dsm_meta_v2.json"))
    else:
        dsm_path = os.path.join(WORK, "dsm.npy")
        dm = load_json(os.path.join(WORK, "dsm_meta.json"))
    if not os.path.exists(dsm_path):
        raise SystemExit("缺少 " + dsm_path)

    tw = th = int(round(2 * a.half / GSD))
    gx0 = a.E - a.half
    gy0 = a.N + a.half

    # 候选视图：水平距离最近的若干近似正射影像
    ids = [i for i in range(len(meta)) if -Rs[i][2, 2] > 0.85]
    dd = sorted(((np.linalg.norm(Cs[i][:2] - [a.E, a.N]), i) for i in ids))[:16]
    views = [i for _, i in dd]
    print(f"窗口 {2*a.half:.0f}m @ {GSD*100:.1f}cm -> {tw}x{th}")
    print("候选视图:", views)
    del dd

    hard, vidx, vmask, s1, avg = render_rect(
        gx0, gy0, GSD, tw, th, views, cam_p, Rs, Cs, paths, gains,
        dsm_path, dm, dm["z_median"], also_avg=True)
    uu, cc = np.unique(vidx[vmask], return_counts=True)
    print("视图使用:", ", ".join(f"#{int(u)}:{c}" for u, c in zip(uu, cc)))

    dji, dji_v = read_geo(DJI_TIF, a.E, a.N, a.half, GSD, tw, th, alpha=True)
    v1 = None
    if os.path.exists(V1_TIF):
        v1, _ = read_geo(V1_TIF, a.E, a.N, a.half, GSD, tw, th)

    m = vmask if dji_v is None else (vmask & dji_v)
    print(f"\n有效区 {m.sum()} px   指标: [拉普拉斯方差 | 边缘锐度(越大越锐) | 边缘宽度(越小越锐)]")
    names = ["硬性单帧", "多视平均", "大疆"]
    imgs = [hard, avg, dji]
    if v1 is not None and v1.shape == hard.shape:
        names.append("旧版v1")
        imgs.append(v1)
    for nm, im in zip(names, imgs):
        es, ne = edge_sharpness(im, m)
        print(f"  {nm:8s} {lapvar(im, m):8.1f} | {es:6.3f} (n={ne}) | {1.0/max(es,1e-9):6.2f}")

    # 拼接痕迹
    bx = np.zeros_like(vmask)
    bx[:, 1:] |= (vidx[:, 1:] != vidx[:, :-1]) & vmask[:, 1:] & vmask[:, :-1]
    bx[1:, :] |= (vidx[1:, :] != vidx[:-1, :]) & vmask[1:, :] & vmask[:-1, :]
    print("\n拼接痕迹（边界带梯度超出对照带的量，≤0 表示无附加跳变）:")
    for nm, im in zip(names, imgs):
        r = seam_excess(im, bx, m)
        print(f"  {nm:8s} " + ("n/a" if r is None else f"{r[0]:+7.2f}  (边界带 {r[1]} px)"))

    z = max(1, int(round(a.zoom)))
    sep = np.full((th, 6, 3), 250, np.uint8)
    row1 = np.concatenate([label(hard, "MINE: single-best-view"),
                           label(avg, "MINE: multi-view average"),
                           label(dji, "DJI")], axis=1)
    c0 = tw // 2 - tw // 5
    c1 = tw // 2 + tw // 5
    r0 = th // 2 - th // 5
    r1 = th // 2 + th // 5
    zc = lambda im: cv2.resize(im[r0:r1, c0:c1], (tw, th), interpolation=cv2.INTER_NEAREST)
    row2 = np.concatenate([label(zc(hard), f"single-best x{tw/(c1-c0):.1f}"),
                           label(zc(avg), "average zoom"),
                           label(zc(dji), "DJI zoom")], axis=1)
    canvas = np.concatenate([row1,
                             np.full((8, row1.shape[1], 3), 250, np.uint8),
                             row2], axis=0)
    tag = a.tag or f"{a.E:.0f}_{a.N:.0f}"
    p = os.path.join(OUT, f"diag_v2_{tag}.jpg")
    imwrite_any(p, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print("\n已保存", p, canvas.shape)

    # 视图索引图
    vv = vidx.copy()
    rng = np.random.RandomState(7)
    lut = rng.randint(40, 255, (int(vv.max()) + 2, 3), dtype=np.uint8)
    vis = np.zeros((th, tw, 3), np.uint8)
    ok = vv >= 0
    vis[ok] = lut[vv[ok]]
    imwrite_any(os.path.join(OUT, f"diag_v2_{tag}_vidx.png"), vis)


if __name__ == "__main__":
    main()
