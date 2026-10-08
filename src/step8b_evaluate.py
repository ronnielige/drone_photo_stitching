# -*- coding: utf-8 -*-
"""Step8b: 最终评估（诚实版）。

指标：
  1) 感知清晰度：强边缘处 |∇I| / 局部对比度（越大越锐；模糊会降低，抗混叠噪声影响小）
  2) 拼接痕迹：视图切换线处的梯度超出量（相对平移对照带）
  3) 几何错位场：分块归一化互相关测局部位移 —— 扣除全局中位位移后的残差
     即"局部扭曲/错位"的真实量级（对比大疆 DOM）
  4) 色彩一致性：有效重叠区逐通道均值差
产出：
  - metrics_v2.json
  - fig_detail_compare.jpg  （旧版 / 新版 / 大疆  三栏对照，多个局部）
  - fig_zoom_compare.jpg    （放大对照，检验"放大后是否清楚"）
需要 rasterio（venv）。
"""
import os
import sys
import json
import math
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, imwrite_any

ROOT = r"D:/lr/project/无人机航拍拼接"
DJI = os.path.join(ROOT, "xiangjia1", "map", "result.tif")
MINE = os.path.join(OUT, "xiangjia_ortho_mine.tif")
V1 = os.path.join(OUT, "xiangjia_ortho_v1.tif")
LOG = print


def label(img, txt, h=36):
    bar = np.full((h, img.shape[1], 3), 25, np.uint8)
    cv2.putText(bar, txt, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def read_grid(path, E0, N1, tw, th, gsd, alpha=False):
    """把影像重采样到目标网格（原生读出后 resize，避免二次模糊）"""
    E1, N0 = E0 + tw * gsd, N1 - th * gsd
    with rasterio.open(path) as ds:
        win = from_bounds(E0, N0, E1, N1, ds.transform)
        nb = min(3, ds.count)
        a = ds.read(indexes=tuple(range(1, nb + 1)), window=win)
        av = ds.read(indexes=4, window=win) > 128 if (alpha and ds.count >= 4) else None
    h0, w0 = a.shape[1], a.shape[2]
    interp = cv2.INTER_AREA if (th < h0 and tw < w0) else cv2.INTER_LINEAR
    bgr = np.stack([cv2.resize(a[i], (tw, th), interpolation=interp)
                    for i in (2, 1, 0)], axis=2)
    if av is not None:
        av = cv2.resize(av.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST) > 0
    return bgr, av


def lapvar(img, m=None):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    l = cv2.Laplacian(g, cv2.CV_32F)
    return float(l[m].var()) if m is not None else float(l.var())


def edge_sharpness(img, m=None, q=90):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3), cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
    k = 9
    rng = np.maximum(cv2.dilate(g, np.ones((k, k), np.uint8)) -
                     (-cv2.dilate(-g, np.ones((k, k), np.uint8))), 1.0)
    r = mag / rng
    thr = np.percentile(mag[m] if m is not None else mag, q)
    sel = mag >= thr
    if m is not None:
        sel &= m
    return float(np.median(r[sel]))


def seam_excess(img, bx, m, shift=7):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3), cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
    k = np.ones((5, 5), np.uint8)
    bd = cv2.dilate(bx.astype(np.uint8), k).astype(bool) & m
    ctrl = cv2.dilate(np.roll(bx, shift, axis=0).astype(np.uint8), k).astype(bool) & m & ~bd
    if bd.sum() < 50 or ctrl.sum() < 50:
        return None
    return float(grad[bd].mean() - grad[ctrl].mean()), int(bd.sum())


def shift_field(mine, dji, m, bs=128, rad=8):
    """分块互相关求局部位移；返回 (中位位移, 残差分布)"""
    gm = cv2.cvtColor(mine, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gd = cv2.cvtColor(dji, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = lambda a: a - cv2.GaussianBlur(a, (0, 0), 3.0)
    hm, hd = hp(gm), hp(gd)
    H, W = gm.shape
    shifts = []
    for by in range(rad, H - bs - rad, bs):
        for bx in range(rad, W - bs - rad, bs):
            if not m[by:by + bs, bx:bx + bs].all():
                continue
            tpl = hm[by:by + bs, bx:bx + bs]
            if tpl.std() < 2.0:
                continue
            reg = hd[by - rad:by + bs + rad, bx - rad:bx + bs + rad]
            res = cv2.matchTemplate(reg, tpl, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if mv < 0.35:
                continue
            shifts.append((ml[0] - rad, ml[1] - rad))
    if len(shifts) < 8:
        return None
    s = np.array(shifts, np.float32)
    med = np.median(s, axis=0)
    res = s - med
    mag = np.hypot(res[:, 0], res[:, 1])
    return dict(n=len(s), median_shift=[float(med[0]), float(med[1])],
                res_p50=float(np.percentile(mag, 50)),
                res_p90=float(np.percentile(mag, 90)),
                res_max=float(mag.max()))


def main():
    om = json.load(open(os.path.join(OUT, "ortho_meta.json"), encoding="utf-8"))
    gsd, gx0, gy0, W, H = om["gsd"], om["x0"], om["y0"], om["w"], om["h"]
    vidx = np.load(os.path.join(OUT, "ortho_vidx.npy"), mmap_mode="r")
    vmask = np.load(os.path.join(OUT, "ortho_valid.npy"), mmap_mode="r")
    LOG(f"我的正射: {W} x {H} @ {gsd*100:.2f}cm, 有效率 {np.array(vmask[::17,::17]).mean()*100:.1f}%")

    with rasterio.open(DJI) as dd:
        db = dd.bounds
        dgsd = dd.transform.a
        al = dd.read(indexes=4, out_shape=(1200, 1200)) > 128
    LOG(f"大疆正射: {db.left:.0f}..{db.right:.0f} E, {db.bottom:.0f}..{db.top:.0f} N @ {dgsd*100:.2f}cm")

    # 重叠区
    ox0 = max(gx0, db.left); ox1 = min(gx0 + W * gsd, db.right)
    oy1 = min(gy0, db.top); oy0 = max(gy0 - H * gsd, db.bottom)
    LOG(f"重叠区 {ox1-ox0:.0f} x {oy1-oy0:.0f} m")
    Wp = int((ox1 - ox0) / gsd); Hp = int((oy1 - oy0) / gsd)
    LOG(f"在全重叠区按我的网格对比: {Wp} x {Hp} @ {gsd*100:.2f}cm ...")
    t0 = os.times().elapsed
    mine, av = read_grid(MINE, ox0, oy1, Wp, Hp, gsd, alpha=True)
    dji_a = None
    dji, dji_a = read_grid(DJI, ox0, oy1, Wp, Hp, gsd, alpha=True)
    v1 = None
    if os.path.exists(V1):
        v1, _ = read_grid(V1, ox0, oy1, Wp, Hp, gsd)
    LOG(f"  读取耗时 {os.times().elapsed-t0:.0f}s")

    c0, r0 = int((ox0 - gx0) / gsd), int((gy0 - oy1) / gsd)
    mv = np.array(vmask[r0:r0 + Hp, c0:c0 + Wp], dtype=bool)
    vi = np.array(vidx[r0:r0 + Hp, c0:c0 + Wp])
    m = mv & (dji_a if dji_a is not None else True)
    LOG(f"双方均有效占比 {m.mean()*100:.1f}%")

    res = {"gsd_mine": gsd, "size_mine": [W, H], "size_dji": [int(Wp), int(Hp)],
           "valid_overlap_pct": float(m.mean() * 100)}
    LOG("\n=== 指标 ===")
    for nm, im in (("MINE v2 (单帧)", mine), ("MINE v1 (旧)", v1), ("DJI", dji)):
        if im is None:
            continue
        es = edge_sharpness(im, m)
        lv = lapvar(im, m)
        res[nm] = {"edge_sharpness": es, "lapvar": lv}
        LOG(f"  {nm:16s} 边缘锐度 {es:.3f} | 拉普拉斯方差 {lv:8.1f} | 灰度均值 "
            f"{cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[m].mean():.1f}")

    LOG("\n  拼接痕迹（视图切换线梯度超出对照带；≤0 为无附加跳变）:")
    for nm, im in (("MINE v2 (单帧)", mine), ("MINE v1 (旧)", v1), ("DJI", dji)):
        if im is None:
            continue
        bx = np.zeros_like(m)
        bx[:, 1:] |= (vi[:, 1:] != vi[:, :-1]) & mv[:, 1:] & mv[:, :-1]
        bx[1:, :] |= (vi[1:, :] != vi[:-1, :]) & mv[1:, :] & mv[:-1, :]
        r = seam_excess(im, bx, m)
        if r:
            res[nm]["seam_excess"] = r[0]
            LOG(f"  {nm:16s} {r[0]:+7.2f}  (切换线 {r[1]} px, 占 {r[1]/mv.sum()*100:.2f}%)")

    LOG("\n  几何局部错位（分块互相关，扣除全局中位位移后的残差，单位=我的像素@5.5cm）:")
    for nm, im in (("MINE v2 (单帧)", mine), ("MINE v1 (旧)", v1)):
        if im is None:
            continue
        r = shift_field(im, dji, m)
        if r:
            res[nm]["shift_field"] = r
            LOG(f"  {nm:16s} n={r['n']} 全局中位位移 {r['median_shift']} | "
                f"残差 p50 {r['res_p50']*gsd*100:.2f}cm p90 {r['res_p90']*gsd*100:.2f}cm "
                f"max {r['res_max']*gsd*100:.2f}cm")

    LOG("\n  色彩（相对大疆，有效区逐通道均值差 BGR）:")
    for nm, im in (("MINE v2 (单帧)", mine), ("MINE v1 (旧)", v1)):
        if im is None:
            continue
        d = im[m].astype(np.float32) - dji[m].astype(np.float32)
        res[nm]["color_delta_bgr"] = [round(float(d[:, i].mean()), 2) for i in range(3)]
        res[nm]["color_std_bgr"] = [round(float(d[:, i].std()), 2) for i in range(3)]
        LOG(f"  {nm:16s} ΔBGR {res[nm]['color_delta_bgr']}  σ {res[nm]['color_std_bgr']}")

    json.dump(res, open(os.path.join(OUT, "metrics_v2.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ---------- 局部对照图 ----------
    def win_ok(E, N, half):
        x = int((E - ox0) / gsd); y = int((oy1 - N) / gsd)
        n = int(2 * half / gsd)
        if x < n or y < n or x + n > Wp or y + n > Hp:
            return False
        return m[y - n // 2:y + n // 2, x - n // 2:x + n // 2].mean() > 0.995

    # 候选窗口：在大疆有效区内、纹理丰富处分散选取
    cands = []
    for fy in np.linspace(0.12, 0.88, 11):
        for fx in np.linspace(0.12, 0.88, 11):
            E = ox0 + fx * (ox1 - ox0)
            N = oy1 - fy * (oy1 - oy0)
            if not win_ok(E, N, 22.0):
                continue
            x = int((E - ox0) / gsd); y = int((oy1 - N) / gsd)
            sub = mine[y - 100:y + 100, x - 100:x + 100]
            cands.append((float(cv2.Laplacian(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
                                             .astype(np.float32), cv2.CV_32F).var()), E, N))
    cands.sort(reverse=True)
    picks = []
    for _, E, N in cands:
        if all(math.hypot(E - a, N - b) > 200 for a, b in picks):
            picks.append((E, N))
        if len(picks) >= 3:
            break
    LOG(f"\n对照窗口: {[(round(a), round(b)) for a, b in picks]}")

    rows = []
    for k, (E, N) in enumerate(picks):
        half = 20.0
        n = int(2 * half / gsd)
        x = int((E - ox0) / gsd) - n // 2
        y = int((oy1 - N) / gsd) - n // 2
        pn = [label(mine[y:y + n, x:x + n], f"#{k+1} MINE v2: single-best-view")]
        if v1 is not None:
            pn.append(label(v1[y:y + n, x:x + n], "MINE v1: multi-view average"))
        pn.append(label(dji[y:y + n, x:x + n], "DJI 智图 5.62cm"))
        rows.append(np.concatenate(pn, axis=1))
    if rows:
        fig = np.concatenate([r for pair in zip(rows, [np.full((10, rows[0].shape[1], 3), 245, np.uint8)] * len(rows)) for r in pair][:-1], axis=0)
        imwrite_any(os.path.join(OUT, "fig_detail_compare.jpg"), fig,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        LOG(f"细节对照图: {fig.shape[1]} x {fig.shape[0]}")

    # 放大对照（检验放大后是否清楚）
    zr = []
    for k, (E, N) in enumerate(picks[:2]):
        half = 7.0
        n = int(2 * half / gsd)
        x = int((E - ox0) / gsd) - n // 2
        y = int((oy1 - N) / gsd) - n // 2
        zf = lambda im: cv2.resize(im[y:y + n, x:x + n], (760, 760), interpolation=cv2.INTER_CUBIC)
        pn = [label(zf(mine), f"#{k+1} MINE v2  x{760/n:.1f}")]
        if v1 is not None:
            pn.append(label(zf(v1), f"MINE v1  x{760/n:.1f}"))
        pn.append(label(zf(dji), f"DJI  x{760/n:.1f}"))
        zr.append(np.concatenate(pn, axis=1))
    if zr:
        zfig = np.concatenate(zr, axis=0)
        imwrite_any(os.path.join(OUT, "fig_zoom_compare.jpg"), zfig,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        LOG(f"放大对照图: {zfig.shape[1]} x {zfig.shape[0]}")


if __name__ == "__main__":
    main()
