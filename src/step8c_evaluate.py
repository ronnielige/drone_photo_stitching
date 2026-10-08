# -*- coding: utf-8 -*-
"""Step8c: 终评（诚实版 v2）—— 四方对照。

候选：
  A. MINE v3  = 硬性单帧细节 + 多视低频色调共识（本轮方案）
  B. MINE v2  = 纯硬性单帧（无色调共识）
  C. MINE v1  = 多视加权平均
  D. DJI 智图 5.62cm

指标：
  1) edge_sharpness  强边缘处 |∇I|/局部对比度（越大越锐；抗混叠影响小）
  2) lapvar          拉普拉斯方差（参考，易被噪声误导）
  3) block_edge  ★   低频色块指标：LP(1.5m) 的梯度在「视图切换线」上相对对照带的
                     超出量。专门捕捉"平滑但可见"的色块台阶 —— 旧指标抓不到。
  4) seam_edge       原始梯度在切换线上的超出量（锐利跳变）
  5) shift_field     分块互相关几何错位残差（扣除全局中位位移；搜索半径 ±2.2m）
  6) color_delta     与大疆的逐通道均值差
产出 metrics_v3.json + 对照图。
需要 venv(rasterio)。
"""
import os
import sys
import json
import math
import time
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, imwrite_any

ROOT = r"D:/lr/project/无人机航拍拼接"
DJI = os.path.join(ROOT, "xiangjia1", "map", "result.tif")
V3 = os.path.join(OUT, "xiangjia_ortho_mine.tif")
V2P = os.path.join(OUT, "xiangjia_ortho_v2pure.tif")
V1 = os.path.join(OUT, "xiangjia_ortho_v1.tif")
LOG = print


def label(img, txt, h=34):
    bar = np.full((h, img.shape[1], 3), 25, np.uint8)
    cv2.putText(bar, txt, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (255, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def read_grid(path, E0, N1, tw, th, gsd, alpha=False, interp=None):
    E1, N0 = E0 + tw * gsd, N1 - th * gsd
    with rasterio.open(path) as ds:
        win = from_bounds(E0, N0, E1, N1, ds.transform)
        nb = min(3, ds.count)
        a = ds.read(indexes=tuple(range(1, nb + 1)), window=win)
        av = ds.read(indexes=4, window=win) > 128 if (alpha and ds.count >= 4) else None
    h0, w0 = a.shape[1], a.shape[2]
    it = interp or (cv2.INTER_AREA if (th < h0 and tw < w0) else cv2.INTER_LINEAR)
    bgr = np.stack([cv2.resize(a[i], (tw, th), interpolation=it) for i in (2, 1, 0)], axis=2)
    if av is not None:
        av = cv2.resize(av.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST) > 0
    return bgr, av


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def edge_sharpness(g, m, q=90):
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3), cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
    k = 9
    rng = np.maximum(cv2.dilate(g, np.ones((k, k), np.uint8)) -
                     (-cv2.dilate(-g, np.ones((k, k), np.uint8))), 1.0)
    r = mag / rng
    thr = np.percentile(mag[m], q)
    sel = (mag >= thr) & m
    return float(np.median(r[sel]))


def band_stats(field, bd, ctrl, q=95):
    """边界带 vs 对照带 的分位数（用 p95 而不是均值，对稀疏强跳变更敏感）"""
    if bd.sum() < 200 or ctrl.sum() < 200:
        return None
    a = np.percentile(field[bd], q)
    b = np.percentile(field[ctrl], q)
    return float(a - b), float(a), float(b), int(bd.sum())


def lp_block_metric(g, bd, ctrl, sigma_px):
    """低频(色块)梯度超出量：对 LP 图求梯度，比较边界带 vs 对照带 p95"""
    lp = cv2.GaussianBlur(g, (0, 0), sigma_px)
    mag = np.hypot(cv2.Sobel(lp, cv2.CV_32F, 1, 0, 3), cv2.Sobel(lp, cv2.CV_32F, 0, 1, 3))
    return band_stats(mag, bd, ctrl)


def seam_bands(vidx, mv, dilate=2, far=26):
    """边界带 bd：视图标签变化处膨胀 dilate 像素；
    对照带 ctrl：距离任何边界 > far 像素的内部像素（按 stride 抽样），
    这样对照带不会被别的边界污染。"""
    bx = np.zeros_like(mv)
    bx[:, 1:] |= (vidx[:, 1:] != vidx[:, :-1]) & mv[:, 1:] & mv[:, :-1]
    bx[1:, :] |= (vidx[1:, :] != vidx[:-1, :]) & mv[1:, :] & mv[:-1, :]
    k = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
    bd = cv2.dilate(bx.astype(np.uint8), k).astype(bool) & mv
    dist = cv2.distanceTransform((~bx).astype(np.uint8), cv2.DIST_L2, 5)
    ctrl = (dist > far) & mv
    ys, xs = np.nonzero(ctrl)
    st = max(1, len(ys) // 400000)
    c2 = np.zeros_like(ctrl)
    c2[ys[::st], xs[::st]] = True
    return bd, c2


def cross_step(g, vidx, mv, w=20):
    """跨切换线的可见落差：在边界像素处，沿垂直/水平方向各跨 w 像素取样，
    若两侧处于不同视图标签，则记录 |I(+w) - I(-w)|。
    这同时捕捉"平滑色块"和"细线"两类接缝 —— 梯度类指标只能捕捉后者。
    返回 (每像素落差数组, 采样位置索引)"""
    H, W = g.shape
    bx = np.zeros((H, W), bool)
    bx[:, 1:] |= (vidx[:, 1:] != vidx[:, :-1]) & mv[:, 1:] & mv[:, :-1]
    bx[1:, :] |= (vidx[1:, :] != vidx[:-1, :]) & mv[1:, :] & mv[:-1, :]
    ys, xs = np.nonzero(bx)
    sel = (ys >= w) & (ys < H - w) & (xs >= w) & (xs < W - w)
    ys, xs = ys[sel], xs[sel]
    st = max(1, len(ys) // 600000)
    ys, xs = ys[::st], xs[::st]
    L = vidx
    v = np.full(len(ys), -1.0, np.float32)
    okh = mv[ys, xs - w] & mv[ys, xs + w] & (L[ys, xs - w] != L[ys, xs + w])
    v[okh] = np.abs(g[ys[okh], xs[okh] + w] - g[ys[okh], xs[okh] - w])
    okv = mv[ys - w, xs] & mv[ys + w, xs] & (L[ys - w, xs] != L[ys + w, xs])
    hv = np.abs(g[ys[okv] + w, xs[okv]] - g[ys[okv] - w, xs[okv]])
    v[okv] = np.maximum(v[okv], hv)
    good = v >= 0
    return v[good], (ys[good], xs[good])


def shift_field(a, b, m, bs=48, rad=8, ds=8):
    """降采样 ds 倍后做分块互相关。
    （实测错位可达 1m 量级）。ds=8 -> 0.44m/px，hp sigma2 -> ~0.9m 尺度，
    block=48*8px=21m，搜索 ±8*8px=±3.5m。"""
    ga = cv2.resize(gray(a), (0, 0), fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
    gb = cv2.resize(gray(b), (0, 0), fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
    mm = cv2.resize(m.astype(np.uint8), (ga.shape[1], ga.shape[0]),
                    interpolation=cv2.INTER_NEAREST) > 0
    hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 2.0)
    ha, hb = hp(ga), hp(gb)
    H, W = ha.shape
    sh, poss = [], []
    for by in range(rad, H - bs - rad, bs):
        for bx in range(rad, W - bs - rad, bs):
            if not mm[by:by + bs, bx:bx + bs].all():
                continue
            tpl = ha[by:by + bs, bx:bx + bs]
            if tpl.std() < 2.0:
                continue
            reg = hb[by - rad:by + bs + rad, bx - rad:bx + bs + rad]
            res = cv2.matchTemplate(reg, tpl, cv2.TM_CCOEFF_NORMED)
            _, mv_, _, ml = cv2.minMaxLoc(res)
            if mv_ < 0.45:            # 低响应 = 匹配不可信（雾霭/均质农田），丢弃
                continue
            sh.append((ml[0] - rad, ml[1] - rad))
            poss.append((bx + bs / 2.0, by + bs / 2.0))
    if len(sh) < 8:
        return None
    s = np.array(sh, np.float32)
    pos = np.array([p for p in poss], np.float32)
    med = np.median(s, axis=0)
    r = s - med
    mag = np.hypot(r[:, 0], r[:, 1]) * ds
    # 再扣除「全局一阶形变」(旋转/尺度/错切)，剩下的才是真正的局部不一致
    A = np.concatenate([np.ones((len(pos), 1)), pos], axis=1)
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    r2 = s - A @ coef
    mag2 = np.hypot(r2[:, 0], r2[:, 1]) * ds
    return dict(n=len(s), median_shift_px=[float(med[0] * ds), float(med[1] * ds)],
                res_p50_px=float(np.percentile(mag, 50)),
                res_p90_px=float(np.percentile(mag, 90)),
                res_max_px=float(mag.max()),
                local_p50_px=float(np.percentile(mag2, 50)),
                local_p90_px=float(np.percentile(mag2, 90)),
                local_max_px=float(mag2.max()))


def main():
    om = json.load(open(os.path.join(OUT, "ortho_meta.json"), encoding="utf-8"))
    gsd, gx0, gy0, W, H = om["gsd"], om["x0"], om["y0"], om["w"], om["h"]
    vidx = np.load(os.path.join(OUT, "ortho_vidx.npy"), mmap_mode="r")
    vmask = np.load(os.path.join(OUT, "ortho_valid.npy"), mmap_mode="r")
    LOG(f"我的网格 {W}x{H} @{gsd*100:.2f}cm  有效率 {np.array(vmask[::17,::17]).mean()*100:.1f}%")

    with rasterio.open(DJI) as dd:
        db, dgsd = dd.bounds, dd.transform.a
    ox0 = max(gx0, db.left); ox1 = min(gx0 + W * gsd, db.right)
    oy1 = min(gy0, db.top);  oy0 = max(gy0 - H * gsd, db.bottom)
    Wp = int((ox1 - ox0) / gsd); Hp = int((oy1 - oy0) / gsd)
    LOG(f"大疆 @{dgsd*100:.2f}cm; 重叠区 {(ox1-ox0):.0f}x{(oy1-oy0):.0f} m -> 对比网格 {Wp}x{Hp}")

    t0 = time.time()
    imgs = {}
    imgs["MINE v3 单帧+色调共识"], _ = read_grid(V3, ox0, oy1, Wp, Hp, gsd)
    imgs["MINE v2 纯单帧"], _ = read_grid(V2P, ox0, oy1, Wp, Hp, gsd)
    if os.path.exists(V1):
        imgs["MINE v1 多视平均"], _ = read_grid(V1, ox0, oy1, Wp, Hp, gsd)
    dji, dji_a = read_grid(DJI, ox0, oy1, Wp, Hp, gsd, alpha=True, interp=cv2.INTER_LINEAR)
    imgs["DJI 智图"] = dji
    LOG(f"  读取 {time.time()-t0:.0f}s")

    c0, r0 = int((ox0 - gx0) / gsd), int((gy0 - oy1) / gsd)
    mv = np.array(vmask[r0:r0 + Hp, c0:c0 + Wp], dtype=bool)
    vi = np.array(vidx[r0:r0 + Hp, c0:c0 + Wp])
    m = mv & (dji_a if dji_a is not None else True)
    LOG(f"双方均有效 {m.mean()*100:.1f}%")

    sigma_px = 1.5 / gsd
    bd, ctrl = seam_bands(vi, mv, dilate=2, far=26)
    LOG(f"视图切换线 {bd.sum()} px (占有效区 {bd.sum()/mv.sum()*100:.2f}%), "
        f"远离边界的对照带 {ctrl.sum()} px")

    res = {"grid": [W, H], "gsd": gsd, "valid_pct": float(m.mean() * 100),
           "seam_px_ratio": float(bd.sum() / mv.sum())}
    # 参考基准：同样掩膜下大疆的超出量。mine - dji 才是"我引入的"接缝。
    gdji = gray(dji)
    ref_blk = lp_block_metric(gdji, bd, ctrl, sigma_px)
    ref_seam = band_stats(np.hypot(cv2.Sobel(gdji, cv2.CV_32F, 1, 0, 3),
                                   cv2.Sobel(gdji, cv2.CV_32F, 0, 1, 3)), bd, ctrl)
    LOG(f"参考(DJI 在同一掩膜上): 色块边界 {ref_blk[0]:+.3f}  锐利跳变 {ref_seam[0]:+.3f}")

    LOG("\n=== 指标 ===")
    LOG(f"{'方案':22s} {'锐度↑':>8s} {'色块边界↓':>10s} {'(相对DJI)':>10s} "
        f"{'锐利跳变↓':>10s} {'(相对DJI)':>10s} {'灰度':>7s}")
    for nm, im in imgs.items():
        g = gray(im)
        es = edge_sharpness(g, m)
        lv = float(cv2.Laplacian(g, cv2.CV_32F)[m].var())
        blk = lp_block_metric(g, bd, ctrl, sigma_px)
        sm = band_stats(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3),
                                 cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)), bd, ctrl)
        d = {"edge_sharpness": es, "lapvar": lv, "mean_gray": float(g[m].mean())}
        if blk:
            d.update(block_edge_excess=blk[0], block_edge_rel_dji=blk[0] - ref_blk[0])
        if sm:
            d.update(seam_edge_excess=sm[0], seam_edge_rel_dji=sm[0] - ref_seam[0])
        res[nm] = d
        LOG(f"{nm:22s} {es:8.3f} "
            f"{(blk[0] if blk else float('nan')):10.3f} "
            f"{(d.get('block_edge_rel_dji', float('nan'))):10.3f} "
            f"{(sm[0] if sm else float('nan')):10.3f} "
            f"{(d.get('seam_edge_rel_dji', float('nan'))):10.3f} {d['mean_gray']:7.1f}")

    LOG("\n  跨切换线可见落差（两侧各 1.1m 的灰度差；以 DJI 在同位置的值为基准）:")
    ref_v, ref_pos = cross_step(gdji, vi, mv, w=20)
    ref_map = np.full((mv.shape[0], mv.shape[1]), np.nan, np.float32)
    ref_map[ref_pos] = ref_v
    LOG(f"  DJI 基准: 中位 {np.median(ref_v):.2f}  p90 {np.percentile(ref_v,90):.2f}  "
        f"(n={len(ref_v)})")
    for nm, im in imgs.items():
        if nm.startswith("DJI"):
            continue
        v, pos = cross_step(gray(im), vi, mv, w=20)
        r = ref_map[pos]
        d = v - r
        res[nm]["cross_step"] = dict(median=float(np.median(v)),
                                     p90=float(np.percentile(v, 90)),
                                     excess_median=float(np.median(d)),
                                     excess_p90=float(np.percentile(d, 90)),
                                     n=int(len(v)))
        LOG(f"  {nm:22s} 中位 {np.median(v):6.2f}  p90 {np.percentile(v,90):6.2f} | "
            f"相对大疆 中位 {np.median(d):+5.2f}  p90 {np.percentile(d,90):+6.2f}")

    LOG("\n  几何局部错位（相对大疆，降采样 2x 互相关，搜索 ±2.2m）:")
    for nm, im in imgs.items():
        if nm.startswith("DJI"):
            continue
        r = shift_field(im, dji, m)
        if r:
            res[nm]["shift_field"] = {k: (round(v, 3) if isinstance(v, float) else v)
                                      for k, v in r.items()}
            LOG(f"  {nm:22s} n={r['n']} 全局中位位移 {[round(v,1) for v in r['median_shift_px']]}px "
                f"| 残差(去中位) p50 {r['res_p50_px']*gsd*100:.0f}cm p90 {r['res_p90_px']*gsd*100:.0f}cm"
                f" | 扣除一阶形变后 p50 {r['local_p50_px']*gsd*100:.0f}cm "
                f"p90 {r['local_p90_px']*gsd*100:.0f}cm max {r['local_max_px']*gsd*100:.0f}cm")

    LOG("\n  色彩（相对大疆，有效区逐通道均值差 BGR）:")
    for nm, im in imgs.items():
        if nm.startswith("DJI"):
            continue
        d = im[m].astype(np.float32) - dji[m].astype(np.float32)
        res[nm]["color_delta_bgr"] = [round(float(d[:, i].mean()), 2) for i in range(3)]
        res[nm]["color_std_bgr"] = [round(float(d[:, i].std()), 2) for i in range(3)]
        LOG(f"  {nm:22s} ΔBGR {res[nm]['color_delta_bgr']}  σ {res[nm]['color_std_bgr']}")

    json.dump(res, open(os.path.join(OUT, "metrics_v3.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ---------- 局部对照图（4 联）----------
    def win_ok(E, N, half):
        x = int((E - ox0) / gsd); y = int((oy1 - N) / gsd)
        n = int(2 * half / gsd)
        if x < n or y < n or x + n > Wp or y + n > Hp:
            return False
        return m[y - n // 2:y + n // 2, x - n // 2:x + n // 2].mean() > 0.995

    cands = []
    for fy in np.linspace(0.10, 0.90, 13):
        for fx in np.linspace(0.10, 0.90, 13):
            E = ox0 + fx * (ox1 - ox0); N = oy1 - fy * (oy1 - oy0)
            if not win_ok(E, N, 22.0):
                continue
            x = int((E - ox0) / gsd); y = int((oy1 - N) / gsd)
            sub = gray(imgs["MINE v2 纯单帧"][y - 110:y + 110, x - 110:x + 110])
            cands.append((float(cv2.Laplacian(sub, cv2.CV_32F).var()), E, N))
    cands.sort(reverse=True)
    picks = []
    for _, E, N in cands:
        if all(math.hypot(E - a, N - b) > 220 for a, b in picks):
            picks.append((E, N))
        if len(picks) >= 4:
            break
    LOG(f"\n对照窗口: {[(round(a), round(b)) for a, b in picks]}")

    names = list(imgs.keys())
    rows = []
    for k, (E, N) in enumerate(picks):
        half = 20.0
        n = int(2 * half / gsd)
        x = int((E - ox0) / gsd) - n // 2; y = int((oy1 - N) / gsd) - n // 2
        pn = [label(imgs[nm][y:y + n, x:x + n], f"#{k+1} " + nm) for nm in names]
        sep = np.full((pn[0].shape[0], 6, 3), 245, np.uint8)
        row = pn[0]
        for p in pn[1:]:
            row = np.concatenate([row, sep, p], axis=1)
        rows.append(row)
    if rows:
        gap = np.full((10, rows[0].shape[1], 3), 245, np.uint8)
        stack = rows[0]
        for r in rows[1:]:
            stack = np.concatenate([stack, gap, r], axis=0)
        imwrite_any(os.path.join(OUT, "fig_v3_detail.jpg"), stack,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        LOG(f"细节对照图 fig_v3_detail.jpg {stack.shape[1]}x{stack.shape[0]}")

    rows2 = []
    for k, (E, N) in enumerate(picks[:2]):
        half = 7.0
        n = int(2 * half / gsd)
        x = int((E - ox0) / gsd) - n // 2; y = int((oy1 - N) / gsd) - n // 2
        zf = lambda im: cv2.resize(im[y:y + n, x:x + n], (700, 700), interpolation=cv2.INTER_CUBIC)
        pn = [label(zf(imgs[nm]), f"#{k+1} " + nm + f"  x{700/n:.1f}") for nm in names]
        sep = np.full((pn[0].shape[0], 6, 3), 245, np.uint8)
        row = pn[0]
        for p in pn[1:]:
            row = np.concatenate([row, sep, p], axis=1)
        rows2.append(row)
    if rows2:
        stack2 = rows2[0] if len(rows2) == 1 else np.concatenate(
            [rows2[0], np.full((10, rows2[0].shape[1], 3), 245, np.uint8), rows2[1]], axis=0)
        imwrite_any(os.path.join(OUT, "fig_v3_zoom.jpg"), stack2,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        LOG(f"放大对照图 fig_v3_zoom.jpg {stack2.shape[1]}x{stack2.shape[0]}")


if __name__ == "__main__":
    main()
