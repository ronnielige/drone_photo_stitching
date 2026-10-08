# -*- coding: utf-8 -*-
"""Step6d: 接缝线优化（seam-line optimization，逐标签对二值 min-cut）。

【问题】硬性单帧选取（step6b）必然产生「砖墙网格」状缝合线：每条航带内影像被
切成竖条、航带之间是横贯全区的水平直线。这些直线与地物无关，会横切房屋、道路；
在 ~1px BA 残差的放大下形成肉眼可见的「断口」（见 out/fig_hotspots.jpg #1，
水平缝合线把一栋房子拦腰切断）。

【方法】能量最小化
    E = Σ_p D(p, l_p) + Σ_(p,q) V(l_p, l_q)
      D(p, a) = λ·(max(s_a, s_b) − s_a)    偏离更优视图的几何代价（天底角/径向）
      V(a, b) = |col_a(p) − col_b(p)|       两视图在该像素的颜色差（含错位效应）
  对每一对相邻视图 (a,b) 在「缝合线 ±BAND_W」的窄带内求解二值 s-t 最小割：
  带外像素标签固定（硬锚定），于是最优点会把缝合线「弯进」颜色差异最小的路径 ——
  沿平坦农田/道路/阴影穿行，绕开屋檐、田埂等强对比地物。

【为什么不是全局 alpha-expansion】标签集太大（每块 10~15 个）时求解代价过高；
逐对二值割 + 迭代 2 轮在本问题上收敛良好（相邻标签对之间基本解耦）。

用法：
  python step6d_seamcut.py                      # 全图
  python step6d_seamcut.py win E0 N1 size       # 只在窗口内做（快速验证）
产出 out/ortho_vidx_cut.npy, out/ortho_rgb_cut.npy
"""
import os
import sys
import math
import time
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, OUT, log, load_json, save_json
import step6b_ortho as S6

# ---------------- 参数 ----------------
TILE = 1024
HALO = 96                # 块外扩（仅供投影/邻域计算）
BAND_W = float(os.environ.get("BAND_W", "160"))   # 缝合线可移动范围(px, 8.8m)
LAM_D = float(os.environ.get("LAM_D", "800"))     # 数据项权重
V_EPS = float(os.environ.get("V_EPS", "4"))       # 平滑项常数（防止零代价乱走）
EDGE_W = float(os.environ.get("EDGE_W", "0.8"))   # 边缘惩罚权重（缝线避开强边缘）
EDGE_CAP = float(os.environ.get("EDGE_CAP", "150"))
FEATHER = float(os.environ.get("FEATHER", "3"))   # 缝线两侧软过渡半宽(px)
SCALE = 8                # 容量量化尺度
INF_Q = 10 ** 7          # "禁止"标签的容量
MIN_SEAM = int(os.environ.get("MIN_SEAM", "40"))  # 一个标签对最少接缝像素数
N_ITER = int(os.environ.get("N_ITER", "2"))


def build_ctx():
    meta = load_json(os.path.join(WORK, "meta.json"))
    cj = load_json(os.path.join(WORK, "camera.json"))
    om = load_json(os.path.join(OUT, "ortho_meta.json"))
    dm2 = load_json(os.path.join(WORK, "dsm_meta_v2.json"))
    radio_p = os.path.join(WORK, "radio.npy")
    dsm_p = os.path.join(WORK, "dsm_v2.npy")
    return dict(
        cam_p=(cj["w"], cj["h"], *cj["init"]),
        Rs=np.load(os.path.join(WORK, "ba_R.npy")),
        Cs=np.load(os.path.join(WORK, "ba_C.npy")),
        paths=[m["path"] for m in meta],
        gains=np.load(os.path.join(WORK, "gains.npy")),
        radio=np.load(radio_p) if os.path.exists(radio_p) else None,
        dm2=dm2, z_fb=float(dm2["z_median"]),
        dsm_path=dsm_p, dsm=np.load(dsm_p),
        gx0=om["x0"], gy0=om["y0"], gsd=om["gsd"],
        W=om["w"], H=om["h"],
    )


def stats(V, M):
    u, c = np.unique(V[M], return_counts=True)
    return len(u), float(M.mean())


# ---------------- 单块处理 ----------------
def process_block(x0, y0, x1, y1, V, M, RGB, ctx, band=BAND_W, halo=HALO):
    """就地优化 [x0,x1)×[y0,y1) 内的接缝（halo 仅供投影与邻域计算）。返回改动像素数。"""
    H, W = V.shape
    rx0, ry0 = max(0, x0 - halo), max(0, y0 - halo)
    rx1, ry1 = min(W, x1 + halo), min(H, y1 + halo)
    v = V[ry0:ry1, rx0:rx1]
    m = M[ry0:ry1, rx0:rx1]
    rgb = RGB[ry0:ry1, rx0:rx1]
    lh, lw = v.shape
    yy, xx = np.mgrid[0:lh, 0:lw]
    core = (yy >= y0 - ry0) & (yy < y1 - ry0) & (xx >= x0 - rx0) & (xx < x1 - rx0)

    # --- 收集需要处理的标签对（按接缝像素数降序）---
    def pairs():
        res = {}
        ne = (v[:, 1:] != v[:, :-1]) & m[:, 1:] & m[:, :-1]
        i2, j2 = np.nonzero(ne)
        A = v[i2, j2].astype(np.int64); B = v[i2, j2 + 1].astype(np.int64)
        k = np.minimum(A, B) * 40000 + np.maximum(A, B)
        k2 = k[(core[i2, j2] | core[i2, j2 + 1])]
        ne = (v[1:, :] != v[:-1, :]) & m[1:, :] & m[:-1, :]
        i3, j3 = np.nonzero(ne)
        A = v[i3, j3].astype(np.int64); B = v[i3 + 1, j3].astype(np.int64)
        k = np.minimum(A, B) * 40000 + np.maximum(A, B)
        k3 = k[(core[i3, j3] | core[i3 + 1, j3])]
        if len(k2) == 0 and len(k3) == 0:
            return []
        allk = np.concatenate([k2, k3])
        uq, ct = np.unique(allk, return_counts=True)
        order = np.argsort(-ct)
        return [(int(uq[i] // 40000), int(uq[i] % 40000), int(ct[i])) for i in order]

    tot_changed = 0
    for a, b, cnt in pairs():
        if cnt < MIN_SEAM:
            break
        tot_changed += solve_pair(a, b, v, m, rgb, ctx, rx0, ry0, core, band)
    return tot_changed


def solve_pair(a, b, v, m, rgb, ctx, rx0, ry0, core, band):
    """对标签对 (a,b) 做一次二值 min-cut，就地更新 v/rgb。返回改动像素数。"""
    A = (v == a) & m
    B = (v == b) & m
    AB = A | B
    if AB.sum() < 32:
        return 0
    # 接缝像素
    sm = np.zeros_like(AB)
    s1 = A[:, 1:] & B[:, :-1]; sm[:, 1:] |= s1; sm[:, :-1] |= s1
    s2 = A[:, :-1] & B[:, 1:]; sm[:, :-1] |= s2; sm[:, 1:] |= s2
    s3 = A[1:, :] & B[:-1, :]; sm[1:, :] |= s3; sm[:-1, :] |= s3
    s4 = A[:-1, :] & B[1:, :]; sm[:-1, :] |= s4; sm[1:, :] |= s4
    if sm.sum() == 0:
        return 0
    dist = cv2.distanceTransform((~sm).astype(np.uint8), cv2.DIST_L2, 5)
    inner = cv2.erode(AB.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    region = AB & (dist <= band) & inner & core
    n = int(region.sum())
    if n < MIN_SEAM:
        return 0
    ys, xs = np.nonzero(region)

    # --- 投影 a、b 到 region 的 bbox ---
    by0, bx0 = int(ys.min()), int(xs.min())
    by1, bx1 = int(ys.max()) + 1, int(xs.max()) + 1
    bh, bw = by1 - by0, bx1 - bx0
    gsd = ctx["gsd"]
    rv = S6.render_views(ctx["gx0"] + (rx0 + bx0) * gsd,
                         ctx["gy0"] - (ry0 + by0) * gsd,
                         gsd, bw, bh, [a, b], ctx["cam_p"], ctx["Rs"], ctx["Cs"],
                         ctx["paths"], ctx["gains"], ctx["dsm_path"], ctx["dm2"],
                         ctx["z_fb"], radio=ctx["radio"], dsm=ctx["dsm"])
    if a not in rv or b not in rv:
        return 0
    ca_f, ga_f, oka_f = rv[a]
    cb_f, gb_f, okb_f = rv[b]
    ca_f = ca_f.astype(np.float32); cb_f = cb_f.astype(np.float32)
    # 数据项（未含外部边贡献）
    smax = np.maximum(ga_f, gb_f)
    d0f = (smax - ga_f) * LAM_D * SCALE
    d1f = (smax - gb_f) * LAM_D * SCALE
    d0f[~oka_f] = INF_Q
    d1f[~okb_f] = INF_Q
    # 边界代价 = 两视图颜色差 + 边缘惩罚
    #   颜色差项：缝线处两侧本就近色 -> 不可见
    #   边缘项  ：1px BA 残差在强边缘（屋檐/田埂）上最扎眼，必须让缝线避开
    gaf = ca_f.mean(2); gbf = cb_f.mean(2)
    eaf = np.hypot(cv2.Sobel(gaf, cv2.CV_32F, 1, 0, 3), cv2.Sobel(gaf, cv2.CV_32F, 0, 1, 3))
    ebf = np.hypot(cv2.Sobel(gbf, cv2.CV_32F, 1, 0, 3), cv2.Sobel(gbf, cv2.CV_32F, 0, 1, 3))
    cdf = np.abs(ca_f - cb_f).mean(2) + EDGE_W * np.minimum(np.maximum(eaf, ebf), EDGE_CAP)

    idx = np.full(v.shape, -1, np.int32)
    idx[ys, xs] = np.arange(n)
    ldy, ldx = ys - by0, xs - bx0              # region 像素在 bbox 中的位置
    cd = cdf[ldy, ldx]
    D0 = d0f[ldy, ldx].astype(np.float64)
    D1 = d1f[ldy, ldx].astype(np.float64)

    rows, cols, vals = [], [], []
    # 源/汇边（同时补零反向边，满足 maximum_flow 的对称结构要求）
    ar = np.arange(n)
    rows += [ar, np.full(n, n), np.full(n, n), ar]
    cols += [np.full(n, n), ar, ar, np.full(n, n)]
    vals += [D1, D0, np.zeros(n), np.zeros(n)]

    # --- 邻域：内部边 + 外部锚定（先统计外部贡献再加入 D）---
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        qy, qx = ys + dy, xs + dx
        inside = (qy >= 0) & (qy < v.shape[0]) & (qx >= 0) & (qx < v.shape[1])
        qyc, qxc = np.clip(qy, 0, v.shape[0] - 1), np.clip(qx, 0, v.shape[1] - 1)
        qi = np.where(inside, idx[qyc, qxc], -1)
        # (1) 邻居也在 region 内 -> 内部边
        inr = qi >= 0
        if inr.any():
            p = ar[inr]; q = qi[inr]
            wv = np.round((0.5 * (cd[inr] + cd[q]) + V_EPS) * SCALE).astype(np.int64)
            rows += [p, q]; cols += [q, p]; vals += [wv, wv]
        # (2) 邻居不在 region（标签固定为 a 或 b）-> 折成数据项
        oth = inside & (qi < 0)
        if oth.any():
            qlab = v[qyc[oth], qxc[oth]]
            # 外部邻居可能落在 bbox 之外 -> 夹取到边缘（近似，影响仅限边界一圈）
            qb_y = np.clip(qyc[oth] - by0, 0, cdf.shape[0] - 1)
            qb_x = np.clip(qxc[oth] - bx0, 0, cdf.shape[1] - 1)
            wv = (0.5 * (cd[oth] + cdf[qb_y, qb_x]) + V_EPS) * SCALE
            sel_a = qlab == a
            sel_b = qlab == b
            pos = np.flatnonzero(oth)
            D1[pos[sel_a]] += wv[sel_a]
            D0[pos[sel_b]] += wv[sel_b]

    R = np.concatenate(rows); C = np.concatenate(cols); Vv = np.concatenate(vals)
    g = csr_matrix((Vv.astype(np.int64), (R, C)), shape=(n + 2, n + 2))

    res = maximum_flow(g, n, n + 1)
    r = res.flow.tocsr()
    r.eliminate_zeros()
    order, _ = breadth_first_order(r, n, directed=True, return_predecessors=True)
    src = np.zeros(n + 2, bool)
    src[np.asarray(order, dtype=np.int64)] = True
    newlab = np.where(src[:n], np.int16(a), np.int16(b))
    oldlab = v[ys, xs]
    ch = newlab != oldlab
    nch = int(ch.sum())
    if nch == 0:
        return 0
    v[ys, xs] = newlab
    # 颜色：缝线两侧 2*FEATHER 的软过渡，把 ~1px 错位造成的"刀口"糊成渐变；
    # 带外仍是纯单帧取色，锐度不受影响
    if FEATHER > 0:
        sm2 = np.zeros_like(v, bool)
        dd_ = (v[:, 1:] != v[:, :-1]); sm2[:, 1:] |= dd_; sm2[:, :-1] |= dd_
        dd_ = (v[1:, :] != v[:-1, :]); sm2[1:, :] |= dd_; sm2[:-1, :] |= dd_
        d2 = cv2.distanceTransform((~sm2).astype(np.uint8), cv2.DIST_L2, 5)
        t = np.clip(d2[ys, xs] / FEATHER, 0.0, 1.0)
        wa = np.where(newlab == np.int16(a), 0.5 + 0.5 * t, 0.5 - 0.5 * t)[:, None]
        col_new = wa * ca_f[ldy, ldx] + (1.0 - wa) * cb_f[ldy, ldx]
    else:
        col_new = np.where((newlab == np.int16(a))[:, None], ca_f[ldy, ldx], cb_f[ldy, ldx])
    rgb[ys, xs] = np.clip(col_new, 0, 255).astype(np.uint8)
    return nch


# ---------------- 主流程 ----------------
def main():
    ctx = build_ctx()
    V = np.load(os.path.join(OUT, "ortho_vidx.npy"))
    M = np.load(os.path.join(OUT, "ortho_valid.npy"))
    RGB = np.load(os.path.join(OUT, "ortho_rgb.npy"))
    H, W = V.shape
    S6._IMG_CACHE_MAX = 20
    log(f"接缝优化: 网格 {W}x{H}, 标签 {len(np.unique(V[M]))} 个, 有效率 {M.mean()*100:.1f}%")
    log(f"参数: BAND_W={BAND_W}px({BAND_W*ctx['gsd']:.1f}m) LAM_D={LAM_D} V_EPS={V_EPS} "
        f"MIN_SEAM={MIN_SEAM} N_ITER={N_ITER}")

    # 窗口模式（快速验证）
    win = None
    if len(sys.argv) > 1 and sys.argv[1] == "win":
        E0, N1, size = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
        x0 = int((E0 - ctx["gx0"]) / ctx["gsd"])
        y0 = int((ctx["gy0"] - N1) / ctx["gsd"])
        n = int(size / ctx["gsd"])
        win = (x0, y0, min(W, x0 + n), min(H, y0 + n))
        log(f"窗口模式 {E0},{N1} {size}m -> px {win}")

    blocks = [win] if win else [(x0, y0, min(W, x0 + TILE), min(H, y0 + TILE))
                                for y0 in range(0, H, TILE) for x0 in range(0, W, TILE)]
    for it in range(N_ITER):
        t0 = time.time()
        tot = 0
        for k, (x0, y0, x1, y1) in enumerate(blocks):
            tot += process_block(x0, y0, x1, y1, V, M, RGB, ctx)
            if (k + 1) % 24 == 0 or k + 1 == len(blocks):
                log(f"  轮 {it+1}: {k+1}/{len(blocks)} 块 ({time.time()-t0:.0f}s) 已改 {tot} px")
        log(f"轮 {it+1} 完成: 改动 {tot} px ({tot/M.sum()*100:.2f}%), 用时 {time.time()-t0:.0f}s")

    np.save(os.path.join(OUT, "ortho_vidx_cut.npy"), V)
    np.save(os.path.join(OUT, "ortho_rgb_cut.npy"), RGB)
    nv, _ = stats(V, M)
    log(f"完成: 标签数 {nv}, 已写 ortho_vidx_cut.npy / ortho_rgb_cut.npy")


if __name__ == "__main__":
    main()
