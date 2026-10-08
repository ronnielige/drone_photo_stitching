"""Step6: 正射纠正(TDOM)渲染 + 曝光增益均衡 + 多视加权无缝融合。"""
import os
import math
import time
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor
from common import (WORK, OUT, log, load_json, save_json, Camera,
                    imread_any, imwrite_any, R_to_rotvec, rotvec_to_R)

# ---------------- 参数 ----------------
TILE = 1024            # 正射分块大小(像素)
MAX_VIEWS = 16         # 每个分块使用的视图数
W_BETA = 3.0           # 权重锐度
BLEND_SIGMA_M = 1.25   # 双频段融合的低频截断尺度 (m)
N_PT_GAIN = 400        # 每对用于估计增益的采样点数


# ---------------- 投影工具 ----------------
def ortho_pixel_xyz(gx0, gy0, gsd, w, h):
    """正射网格像素中心 -> 平面坐标 (E,N)"""
    xs = gx0 + (np.arange(w) + 0.5) * gsd
    ys = gy0 - (np.arange(h) + 0.5) * gsd
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def sample_dsm(dsm, dmeta, X, Y):
    """双线性采样 DSM 高度"""
    gsd = dmeta["gsd"]
    fx = (X - dmeta["x0"]) / gsd - 0.5
    fy = (dmeta["y0"] - Y) / gsd - 0.5
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    tx = fx - x0
    ty = fy - y0
    H, W = dsm.shape
    x0c = np.clip(x0, 0, W - 2)
    y0c = np.clip(y0, 0, H - 2)
    v00 = dsm[y0c, x0c]
    v01 = dsm[y0c, x0c + 1]
    v10 = dsm[y0c + 1, x0c]
    v11 = dsm[y0c + 1, x0c + 1]
    z = (v00 * (1 - tx) * (1 - ty) + v01 * tx * (1 - ty) +
         v10 * (1 - tx) * ty + v11 * tx * ty)
    return z.astype(np.float32)


def project_world(Xw, R, C, cam):
    """世界点 -> (像素u,v, 视线天底余弦, 归一化半径)"""
    Xc = (Xw - C) @ R
    z = Xc[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    xn = Xc[:, 0] / zs
    yn = Xc[:, 1] / zs
    r2 = xn * xn + yn * yn
    rad = 1.0 + cam.k1 * r2 + cam.k2 * r2 * r2 + cam.k3 * r2 * r2 * r2
    xd = xn * rad + 2 * cam.p1 * xn * yn + cam.p2 * (r2 + 2 * xn * xn)
    yd = yn * rad + cam.p1 * (r2 + 2 * yn * yn) + 2 * cam.p2 * xn * yn
    u = cam.fx * xd + cam.cx
    v = cam.fy * yd + cam.cy
    # 天底方向余弦：世界点到相机方向的 -z 分量
    d = C[None, :] - Xw
    dn = np.linalg.norm(d, axis=1)
    nadir = -d[:, 2] / np.maximum(dn, 1e-9)
    return u, v, nadir, (z > 1.0)


def image_ortho_bbox(i, Rs, Cs, cam, grid, zrange):
    """影像 i 在正射网格上的覆盖 bbox（考虑地形高度范围）"""
    w, h = cam.w, cam.h
    us = np.linspace(2, w - 3, 24)
    vs = np.linspace(2, h - 3, 18)
    border = np.concatenate([
        np.stack([us, np.full_like(us, 2.0)], 1),
        np.stack([us, np.full_like(us, h - 3.0)], 1),
        np.stack([np.full_like(vs, 2.0), vs], 1),
        np.stack([np.full_like(vs, w - 3.0), vs], 1)])
    pts = []
    for z in zrange:
        # 像素 -> 射线
        # 去畸变：由畸变像素坐标反求无畸变归一化方向（Newton 迭代）
        xn = (border[:, 0] - cam.cx) / cam.fx
        yn = (border[:, 1] - cam.cy) / cam.fy
        for _ in range(6):
            r2 = xn * xn + yn * yn
            rad = 1.0 + cam.k1 * r2 + cam.k2 * r2 * r2 + cam.k3 * r2 * r2 * r2
            xd = xn * rad + 2 * cam.p1 * xn * yn + cam.p2 * (r2 + 2 * xn * xn)
            yd = yn * rad + cam.p1 * (r2 + 2 * yn * yn) + 2 * cam.p2 * xn * yn
            xn = xn - ((cam.fx * xd + cam.cx) - border[:, 0]) / cam.fx
            yn = yn - ((cam.fy * yd + cam.cy) - border[:, 1]) / cam.fy
        dirs = np.stack([xn, yn, np.ones_like(xn)], 1) @ Rs[i].T
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        dz = np.where(np.abs(dirs[:, 2]) < 1e-3, 1e-3, dirs[:, 2])
        t = (z - Cs[i][2]) / dz
        t = np.clip(t, 0.5, 3000.0)
        P = Cs[i][None, :] + t[:, None] * dirs
        pts.append(P[:, :2])
    P = np.concatenate(pts, 0)
    gx0, gy0, gsd = grid["x0"], grid["y0"], grid["gsd"]
    col = (P[:, 0] - gx0) / gsd
    row = (gy0 - P[:, 1]) / gsd
    m = 12
    return (int(np.floor(col.min())) - m, int(np.floor(row.min())) - m,
            int(np.ceil(col.max())) + m, int(np.ceil(row.max())) + m)


# ---------------- 曝光增益均衡 ----------------
def estimate_gains(meta, feats, matches, pairs, cam, cache=None):
    """用同名点颜色估计每张影像的逐通道增益 (median-ratio + 全局最小二乘)"""
    n = len(meta)
    # 收集每张影像需采样的特征点
    need = [set() for _ in range(n)]
    for (i, j, _), m in zip(pairs, matches):
        if len(m) == 0:
            continue
        step = max(1, len(m) // N_PT_GAIN)
        sel = m[::step][:N_PT_GAIN]
        for a, b in sel:
            need[i].add(int(a))
            need[j].add(int(b))
    cols = []
    idxmap = []
    for i in range(n):
        idxs = np.array(sorted(need[i]), dtype=np.int64)
        idxmap.append({v: k for k, v in enumerate(idxs)})
        img = imread_any(meta[i]["path"])
        if img is None:
            cols.append(np.zeros((len(idxs), 3), np.float32))
            continue
        # 特征归一化坐标 -> 带畸变像素
        nx = feats[i]["nx"][idxs].astype(np.float64)
        ny = feats[i]["ny"][idxs].astype(np.float64)
        r2 = nx * nx + ny * ny
        rad = 1.0 + cam.k1 * r2 + cam.k2 * r2 * r2 + cam.k3 * r2 * r2 * r2
        xd = nx * rad + 2 * cam.p1 * nx * ny + cam.p2 * (r2 + 2 * nx * nx)
        yd = ny * rad + cam.p1 * (r2 + 2 * ny * ny) + 2 * cam.p2 * nx * ny
        u = np.clip((cam.fx * xd + cam.cx).astype(np.int32), 0, cam.w - 1)
        v = np.clip((cam.fy * yd + cam.cy).astype(np.int32), 0, cam.h - 1)
        cols.append(img[v, u].astype(np.float32))
    # 每对的中值比
    ratios = {}
    for (i, j, _), m in zip(pairs, matches):
        if len(m) == 0:
            continue
        step = max(1, len(m) // N_PT_GAIN)
        sel = m[::step][:N_PT_GAIN]
        ci = cols[i][[idxmap[i][int(a)] for a in sel[:, 0]]]
        cj = cols[j][[idxmap[j][int(b)] for b in sel[:, 1]]]
        mi = np.median(ci, axis=0)
        mj = np.median(cj, axis=0)
        ok = (mi > 12) & (mj > 12)
        r = np.where(ok, mj / np.maximum(mi, 1e-3), 1.0)
        ratios[(i, j)] = np.clip(r, 0.4, 2.5)

    # 最小二乘: log g_j - log g_i = log r_ij, 约束 sum log g = 0
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    rows, colsA, vals, bs = [], [], [], []
    for k, (i, j) in enumerate(ratios.keys()):
        r = ratios[(i, j)]
        for c in range(3):
            rows.append(k * 3 + c)
            colsA.append(j * 3 + c)
            vals.append(1.0)
            rows.append(k * 3 + c)
            colsA.append(i * 3 + c)
            vals.append(-1.0)
            bs.append(math.log(max(r[c], 1e-3)))
    A = sp.csr_matrix((vals, (rows, colsA)), shape=(len(bs), n * 3))
    b = np.array(bs)
    # 加约束 sum log g = 0（弱）
    lam = 0.5
    A2 = sp.vstack([A, lam * sp.csr_matrix((np.ones(n * 3), (np.zeros(n * 3, int), np.arange(n * 3))),
                                           shape=(1, n * 3))])
    b2 = np.concatenate([b, [0.0]])
    sol = spl.lsqr(A2, b2, atol=1e-10, btol=1e-10, iter_lim=2000)[0]
    g = np.exp(sol.reshape(n, 3))
    return g.astype(np.float32), ratios


# ---------------- 分块渲染 ----------------
def _render_tile(args):
    (tx, ty, gx0, gy0, gsd, tw, th, views, cam_p, Rs, Cs, paths, gains,
     dsm_path, dmeta, dsm_conf_path) = args
    cam = Camera(*cam_p)
    dsm = np.load(dsm_path, mmap_mode="r")
    conf = (np.load(dsm_conf_path, mmap_mode="r") if dsm_conf_path else None)
    X, Y = ortho_pixel_xyz(gx0, gy0, gsd, tw, th)
    Z = sample_dsm(dsm, dmeta, X, Y)
    Xf = X.ravel()
    Yf = Y.ravel()
    Zf = Z.ravel()
    Xw = np.stack([Xf, Yf, Zf], axis=1)

    acc = np.zeros((th * tw, 3), dtype=np.float64)
    wsum = np.zeros(th * tw, dtype=np.float64)
    wmax = np.zeros(th * tw, dtype=np.float32)
    dom_col = np.zeros((th * tw, 3), dtype=np.float64)

    for i in views:
        R, C = Rs[i], Cs[i]
        u, v, nadir, front = project_world(Xw, R, C, cam)
        dv = Xw - C[None, :]
        dvn = np.linalg.norm(dv, axis=1)
        cosn = -dv[:, 2] / np.maximum(dvn, 1e-9)
        ru = np.abs(u - cam.cx) / (cam.w / 2.0)
        rv = np.abs(v - cam.cy) / (cam.h / 2.0)
        r = np.maximum(ru, rv)
        ok = (front & (cosn > 0.35) & (u > 1) & (v > 1) &
              (u < cam.w - 2) & (v < cam.h - 2) & (r < 1.0))
        if ok.sum() == 0:
            continue
        w = np.zeros(th * tw, dtype=np.float64)
        w[ok] = (np.clip(cosn[ok], 0, 1) ** 2.0) * np.cos(np.clip(r[ok], 0, 1) * math.pi / 2) ** 0.6
        w[ok] = w[ok] ** W_BETA
        if w.max() <= 0:
            continue
        img = imread_any(paths[i])
        if img is None:
            continue
        mapx = u.reshape(th, tw).astype(np.float32)
        mapy = v.reshape(th, tw).astype(np.float32)
        samp = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0).reshape(-1, 3)
        g = gains[i]
        col = samp.astype(np.float64) * g[None, :]
        col[~ok] = 0.0
        acc += col * w[:, None]
        wsum += w
        upd = w > wmax
        wmax = np.where(upd, w, wmax)
        dom_col[upd] = col[upd]

    valid = wsum > 1e-9
    I_avg = np.zeros((th * tw, 3), dtype=np.float32)
    I_avg[valid] = (acc[valid] / wsum[valid, None]).astype(np.float32)
    I_dom = dom_col.astype(np.float32)

    # ---- 双频段融合：低频取共识平均（无缝），高频取主导视图（清晰）----
    s = int(max(5, round(BLEND_SIGMA_M / gsd)))
    if s % 2 == 0:
        s += 1
    L_avg = cv2.boxFilter(I_avg.reshape(th, tw, 3), -1, (s, s), borderType=cv2.BORDER_REFLECT)
    L_dom = cv2.boxFilter(I_dom.reshape(th, tw, 3), -1, (s, s), borderType=cv2.BORDER_REFLECT)
    fused = np.clip(L_avg.reshape(-1, 3) + (I_dom - L_dom.reshape(-1, 3)), 0, 255)

    out = np.zeros((th * tw, 3), dtype=np.uint8)
    out[valid] = fused[valid].astype(np.uint8)
    return tx, ty, out.reshape(th, tw, 3), valid.reshape(th, tw)


def main():
    meta = load_json(os.path.join(WORK, "meta.json"))
    cj = load_json(os.path.join(WORK, "camera.json"))
    cam = Camera(cj["w"], cj["h"], *cj["init"])
    cam_p = (cj["w"], cj["h"], *cj["init"])
    paths = [m["path"] for m in meta]
    grid = load_json(os.path.join(WORK, "grid.json"))
    dmeta = load_json(os.path.join(WORK, "dsm_meta.json"))
    Rs = np.load(os.path.join(WORK, "ba_R.npy"))
    Cs = np.load(os.path.join(WORK, "ba_C.npy"))
    dsm = np.load(os.path.join(WORK, "dsm.npy"))
    n = len(meta)

    gx0, gy0, gsd, W, H = grid["x0"], grid["y0"], grid["gsd"], grid["w"], grid["h"]
    log(f"正射范围 {W} x {H} @ {gsd}m, 左上角 ({gx0:.2f}, {gy0:.2f})")

    # 覆盖统计 & 每 tile 视图选择
    zr = (dmeta["z_min"] - 2.0, dmeta["z_max"] + 2.0)
    bboxes = [image_ortho_bbox(i, Rs, Cs, cam, grid, zr) for i in range(n)]
    np.save(os.path.join(WORK, "ortho_bboxes.npy"), np.array(bboxes, dtype=np.int32))
    # 增益均衡
    gc_path = os.path.join(WORK, "gains.npy")
    if os.path.exists(gc_path):
        gains = np.load(gc_path)
        log("读取缓存增益")
    else:
        z = np.load(os.path.join(WORK, "feats.npz"), allow_pickle=True)
        feats = [{k: z[f"{i}_{k}"] for k in ("px", "nx", "ny", "scale", "ang", "resp", "des")}
                 for i in range(n)]
        zm = np.load(os.path.join(WORK, "matches.npz"), allow_pickle=True)
        keys = sorted(zm.files, key=lambda s: int(s.split("_")[0]))
        matches = [zm[k] for k in keys]
        pairs = load_json(os.path.join(WORK, "pairs.json"))
        t0 = time.time()
        gains, ratios = estimate_gains(meta, feats, matches, pairs, cam)
        # 限制增益范围，避免极端修正
        gains = np.clip(gains, 0.7, 1.45)
        gains = gains / np.exp(np.mean(np.log(gains), axis=0))[None, :]
        np.save(gc_path, gains)
        log(f"曝光增益估计完成 ({time.time()-t0:.0f}s): "
            f"R[{gains[:,0].min():.3f},{gains[:,0].max():.3f}] "
            f"G[{gains[:,1].min():.3f},{gains[:,1].max():.3f}] "
            f"B[{gains[:,2].min():.3f},{gains[:,2].max():.3f}]")

    # 正射渲染只用近似正射影像（倾斜影像色彩/几何差异大，易产生色块）
    ortho_ids = [i for i in range(n) if -Rs[i][2, 2] > 0.85]
    log(f"参与正射渲染的影像: {len(ortho_ids)} / {n} (排除倾斜影像)")

    # 构建 tile 任务
    tasks = []
    ntiles_x = int(math.ceil(W / TILE))
    ntiles_y = int(math.ceil(H / TILE))
    cov_count = np.zeros((ntiles_y, ntiles_x), dtype=np.int32)
    for ty in range(ntiles_y):
        for tx in range(ntiles_x):
            x0 = tx * TILE
            y0 = ty * TILE
            tw = min(TILE, W - x0)
            th = min(TILE, H - y0)
            bx = (x0, y0, x0 + tw, y0 + th)
            cands = []
            for i in ortho_ids:
                c0, r0, c1, r1 = bboxes[i]
                ox = max(0, min(c1, bx[2]) - max(c0, bx[0]))
                oy = max(0, min(r1, bx[3]) - max(r0, bx[1]))
                if ox > 0 and oy > 0:
                    frac = (ox * oy) / float(tw * th)
                    cosn = -Rs[i][2, 2]
                    cands.append((frac * (0.3 + 0.7 * cosn), i))
            if not cands:
                continue
            cands.sort(key=lambda t: -t[0])
            views = [i for _, i in cands[:MAX_VIEWS]]
            cov_count[ty, tx] = len(cands)
            tasks.append((tx, ty, gx0 + x0 * gsd, gy0 - y0 * gsd, gsd, tw, th,
                          views, cam_p, Rs, Cs, paths, gains,
                          os.path.join(WORK, "dsm.npy"), dmeta,
                          os.path.join(WORK, "dsm_conf.npy")))
    log(f"分块任务: {len(tasks)}, 平均每块候选视图 {cov_count[cov_count>0].mean():.1f}")

    quick = os.environ.get("QUICK", "0") == "1"
    if quick:
        # 低分辨率全图 + 若干高清 crop（用于快速验证拼接质量）
        qgsd = 0.20
        qW = int((W * gsd) / qgsd)
        qH = int((H * gsd) / qgsd)
        log(f"[QUICK] 渲染概览 {qW} x {qH} @ {qgsd}m")
        tasks_q = []
        for ty in range(0, qH, TILE):
            for tx in range(0, qW, TILE):
                tw = min(TILE, qW - tx)
                th = min(TILE, qH - ty)
                bx = (tx, ty, tx + tw, ty + th)
                cands = []
                for i in ortho_ids:
                    c0, r0, c1, r1 = bboxes[i]
                    # bbox 是正射像素坐标 -> 换算到概览坐标
                    c0q, c1q = c0 * gsd / qgsd, c1 * gsd / qgsd
                    r0q, r1q = r0 * gsd / qgsd, r1 * gsd / qgsd
                    ox = max(0, min(c1q, bx[2]) - max(c0q, bx[0]))
                    oy = max(0, min(r1q, bx[3]) - max(r0q, bx[1]))
                    if ox > 0 and oy > 0:
                        frac = (ox * oy) / float(tw * th)
                        cands.append((frac * (0.3 + 0.7 * (-Rs[i][2, 2])), i))
                if not cands:
                    continue
                cands.sort(key=lambda t: -t[0])
                views = [i for _, i in cands[:MAX_VIEWS]]
                tasks_q.append((tx, ty, gx0 + tx * qgsd, gy0 - ty * qgsd, qgsd, tw, th,
                                views, cam_p, Rs, Cs, paths, gains,
                                os.path.join(WORK, "dsm.npy"), dmeta,
                                os.path.join(WORK, "dsm_conf.npy")))
        qimg = np.zeros((qH, qW, 3), dtype=np.uint8)
        with ProcessPoolExecutor(max_workers=min(12, os.cpu_count())) as ex:
            for tx, ty, blk, vmask in ex.map(_render_tile, tasks_q, chunksize=1):
                th, tw = blk.shape[:2]
                qimg[ty:ty + th, tx:tx + tw] = blk
        imwrite_any(os.path.join(OUT, "quick_overview.png"), qimg,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
        log(f"[QUICK] 概览已保存 {qW} x {qH}")

        # 高清 crop：选在影像重叠边界附近（最易出现接缝的位置）
        cands = []
        for i in ortho_ids:
            for j in ortho_ids:
                if j <= i:
                    continue
                d = np.linalg.norm(Cs[i] - Cs[j])
                if 15 < d < 60:
                    cands.append((d, i, j))
        cands.sort()
        picks = []
        for d, i, j in cands:
            mid = (Cs[i] + Cs[j]) / 2.0
            if len(picks) >= 4:
                break
            if all(np.linalg.norm(mid - p) > 60 for p in picks):
                picks.append(mid)
        for k, p in enumerate(picks):
            cg = 0.025
            cw = ch = 1400
            tasks_c = []
            for ty in range(0, ch, 512):
                for tx in range(0, cw, 512):
                    tw = min(512, cw - tx)
                    th = min(512, ch - ty)
                    tasks_c.append((tx, ty, p[0] - cw / 2 * cg + tx * cg,
                                    p[1] + ch / 2 * cg - ty * cg, cg, tw, th,
                                    None, cam_p, Rs, Cs, paths, gains,
                                    os.path.join(WORK, "dsm.npy"), dmeta,
                                    os.path.join(WORK, "dsm_conf.npy")))
            cimg = np.zeros((ch, cw, 3), dtype=np.uint8)
            for t in tasks_c:
                bb = (t[1], t[0], t[1] + t[6], t[0] + t[5])
                cds = []
                for i2 in ortho_ids:
                    c0, r0, c1, r1 = bboxes[i2]
                    c0q, c1q = c0 * gsd / cg, c1 * gsd / cg
                    r0q, r1q = r0 * gsd / cg, r1 * gsd / cg
                    ox0 = gx0 + t[2] * 1
                    # 换算：bbox(正射像素) -> 地理 -> crop 局部像素
                    geo_x0 = gx0 + c0 * gsd
                    geo_x1 = gx0 + c1 * gsd
                    geo_y0 = gy0 - r1 * gsd
                    geo_y1 = gy0 - r0 * gsd
                    lx0 = (geo_x0 - (p[0] - cw / 2 * cg)) / cg
                    lx1 = (geo_x1 - (p[0] - cw / 2 * cg)) / cg
                    ly0 = ((p[1] + ch / 2 * cg) - geo_y1) / cg
                    ly1 = ((p[1] + ch / 2 * cg) - geo_y0) / cg
                    ox = max(0, min(lx1, bb[3]) - max(lx0, bb[1]))
                    oy = max(0, min(ly1, bb[2]) - max(ly0, bb[0]))
                    if ox > 0 and oy > 0:
                        cds.append(((ox * oy) * (0.3 + 0.7 * (-Rs[i2][2, 2])), i2))
                if not cds:
                    continue
                cds.sort(key=lambda z: -z[0])
                t2 = list(t)
                t2[7] = [i2 for _, i2 in cds[:MAX_VIEWS]]
                tx2, ty2, blk, vmask = _render_tile(tuple(t2))
                cimg[ty2:ty2 + blk.shape[0], tx2:tx2 + blk.shape[1]] = blk
            imwrite_any(os.path.join(OUT, f"quick_crop{k}.png"), cimg,
                        [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
            log(f"[QUICK] crop{k} @ E{p[0]:.0f} N{p[1]:.0f} 已保存 {cw}x{ch} @2.5cm")
        return

    out_img = np.zeros((H, W, 3), dtype=np.uint8)
    valid_mask = np.zeros((H, W), dtype=bool)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count())) as ex:
        for tx, ty, blk, vmask in ex.map(_render_tile, tasks, chunksize=1):
            x0 = tx * TILE
            y0 = ty * TILE
            th, tw = blk.shape[:2]
            out_img[y0:y0 + th, x0:x0 + tw] = blk
            valid_mask[y0:y0 + th, x0:x0 + tw] = vmask
            done += 1
            if done % 40 == 0:
                log(f"  渲染 {done}/{len(tasks)} 块 ({time.time()-t0:.0f}s)")
    log(f"渲染完成, 有效率 {valid_mask.mean()*100:.2f}%, 用时 {time.time()-t0:.0f}s")

    np.save(os.path.join(OUT, "ortho_rgb.npy"), out_img)
    np.save(os.path.join(OUT, "ortho_valid.npy"), valid_mask)
    save_json(os.path.join(OUT, "ortho_meta.json"),
              {"x0": gx0, "y0": gy0, "gsd": gsd, "w": W, "h": H})
    log(f"保存中间结果 -> {OUT}")


if __name__ == "__main__":
    main()
