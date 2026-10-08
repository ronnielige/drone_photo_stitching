"""Step5: 自主多视立体(plane-sweep)生成 DSM，并确定正射网格。"""
import os
import math
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor
from common import (WORK, OUT, log, load_json, save_json, Camera, imread_any)

# ---- 参数 ----
ORTHO_GSD = 0.05        # 正射输出分辨率 (m)  —— 接近原始影像 GSD
DSM_GSD = 0.25          # DSM 网格分辨率 (m)
DS = 3                  # 深度图计算时的降采样因子
PATCH = 7               # NCC 窗口 (在降采样图上)
MIN_NCC = 0.25
N_NEIGH = 3             # 每个参考视图使用的邻视数量


def coverage_bbox(fps, res=4.0, min_cov=3):
    """计算被至少 min_cov 张影像覆盖的区域 bbox"""
    allp = np.concatenate(fps, axis=0)
    x0, y0 = allp[:, 0].min(), allp[:, 1].min()
    x1, y1 = allp[:, 0].max(), allp[:, 1].max()
    xs = np.arange(x0, x1, res)
    ys = np.arange(y0, y1, res)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel()], axis=1)
    cnt = np.zeros(len(P), dtype=np.int32)
    for f in fps:
        poly = f if _signed_area(f) > 0 else f[::-1]
        inside = np.ones(len(P), dtype=bool)
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            cr = (b[0] - a[0]) * (P[:, 1] - a[1]) - (b[1] - a[1]) * (P[:, 0] - a[0])
            inside &= (cr >= 0)
        cnt += inside
    C = cnt.reshape(Y.shape)
    m = C >= min_cov
    if not m.any():
        m = C >= 1
    ii, jj = np.nonzero(m)
    bx0 = xs[jj.min()]
    bx1 = xs[jj.max()] + res
    by0 = ys[ii.min()]
    by1 = ys[ii.max()] + res
    return bx0, bx1, by0, by1, C, xs, ys


def _signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def project_points(Xw, R, C, cam):
    """世界点 (N,3) -> 像素 (N,2)（含畸变）"""
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
    return np.stack([u, v], axis=1), (z > 1.0)


def box_sum(img, k):
    return cv2.boxFilter(img, -1, (k, k), normalize=False, borderType=cv2.BORDER_REFLECT)


def _sweep_one(args):
    """对单个参考视图做 plane-sweep，返回 (h,w) 的最佳高度层索引与 NCC 峰值"""
    (ri, cam_p, neigh, Rs, Cs, zlayers, ds, patch, paths) = args
    w, h, fx, fy, cx, cy, k1, k2, k3, p1, p2 = cam_p
    cam = Camera(w, h, fx, fy, cx, cy, k1, k2, k3, p1, p2)
    R, C = Rs[ri], Cs[ri]
    gw, gh = w // ds, h // ds

    # 参考视图降采样灰度图 + 无畸变归一化射线
    g0 = imread_any(paths[ri], gray=True)
    if g0 is None:
        return ri, None, None
    g0 = cv2.resize(g0, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32)
    jj, ii = np.meshgrid(np.arange(gw), np.arange(gh))
    upx = jj.astype(np.float64) * ds + ds / 2.0
    upy = ii.astype(np.float64) * ds + ds / 2.0
    # 去畸变（迭代 3 次）
    xn = (upx - cx) / fx
    yn = (upy - cy) / fy
    for _ in range(4):
        r2 = xn * xn + yn * yn
        rad = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        xn = (upx - cx) / fx - (xd - xn)
        yn = (upy - cy) / fy - (yd - yn)
    dirs = np.stack([xn, yn, np.ones_like(xn)], axis=2) @ R.T   # (gh,gw,3)
    dn = np.linalg.norm(dirs, axis=2, keepdims=True)
    dirs = dirs / dn
    dz = dirs[:, :, 2]

    # 预加载邻视灰度图（避免逐层重复解码）
    nb_imgs = {}
    for v in neigh:
        g = imread_any(paths[v], gray=True)
        if g is not None:
            nb_imgs[v] = g

    # 参考图局部统计
    n = float(patch * patch)
    s1 = box_sum(g0, patch)
    s2 = box_sum(g0 * g0, patch)
    mean_r = s1 / n
    var_r = np.maximum(s2 / n - mean_r * mean_r, 1e-6)

    best_cost = np.full((gh, gw), -1e9, dtype=np.float32)
    best_idx = np.zeros((gh, gw), dtype=np.int16)
    cost_stack = np.full((len(zlayers), gh, gw), -1e9, dtype=np.float32)

    for li, z in enumerate(zlayers):
        dzs = np.where(np.abs(dz) < 1e-3, 1e-3, dz)
        t = (z - C[2]) / dzs
        G = C[None, None, :] + t[:, :, None] * dirs        # (gh,gw,3)
        Gf = G.reshape(-1, 3)
        acc = np.zeros((gh, gw), dtype=np.float32)
        nvis = np.zeros((gh, gw), dtype=np.float32)
        for v in neigh:
            px, front = project_points(Gf, Rs[v], Cs[v], cam)
            px = px.reshape(gh, gw, 2)
            front = front.reshape(gh, gw)
            mapx = px[:, :, 0].astype(np.float32)
            mapy = px[:, :, 1].astype(np.float32)
            gv = nb_imgs.get(v)
            if gv is None:
                continue
            warp = cv2.remap(gv, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
            warp = warp.astype(np.float32)
            ok = front & (px[:, :, 0] > patch) & (px[:, :, 1] > patch) & \
                 (px[:, :, 0] < w - patch) & (px[:, :, 1] < h - patch) & (warp > 0)
            m1 = box_sum(warp, patch)
            m2 = box_sum(warp * warp, patch)
            mrv = box_sum(warp * g0, patch)
            mean_v = m1 / n
            var_v = np.maximum(m2 / n - mean_v * mean_v, 1e-6)
            cov = mrv / n - mean_r * mean_v
            ncc = cov / np.sqrt(var_r * var_v)
            ncc = np.where(ok, ncc, -1.0)
            np.clip(ncc, -1.0, 1.0, out=ncc)
            acc += np.maximum(ncc, 0.0)
            nvis += ok.astype(np.float32)
        with np.errstate(invalid="ignore"):
            cost = np.where(nvis >= 2, acc / np.maximum(nvis, 1), -1e9)
        cost_stack[li] = cost
        better = cost > best_cost
        best_cost = np.where(better, cost, best_cost)
        best_idx = np.where(better, li, best_idx).astype(np.int16)

    # 抛物线细化
    li = best_idx
    zbest = np.array(zlayers, dtype=np.float64)[li]
    step = zlayers[1] - zlayers[0] if len(zlayers) > 1 else 1.0
    okm = (li > 0) & (li < len(zlayers) - 1) & (best_cost > MIN_NCC)
    ii, jj = np.nonzero(okm)
    if len(ii) > 0:
        lic = li[ii, jj]
        a_ = cost_stack[lic - 1, ii, jj]
        b_ = cost_stack[lic, ii, jj]
        c_ = cost_stack[lic + 1, ii, jj]
        denom = (a_ - 2 * b_ + c_)
        denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        delta = 0.5 * (a_ - c_) / denom
        delta = np.clip(delta, -1.0, 1.0)
        zbest[ii, jj] = zbest[ii, jj] + delta * step
    conf = best_cost
    # 地面点位置（用最佳层重算）
    dzs = np.where(np.abs(dz) < 1e-3, 1e-3, dz)
    t = (zbest - C[2]) / dzs
    G = C[None, None, :] + t[:, :, None] * dirs
    return ri, G.astype(np.float32), conf.astype(np.float32)


def main():
    meta = load_json(os.path.join(WORK, "meta.json"))
    cj = load_json(os.path.join(WORK, "camera.json"))
    cam = Camera(cj["w"], cj["h"], *cj["init"])
    cam_p = (cj["w"], cj["h"], *cj["init"])
    paths = [m["path"] for m in meta]
    Rs = np.load(os.path.join(WORK, "ba_R.npy"))
    Cs = np.load(os.path.join(WORK, "ba_C.npy"))
    # 用 BA 后的位姿重算 footprint
    from step2_init import footprint
    ground0 = load_json(os.path.join(WORK, "ground0.json"))["z_ground"]
    fps = np.array([footprint(Rs[i], Cs[i], cam, ground0) for i in range(len(meta))])
    np.save(os.path.join(WORK, "fps_ba.npy"), fps)

    # 覆盖范围只统计近似正射的影像（倾斜影像会向测区外大幅拉伸）
    nad = np.array([Rs[i][2, 2] for i in range(len(meta))])
    sel = np.flatnonzero(nad < -0.80)
    bx0, bx1, by0, by1, C, xs, ys = coverage_bbox(fps[sel], res=4.0, min_cov=7)
    log(f"参与覆盖统计的影像: {len(sel)} / {len(meta)}")
    log(f"有效覆盖 bbox: E[{bx0:.1f},{bx1:.1f}] N[{by0:.1f},{by1:.1f}] "
        f"尺寸 {bx1-bx0:.1f} x {by1-by0:.1f} m")

    # 正射网格
    Wo = int(math.ceil((bx1 - bx0) / ORTHO_GSD))
    Ho = int(math.ceil((by1 - by0) / ORTHO_GSD))
    log(f"正射网格: {Wo} x {Ho} @ {ORTHO_GSD*100:.1f}cm  ({Wo*Ho/1e6:.1f} M像素)")
    save_json(os.path.join(WORK, "grid.json"),
              {"x0": bx0, "y0": by1, "gsd": ORTHO_GSD, "w": Wo, "h": Ho,
               "dsm_gsd": DSM_GSD})

    # DSM 网格
    Wd = int(math.ceil((bx1 - bx0) / DSM_GSD))
    Hd = int(math.ceil((by1 - by0) / DSM_GSD))
    dsm_sum = np.zeros((Hd, Wd), dtype=np.float64)
    dsm_cnt = np.zeros((Hd, Wd), dtype=np.float32)

    # 粗地面高度（稀疏点云中位数）
    Xp = np.load(os.path.join(WORK, "ba_X.npy"))
    gz = float(np.nanmedian(Xp[:, 2]))
    log(f"稀疏点云中位高程 = {gz:.2f} m")
    base = gz
    zlayers = np.round(np.arange(base - 4.0, base + 20.01, 1.0), 3)
    log(f"高度搜索层: {zlayers[0]:.1f} .. {zlayers[-1]:.1f} m, 共 {len(zlayers)} 层")

    # 邻视选择
    pairs = load_json(os.path.join(WORK, "pairs.json"))
    nadir = np.array([-Rs[i][2, 2] for i in range(len(meta))])   # 光轴朝下的程度
    adj = {}
    for i, j, ov in pairs:
        adj.setdefault(i, []).append((j, ov))
        adj.setdefault(j, []).append((i, ov))
    refs = [i for i in range(len(meta)) if nadir[i] > 0.85]
    log(f"参考视图数: {len(refs)} / {len(meta)} (nadir>0.85)")
    tasks = []
    for i in refs:
        cand = sorted(adj.get(i, []), key=lambda t: -t[1])[:10]
        # 视差优先：在 IoU>0.25 的候选中按基线从大到小取 N_NEIGH 个
        cand2 = [(j, ov, np.linalg.norm(Cs[j] - Cs[i])) for j, ov in cand if ov > 0.25]
        cand2.sort(key=lambda t: -t[2])
        nb = [j for j, _, _ in cand2[:N_NEIGH]]
        while len(nb) < N_NEIGH and len(cand) > len(nb):
            for j, ov in cand:
                if j not in nb:
                    nb.append(j)
                if len(nb) >= N_NEIGH:
                    break
        tasks.append((i, cam_p, nb, Rs, Cs, zlayers, DS, PATCH, paths))

    log(f"开始 plane-sweep: {len(tasks)} 个参考视图 x {len(zlayers)} 层 x {N_NEIGH} 邻视")
    t0 = os.times().elapsed
    n_ok = 0
    with ProcessPoolExecutor(max_workers=min(14, os.cpu_count())) as ex:
        for ri, G, conf in ex.map(_sweep_one, tasks, chunksize=1):
            if G is None:
                continue
            n_ok += 1
            pts = G.reshape(-1, 3)
            cf = conf.reshape(-1)
            m = (cf > MIN_NCC) & (pts[:, 0] >= bx0) & (pts[:, 0] < bx1) & \
                (pts[:, 1] >= by0) & (pts[:, 1] < by1)
            p = pts[m]
            c_ = cf[m]
            if len(p) == 0:
                continue
            ix = np.clip(((p[:, 0] - bx0) / DSM_GSD).astype(np.int32), 0, Wd - 1)
            iy = np.clip(((by1 - p[:, 1]) / DSM_GSD).astype(np.int32), 0, Hd - 1)
            np.add.at(dsm_sum, (iy, ix), p[:, 2] * c_)
            np.add.at(dsm_cnt, (iy, ix), c_)
            if n_ok % 20 == 0:
                log(f"  已完成 {n_ok}/{len(tasks)} 个参考视图")
    log(f"plane-sweep 完成: {n_ok} 视图, 用时 {os.times().elapsed - t0:.0f}s")

    dsm = np.where(dsm_cnt > 0, dsm_sum / np.maximum(dsm_cnt, 1e-6), np.nan)
    fill = np.nanmedian(dsm)
    log(f"DSM 有效率: {np.mean(~np.isnan(dsm))*100:.1f}%, 中位数高程 {fill:.2f} m")

    # 空洞填充 + 中值滤波
    from scipy.ndimage import median_filter, grey_dilation, binary_dilation
    mask = ~np.isnan(dsm)
    # 距离加权最近邻填充（简化：迭代膨胀）
    d_filled = dsm.copy()
    for _ in range(40):
        if not np.isnan(d_filled).any():
            break
        nb = np.isnan(d_filled)
        sh = np.roll(d_filled, 1, 0)
        sh[0, :] = np.nan
        d_filled[nb] = sh[nb]
        for ax in (0, 1):
            for s in (1, -1):
                sh = np.roll(d_filled, s, ax)
                idx = [slice(None)] * 2
                idx[ax] = 0 if s > 0 else -1
                sh[tuple(idx)] = np.nan
                m2 = np.isnan(d_filled)
                d_filled[m2] = sh[m2]
    dsm_f = median_filter(d_filled, size=5)
    # 保边：置信度高的地方保留原值
    dsm_f = np.where(mask, np.minimum(np.maximum(dsm_f, np.nanmin(dsm_f)), np.nanmax(dsm_f)), dsm_f)
    np.save(os.path.join(WORK, "dsm.npy"), dsm_f.astype(np.float32))
    np.save(os.path.join(WORK, "dsm_conf.npy"), dsm_cnt.astype(np.float32))
    save_json(os.path.join(WORK, "dsm_meta.json"),
              {"x0": bx0, "y0": by1, "gsd": DSM_GSD, "w": Wd, "h": Hd,
               "z_min": float(np.nanmin(dsm_f)), "z_max": float(np.nanmax(dsm_f)),
               "z_median": float(fill)})
    log(f"DSM 保存: {Wd} x {Hd} @ {DSM_GSD}m, 高程范围 "
        f"[{np.nanmin(dsm_f):.1f}, {np.nanmax(dsm_f):.1f}]")


if __name__ == "__main__":
    main()
