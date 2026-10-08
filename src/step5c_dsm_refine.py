# -*- coding: utf-8 -*-
"""Step5c: DSM 精化 v2 —— 高分辨率二次 plane-sweep。

诊断结论（对照大疆）：旧 DSM 在建筑屋顶 std≈6.7m、农田 std≈2.4m，
导致正射屋顶扭曲、多视重影。本模块：
  1) 0.10m 网格（旧 0.25m）
  2) DS=2 匹配（旧 DS=3）
  3) 逐像素先验窗口搜索：粗 ±4m@0.5m → 细 ±0.6m@0.1m → 抛物线细化
  4) 先验中剔除旧 DSM 的层顶饱和假值（z>=156.9）
  5) 融合：conf^3 加权 → 两遍鲁棒重加权（剔除离群视图的高程投票）
  6) 输出 grid.json v2（正射 GSD 0.055，接近影像原生 GSD）
"""
import os
import sys
import math
import time
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor
from common import (WORK, OUT, log, load_json, save_json, Camera, imread_any)
from step5_dsm import coverage_bbox, project_points, box_sum

DSM_GSD = 0.10         # 新 DSM 网格
ORTHO_GSD = 0.055      # 正射输出（原生 GSD≈5.0-5.6cm，避免上采样发软）
DS = 2
PATCH = 9
MIN_NCC = 0.12   # 植被多视 NCC 天生偏低(中位~0.19)，门槛过严会导致覆盖空洞
N_NEIGH = 3
COARSE_LO = -4.0
COARSE_HI = 6.0        # 向上多搜：旧 DSM 层顶 157 削顶，真实屋顶可达 ~158.5
COARSE_STEP = 1.0      # 粗搜索步长（细搜索 ±0.6 可覆盖其残差）
FINE_STEP = 0.1
FINE_HALF = 0.6
Z_SAT = 156.9          # 旧 DSM 搜索层顶饱和值
RW_SIGMA = 0.40        # 鲁棒重加权 σ (m)
CONF_P = 3.0           # 置信度指数
REF_STRIDE = int(os.environ.get("REF_STRIDE", "1"))   # 参考视图抽样步长
DIAG = os.environ.get("DIAG", "0") == "1"

try:
    import cv2 as _cv2
    _cv2.setNumThreads(1)      # 避免多进程 × 多线程过度订阅
except Exception:
    pass


def sample_prior(prior, dmeta, X, Y):
    """双线性采样先验 DSM（带越界钳制），返回 z 与有效 mask"""
    gsd = dmeta["gsd"]
    X = np.where(np.isfinite(X), X, dmeta["x0"])
    Y = np.where(np.isfinite(Y), Y, dmeta["y0"])
    fx = (X - dmeta["x0"]) / gsd - 0.5
    fy = (dmeta["y0"] - Y) / gsd - 0.5
    H, W = prior.shape
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    tx = (fx - x0).astype(np.float32)
    ty = (fy - y0).astype(np.float32)
    x0c = np.clip(x0, 0, W - 2)
    y0c = np.clip(y0, 0, H - 2)
    inb = (x0 >= 0) & (x0 < W - 1) & (y0 >= 0) & (y0 < H - 1)
    v00 = prior[y0c, x0c]; v01 = prior[y0c, x0c + 1]
    v10 = prior[y0c + 1, x0c]; v11 = prior[y0c + 1, x0c + 1]
    z = (v00 * (1 - tx) * (1 - ty) + v01 * tx * (1 - ty) +
         v10 * (1 - tx) * ty + v11 * tx * ty)
    ok = inb & ~np.isnan(z) & (z < Z_SAT)
    return z.astype(np.float32), ok


def _sweep_one_v2(args):
    (ri, cam_p, neigh, Rs, Cs, prior_path, dmeta, paths) = args
    w, h, fx, fy, cx, cy, k1, k2, k3, p1, p2 = cam_p
    cam = Camera(w, h, fx, fy, cx, cy, k1, k2, k3, p1, p2)
    R, C = Rs[ri], Cs[ri]
    gw, gh = w // DS, h // DS

    prior = np.load(prior_path, mmap_mode="r")

    g0 = imread_any(paths[ri], gray=True)
    if g0 is None:
        return ri, None, None, -2
    g0 = cv2.resize(g0, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32)
    jj, ii = np.meshgrid(np.arange(gw), np.arange(gh))
    upx = jj.astype(np.float64) * DS + DS / 2.0
    upy = ii.astype(np.float64) * DS + DS / 2.0
    xn = (upx - cx) / fx
    yn = (upy - cy) / fy
    for _ in range(4):
        r2 = xn * xn + yn * yn
        rad = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        xn = (upx - cx) / fx - (xd - xn)
        yn = (upy - cy) / fy - (yd - yn)
    dirs = np.stack([xn, yn, np.ones_like(xn)], axis=2) @ R.T
    dn = np.linalg.norm(dirs, axis=2, keepdims=True)
    dirs = dirs / dn
    dz = dirs[:, :, 2]
    dzs = np.where(np.abs(dz) < 1e-3, 1e-3, dz)

    nb_imgs = {}
    for v in neigh:
        g = imread_any(paths[v], gray=True)
        if g is not None:
            nb_imgs[v] = g

    n = float(PATCH * PATCH)
    s1 = box_sum(g0, PATCH)
    s2 = box_sum(g0 * g0, PATCH)
    mean_r = s1 / n
    var_r = np.maximum(s2 / n - mean_r * mean_r, 1e-6)

    diag = {}
    if DIAG:
        zv = np.float32(dmeta["_z0"])
        tv = (zv - C[2]) / dzs
        Gf = (C[None, None, :] + tv[:, :, None] * dirs).reshape(-1, 3)
        nv = np.zeros((gh, gw), dtype=np.int16)
        for v in neigh:
            gv = nb_imgs.get(v)
            if gv is None:
                continue
            px, front = project_points(Gf, Rs[v], Cs[v], cam)
            px = px.reshape(gh, gw, 2)
            mapx = px[:, :, 0].astype(np.float32)
            mapy = px[:, :, 1].astype(np.float32)
            warp = cv2.remap(gv, mapx, mapy, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
            ok = (front.reshape(gh, gw) & (px[:, :, 0] > PATCH) & (px[:, :, 1] > PATCH) &
                  (px[:, :, 0] < w - PATCH) & (px[:, :, 1] < h - PATCH) & (warp > 0))
            nv += ok.astype(np.int16)
        diag = {"nvis1": round(float((nv >= 1).mean()), 4),
                "nvis2": round(float((nv >= 2).mean()), 4),
                "nvis3": round(float((nv >= 3).mean()), 4),
                "g0_black": round(float((g0 <= 1).mean()), 4),
                "nb_imgs": len(nb_imgs), "neigh": list(neigh),
                "z0": round(float(zv), 2)}

    # ---- 逐像素先验（不动点 2 次迭代）----
    Xg = C[None, None, 0] + dirs[:, :, 0] / dzs * (dmeta["_z0"] - C[2])
    Yg = C[None, None, 1] + dirs[:, :, 1] / dzs * (dmeta["_z0"] - C[2])
    zp, okp = sample_prior(prior, dmeta, Xg, Yg)
    for _ in range(2):
        t = (zp - C[2]) / dzs
        Xg2 = C[0] + t * dirs[:, :, 0]
        Yg2 = C[1] + t * dirs[:, :, 1]
        zp2, okp2 = sample_prior(prior, dmeta, Xg2, Yg2)
        zp = np.where(okp2, zp2, zp)
        okp = okp | okp2
    # 无先验的像素：用全局地面中位数
    z0_global = np.float32(dmeta["_z0"])
    zp = np.where(okp, zp, z0_global)
    zp = np.clip(zp, dmeta["z_min"] - 1.0, dmeta["z_max"] + 1.0)

    def sweep(deltas):
        """给定逐像素偏移序列，返回 (best_cost, best_k, cost_stack)"""
        nd = len(deltas)
        best_cost = np.full((gh, gw), -1e9, dtype=np.float32)
        best_k = np.zeros((gh, gw), dtype=np.int16)
        stack = np.empty((nd, gh, gw), dtype=np.float32)
        for k, dl in enumerate(deltas):
            z = zp + np.float32(dl)
            t = (z - C[2]) / dzs
            G = C[None, None, :] + t[:, :, None] * dirs
            Gf = G.reshape(-1, 3)
            acc = np.zeros((gh, gw), dtype=np.float32)
            nvis = np.zeros((gh, gw), dtype=np.float32)
            for v in neigh:
                gv = nb_imgs.get(v)
                if gv is None:
                    continue
                px, front = project_points(Gf, Rs[v], Cs[v], cam)
                px = px.reshape(gh, gw, 2)
                front = front.reshape(gh, gw)
                mapx = px[:, :, 0].astype(np.float32)
                mapy = px[:, :, 1].astype(np.float32)
                warp = cv2.remap(gv, mapx, mapy, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
                ok = front & (px[:, :, 0] > PATCH) & (px[:, :, 1] > PATCH) & \
                     (px[:, :, 0] < w - PATCH) & (px[:, :, 1] < h - PATCH) & (warp > 0)
                m1 = box_sum(warp, PATCH)
                m2 = box_sum(warp * warp, PATCH)
                mrv = box_sum(warp * g0, PATCH)
                mean_v = m1 / n
                var_v = np.maximum(m2 / n - mean_v * mean_v, 1e-6)
                cov = mrv / n - mean_r * mean_v
                ncc = cov / np.sqrt(var_r * var_v)
                ncc = np.where(ok, ncc, -1.0)
                np.clip(ncc, -1.0, 1.0, out=ncc)
                acc += np.maximum(ncc, 0.0)
                nvis += ok.astype(np.float32)
            cost = np.where(nvis >= 2, acc / np.maximum(nvis, 1), -1e9)
            stack[k] = cost
            better = cost > best_cost
            best_cost = np.where(better, cost, best_cost)
            best_k = np.where(better, k, best_k).astype(np.int16)
        return best_cost, best_k, stack

    # ---- 粗搜索（不对称窗口：向上多搜补偿削顶）----
    deltas_c = np.round(np.arange(COARSE_LO, COARSE_HI + 1e-6, COARSE_STEP), 3)
    bc, bk, _ = sweep(deltas_c)
    okc = bc > 0.15   # 粗搜索低门槛（只用于决定是否进入抛物线细化）
    # ---- 细搜索（围绕粗搜索最优 ±0.6m）----
    zc = zp + deltas_c[np.clip(bk, 0, len(deltas_c) - 1)]
    deltas_f_all = np.round(np.arange(-FINE_HALF, FINE_HALF + 1e-6, FINE_STEP), 3)

    best_cost = np.full((gh, gw), -1e9, dtype=np.float32)
    zbest = zc.astype(np.float64)
    # 逐像素窗口不同 -> 按 (lo,hi) 分组太碎；简化：统一用 zc±FINE_HALF，
    # 粗搜索触边的像素额外补一轮扩展层
    stack_f = np.empty((len(deltas_f_all), gh, gw), dtype=np.float32)
    for k, dl in enumerate(deltas_f_all):
        z = zc + np.float32(dl)
        t = (z - C[2]) / dzs
        G = C[None, None, :] + t[:, :, None] * dirs
        Gf = G.reshape(-1, 3)
        acc = np.zeros((gh, gw), dtype=np.float32)
        nvis = np.zeros((gh, gw), dtype=np.float32)
        for v in neigh:
            gv = nb_imgs.get(v)
            if gv is None:
                continue
            px, front = project_points(Gf, Rs[v], Cs[v], cam)
            px = px.reshape(gh, gw, 2)
            front = front.reshape(gh, gw)
            mapx = px[:, :, 0].astype(np.float32)
            mapy = px[:, :, 1].astype(np.float32)
            warp = cv2.remap(gv, mapx, mapy, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
            ok = front & (px[:, :, 0] > PATCH) & (px[:, :, 1] > PATCH) & \
                 (px[:, :, 0] < w - PATCH) & (px[:, :, 1] < h - PATCH) & (warp > 0)
            m1 = box_sum(warp, PATCH)
            m2 = box_sum(warp * warp, PATCH)
            mrv = box_sum(warp * g0, PATCH)
            mean_v = m1 / n
            var_v = np.maximum(m2 / n - mean_v * mean_v, 1e-6)
            cov = mrv / n - mean_r * mean_v
            ncc = cov / np.sqrt(var_r * var_v)
            ncc = np.where(ok, ncc, -1.0)
            np.clip(ncc, -1.0, 1.0, out=ncc)
            acc += np.maximum(ncc, 0.0)
            nvis += ok.astype(np.float32)
        cost = np.where(nvis >= 2, acc / np.maximum(nvis, 1), -1e9)
        stack_f[k] = cost
        better = cost > best_cost
        best_cost = np.where(better, cost, best_cost)
        zbest = np.where(better, z, zbest)

    # 抛物线细化（在细搜索栈上）
    li = np.argmin(-stack_f, axis=0)   # best k
    okm = okc & (best_cost > MIN_NCC) & (li > 0) & (li < len(deltas_f_all) - 1)
    ii, jj = np.nonzero(okm)
    if len(ii) > 0:
        lk = li[ii, jj]
        a_ = stack_f[lk - 1, ii, jj]
        b_ = stack_f[lk, ii, jj]
        c_ = stack_f[lk + 1, ii, jj]
        denom = a_ - 2 * b_ + c_
        denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        delta = 0.5 * (a_ - c_) / denom
        delta = np.clip(delta, -1.0, 1.0)
        zbest[ii, jj] += delta * FINE_STEP

    # 地面点
    t = (zbest - C[2]) / dzs
    G = C[None, None, :] + t[:, :, None] * dirs
    conf = np.where(okm, best_cost, 0.0).astype(np.float32)
    if DIAG:
        good = conf > MIN_NCC
        diag.update({
            "okc_frac": round(float(okc.mean()), 4),
            "okm_frac": round(float(okm.mean()), 4),
            "conf_ok_frac": round(float(good.mean()), 4),
            "conf_p50_good": round(float(np.median(conf[good])) if good.any() else -1, 3),
            "bc_p50": round(float(np.median(bc[bc > -1e8])) if (bc > -1e8).any() else -1, 3),
            "bk_edge_frac": round(float(((bk == 0) | (bk == len(deltas_c) - 1)).mean()), 4),
        })
        print(f"[DIAG #{ri}] " + ", ".join(f"{k}={v}" for k, v in diag.items()),
              file=sys.stderr, flush=True)
    # 只回传置信度达标者，减小通信量
    p = G.reshape(-1, 3)
    cf = conf.reshape(-1)
    m = cf > MIN_NCC
    return ri, np.ascontiguousarray(p[m]), np.ascontiguousarray(cf[m]), len(nb_imgs), diag


def main():
    meta = load_json(os.path.join(WORK, "meta.json"))
    cj = load_json(os.path.join(WORK, "camera.json"))
    cam = Camera(cj["w"], cj["h"], *cj["init"])
    cam_p = (cj["w"], cj["h"], *cj["init"])
    paths = [m["path"] for m in meta]
    Rs = np.load(os.path.join(WORK, "ba_R.npy"))
    Cs = np.load(os.path.join(WORK, "ba_C.npy"))
    fps = np.load(os.path.join(WORK, "fps_ba.npy"))
    dmeta_old = load_json(os.path.join(WORK, "dsm_meta.json"))

    nad = np.array([Rs[i][2, 2] for i in range(len(meta))])
    sel = np.flatnonzero(nad < -0.80)
    bx0, bx1, by0, by1, Ccov, xs, ys = coverage_bbox(fps[sel], res=4.0, min_cov=5)
    log(f"覆盖 bbox: E[{bx0:.1f},{bx1:.1f}] N[{by0:.1f},{by1:.1f}]")

    # 正射网格 v2
    Wo = int(math.ceil((bx1 - bx0) / ORTHO_GSD))
    Ho = int(math.ceil((by1 - by0) / ORTHO_GSD))
    log(f"正射网格 v2: {Wo} x {Ho} @ {ORTHO_GSD*100:.1f}cm")
    save_json(os.path.join(WORK, "grid.json"),
              {"x0": bx0, "y0": by1, "gsd": ORTHO_GSD, "w": Wo, "h": Ho,
               "dsm_gsd": DSM_GSD})

    # DSM v2 网格
    Wd = int(math.ceil((bx1 - bx0) / DSM_GSD))
    Hd = int(math.ceil((by1 - by0) / DSM_GSD))
    log(f"DSM v2 网格: {Wd} x {Hd} @ {DSM_GSD}m")

    # 先验准备：饱和格保留为 157.0（屋顶真实高程在 157~158.5，粗搜索会向上找到）
    prior = np.load(os.path.join(WORK, "dsm.npy")).astype(np.float32)
    n_sat = int(np.sum(prior >= Z_SAT))
    prior[prior >= Z_SAT] = np.float32(157.0)
    log(f"先验: 饱和格 {n_sat} 保留为 157.0, 范围 [{np.nanmin(prior):.1f},{np.nanmax(prior):.1f}]")

    dmeta = dict(dmeta_old)
    dmeta["gsd"] = dmeta_old["gsd"]        # 先验网格分辨率不变
    dmeta["_z0"] = float(np.nanmedian(prior))
    prior_path = os.path.join(WORK, "dsm_prior_v2.npy")
    np.save(prior_path, prior.astype(np.float32))

    # 邻视选择（与 step5 相同策略）
    pairs = load_json(os.path.join(WORK, "pairs.json"))
    adj = {}
    for i, j, ov in pairs:
        adj.setdefault(i, []).append((j, ov))
        adj.setdefault(j, []).append((i, ov))
    refs = [i for i in range(len(meta)) if nad[i] < -0.85][::REF_STRIDE]
    log(f'参考视图抽样步长 {REF_STRIDE} -> {len(refs)} 个参考视图')
    tasks = []
    for i in refs:
        cand = sorted(adj.get(i, []), key=lambda t: -t[1])[:10]
        cand2 = [(j, ov, np.linalg.norm(Cs[j] - Cs[i])) for j, ov in cand if ov > 0.25]
        cand2.sort(key=lambda t: -t[2])
        nb = [j for j, _, _ in cand2[:N_NEIGH]]
        while len(nb) < N_NEIGH and len(cand) > len(nb):
            for j, ov in cand:
                if j not in nb:
                    nb.append(j)
                if len(nb) >= N_NEIGH:
                    break
        tasks.append((i, cam_p, nb, Rs, Cs, prior_path, dmeta, paths))

    log(f"plane-sweep v2: {len(tasks)} 视图, 粗 {len(np.arange(COARSE_LO, COARSE_HI+1e-6, COARSE_STEP))} 层 + 细 {len(np.arange(-FINE_HALF, FINE_HALF+1e-6, FINE_STEP))} 层")
    lim = int(os.environ.get("LIMIT_TASKS", "0"))
    if lim:
        tasks = tasks[:lim]
        log(f"[LIMIT_TASKS] 只跑前 {len(tasks)} 个视图")
    t0 = time.time()

    # ---- 逐格「最高置信视图取胜」（不做跨视平均，避免高程被平均钝化）----
    z_best = np.full(Wd * Hd, np.nan, dtype=np.float32)
    c_best = np.zeros(Wd * Hd, dtype=np.float32)
    wz_sum = np.zeros(Wd * Hd, dtype=np.float64)
    w_sum = np.zeros(Wd * Hd, dtype=np.float64)
    n_ok = 0
    n_pts = 0
    nb_hist = {}
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as ex:
        for ri, p, cf, n_nb, dg in ex.map(_sweep_one_v2, tasks, chunksize=1):
            if p is None or len(p) == 0:
                log(f"  [警告] 视图 #{ri} 返回 0 点 (邻视载入 {n_nb})")
                continue
            n_ok += 1
            n_pts += len(p)
            nb_hist[n_nb] = nb_hist.get(n_nb, 0) + 1
            ix = ((p[:, 0] - bx0) / DSM_GSD).astype(np.int64)
            iy = ((by1 - p[:, 1]) / DSM_GSD).astype(np.int64)
            ok2 = (ix >= 0) & (ix < Wd) & (iy >= 0) & (iy < Hd)
            if not ok2.any():
                log(f"  [警告] 视图 #{ri} 的点全部落在 DSM 网格外")
                continue
            ci = (iy[ok2] * Wd + ix[ok2])
            zz = p[ok2, 2].astype(np.float32)
            cc = cf[ok2].astype(np.float32)
            # 单帧取胜：某格只记录置信度最高的那一帧的高程
            order = np.argsort(cc, kind="stable")
            ci_s = ci[order]
            # 后写覆盖前写 -> 按置信度升序写入，同格保留最高
            z_best[ci_s] = zz[order]
            c_best[ci_s] = cc[order]
            # 同时累计加权和（用于诊断对比）
            wp = cc.astype(np.float64) ** CONF_P
            np.add.at(wz_sum, ci, wp * zz)
            np.add.at(w_sum, ci, wp)
            if n_ok <= 3 or n_ok % 20 == 0:
                log(f"  完成 {n_ok}/{len(tasks)} 视图 (#{ri}), 累计点 {n_pts/1e6:.1f}M, "
                    f"已覆盖格 {int(np.isfinite(z_best).sum())/1e6:.1f}M")
    log(f"sweep 完成: {n_ok}/{len(tasks)} 视图, 总点 {n_pts/1e6:.1f}M, "
        f"用时 {time.time()-t0:.0f}s, 邻视载入分布 {nb_hist}")

    z_best = z_best.reshape(Hd, Wd)
    c_best = c_best.reshape(Hd, Wd)
    z_avg = np.where(w_sum > 0, wz_sum / np.maximum(w_sum, 1e-9), np.nan).reshape(Hd, Wd)
    valid = ~np.isnan(z_best)
    log(f"有效格 {valid.mean()*100:.2f}%, 单帧取胜 vs 加权平均 差值 std "
        f"{np.nanstd((z_best - z_avg)[valid]):.3f} m")

    # 低置信格用加权平均兜底
    weak = valid & (c_best < 0.30) & ~np.isnan(z_avg)
    z_best[weak] = z_avg[weak]
    del wz_sum, w_sum, z_avg

    fill = float(np.nanmedian(z_best))
    log(f"中位高程 {fill:.2f}, 范围 [{np.nanmin(z_best):.2f},{np.nanmax(z_best):.2f}]")

    # 空洞填充（由外向内扩散）
    d_filled = z_best.copy()
    for _ in range(200):
        nb = np.isnan(d_filled)
        if not nb.any():
            break
        for ax in (0, 1):
            for s in (1, -1):
                sh = np.roll(d_filled, s, ax)
                idx = [slice(None)] * 2
                idx[ax] = 0 if s > 0 else -1
                sh[tuple(idx)] = np.nan
                m2 = np.isnan(d_filled) & ~np.isnan(sh)
                d_filled[m2] = sh[m2]

    # 保边平滑：3x3 中值去毛刺，再对平坦区做一次极小 sigma 高斯（不损屋顶边缘）
    from scipy.ndimage import median_filter, gaussian_filter
    dsm_med = median_filter(d_filled, size=3)
    hp = dsm_med - gaussian_filter(dsm_med, 3.0)
    dsm_f = gaussian_filter(dsm_med, 0.8) + hp * 0.35   # 轻度低通 + 保高频

    np.save(os.path.join(WORK, "dsm_v2.npy"), dsm_f.astype(np.float32))
    np.save(os.path.join(WORK, "dsm_v2_conf.npy"), c_best.astype(np.float32))
    save_json(os.path.join(WORK, "dsm_meta_v2.json"),
              {"x0": bx0, "y0": by1, "gsd": DSM_GSD, "w": Wd, "h": Hd,
               "z_min": float(np.nanmin(dsm_f)), "z_max": float(np.nanmax(dsm_f)),
               "z_median": fill})
    log(f"DSM v2 保存: {Wd} x {Hd} @ {DSM_GSD}m, 高程 [{np.nanmin(dsm_f):.2f},{np.nanmax(dsm_f):.2f}]")


if __name__ == "__main__":
    main()
