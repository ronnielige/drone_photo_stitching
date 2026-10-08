# -*- coding: utf-8 -*-
"""Step6b: 正射渲染 v2 —— 逐像素「单帧胜出」(best-view selection)。

针对用户反馈的「模糊 / 扭曲 / 拼接痕迹」的根本改动：
  1) 取消多视加权平均：每个输出像素只取「质量最高的那一张影像」的颜色，
     从机制上消除多帧平均造成的钝化与重影。
  2) 视图评分 = 天底角余弦^P × 离主点径向衰减 × 局部纹理清晰度：
     最正、最靠影像中心、最清晰的那一帧胜出（纹理项权重小，只做近并列时的裁决）。
  3) 可见性(遮挡)检测：沿「相机→地面点」光线按步采样 DSM，若中途地形高于光线，
     该视图被判为遮挡、禁止参与选取——避免建筑像素被抹到旁边地面形成"波纹"。
  4) 接缝：默认纯硬切换（SEAM_EPS=0）；仅当 SEAM_EPS>0 时，在「胜出/次优得分
     几乎相等」的窄过渡带内做线性过渡（过渡带宽度由 eps 自动决定）。
  5) 色彩：沿用同名点全局最小二乘增益，使相邻帧色彩一致，硬切换也难看出边界。
"""
import os
import math
import time
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor
from common import (WORK, OUT, log, load_json, save_json, Camera,
                    imread_any, imwrite_any)
from step6_ortho import image_ortho_bbox

# ---------------- 参数 ----------------
TILE = 1024
MAX_VIEWS = 14            # 每个分块最多参与竞选的视图数
NADIR_P = 4.0             # 天底度权重指数
RAD_P = 0.6               # 径向衰减指数
TEX_W = float(os.environ.get("TEX_W", "0.0"))   # 纹理项权重，0=纯几何评分（选区最干净）
TEX_REF = 10.0            # 纹理归一化参考（灰度局部 std）
OCC_MARGIN = 0.6          # 遮挡判定余量 (m)
OCC_STEPS = 12            # 遮挡采样步数
COS_MIN = 0.55            # 参与竞选的最小天底余弦
SEAM_EPS = float(os.environ.get("SEAM_EPS", "0.0"))
TRACK2 = SEAM_EPS > 1e-9
TEX_DIR = os.path.join(WORK, "tex")

_IMG_CACHE = {}
_TEX_CACHE = {}
_IMG_CACHE_MAX = 6


# ---------------- 采样工具 ----------------
def dsm_sample(dsm, dmeta, X, Y):
    """双线性采样 DSM（展平数组版），返回 (z, 是否落在 DSM 范围内)"""
    gsd = dmeta["gsd"]
    x0d, y0d = dmeta["x0"], dmeta["y0"]
    H, W = dsm.shape
    fx = (X - x0d) / gsd - 0.5
    fy = (y0d - Y) / gsd - 0.5
    ix = np.floor(fx).astype(np.int32)
    iy = np.floor(fy).astype(np.int32)
    inb = (ix >= 0) & (ix < W - 1) & (iy >= 0) & (iy < H - 1)
    ixc = np.clip(ix, 0, W - 2)
    iyc = np.clip(iy, 0, H - 2)
    tx = (fx - ixc).astype(np.float32)
    ty = (fy - iyc).astype(np.float32)
    z = (dsm[iyc, ixc] * (1 - tx) * (1 - ty) + dsm[iyc, ixc + 1] * tx * (1 - ty) +
         dsm[iyc + 1, ixc] * (1 - tx) * ty + dsm[iyc + 1, ixc + 1] * tx * ty)
    return z.astype(np.float32), inb


def occl_mask(Xw, C, dsm, dmeta, nstep=OCC_STEPS):
    """沿相机->地面点光线采样 DSM，判断是否被地形遮挡（2.5D 精确判据）。

    抗噪：要求至少 2 个采样点持续超限，或单点超限很大，才判定遮挡，
    避免 DSM 局部噪声导致视图像素级碎片化切换。
    """
    n = Xw.shape[0]
    dx = Xw[:, 0] - C[0]
    dy = Xw[:, 1] - C[1]
    dz = Xw[:, 2] - C[2]
    cnt = np.zeros(n, np.int16)
    mx = np.zeros(n, np.float32)
    for t in np.linspace(0.03, 0.97, nstep):
        zr = C[2] + t * dz
        zt, inb = dsm_sample(dsm, dmeta, C[0] + t * dx, C[1] + t * dy)
        exc = np.where(inb, zt - zr, -1e9)
        np.maximum(mx, exc, out=mx)
        cnt += (exc > OCC_MARGIN)
    return (cnt >= 2) | (mx > 2.5 * OCC_MARGIN)


def _get_img(path):
    d = _IMG_CACHE
    img = d.get(path)
    if img is not None:
        return img
    img = imread_any(path)
    if img is not None:
        d[path] = img
        while len(d) > _IMG_CACHE_MAX:
            d.pop(next(iter(d)))
    return img


def _get_tex(i):
    t = _TEX_CACHE.get(i)
    if t is not None:
        return t
    p = os.path.join(TEX_DIR, f"{i:03d}.npy")
    if not os.path.exists(p):
        return None
    t = np.load(p)
    _TEX_CACHE[i] = t
    return t


def make_tex(i, path):
    """1/8 分辨率灰度局部 std（清晰度代理，尺度≈2.8m，保证选区不碎片化），uint8 量化"""
    g = imread_any(path, gray=True)
    if g is None:
        return None
    g2 = cv2.resize(g, (max(1, g.shape[1] // 8), max(1, g.shape[0] // 8)),
                    interpolation=cv2.INTER_AREA).astype(np.float32)
    m1 = cv2.boxFilter(g2, -1, (7, 7), normalize=True, borderType=cv2.BORDER_REFLECT)
    m2 = cv2.boxFilter(g2 * g2, -1, (7, 7), normalize=True, borderType=cv2.BORDER_REFLECT)
    sd = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    return np.clip(sd * 8.0, 0, 255).astype(np.uint8)


def _make_tex_task(args):
    i, path, out = args
    if os.path.exists(out):
        return i
    t = make_tex(i, path)
    if t is not None:
        np.save(out, t)
    return i


# ---------------- 单块 / 单窗口渲染 ----------------
def _project_views(gx0, gy0, gsd, tw, th, views, cam_p, Rs, Cs, paths,
                   dsm_path, dmeta, z_fallback, radio=None, gains=None, dsm=None):
    """render_rect 与 render_views 共用的投影骨架。

    对该矩形逐个有效视图 yield (i, col(N,3)float32, geom(N)float32,
    s(N)float32, ok(N)bool)。颜色已应用辐射校正（radio 优先，否则 gains）。
    dsm 可外部传入已加载的数组，避免重复 np.load（step6d 会大量调用）。
    """
    cam = Camera(*cam_p)
    if dsm is None:
        dsm = np.load(dsm_path)   # 全量入内存，避免 mmap 随机访问抖动
    xs = gx0 + (np.arange(tw) + 0.5) * gsd
    ys = gy0 - (np.arange(th) + 0.5) * gsd
    X, Y = np.meshgrid(xs, ys)
    Xf = X.ravel().astype(np.float64)
    Yf = Y.ravel().astype(np.float64)
    Zf, inb = dsm_sample(dsm, dmeta, Xf, Yf)
    Zf = np.where(inb, Zf, np.float32(z_fallback))
    Xw = np.stack([Xf, Yf, Zf.astype(np.float64)], axis=1)
    N = Xw.shape[0]

    if views is None:
        views = list(range(len(paths)))

    for i in views:
        R, C = Rs[i], Cs[i]
        rel = Xw - C[None, :]
        dist = np.linalg.norm(rel, axis=1)
        dist_s = np.maximum(dist, 1e-6)
        cosn = -rel[:, 2] / dist_s
        Xc = rel @ R
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
        m = 8
        ru = np.abs(u - cam.cx) / (cam.w / 2.0)
        rv = np.abs(v - cam.cy) / (cam.h / 2.0)
        rr = np.maximum(ru, rv)
        ok = ((z > 1.0) & (cosn > COS_MIN) & (u > m) & (v > m) &
              (u < cam.w - m) & (v < cam.h - m) & (rr < 0.98))
        if not ok.any():
            continue
        # ---- 遮挡剔除 ----
        k = np.flatnonzero(ok)
        oc = occl_mask(Xw[k], C, dsm, dmeta)
        if oc.any():
            ok[k[oc]] = False
        if not ok.any():
            continue
        # ---- 评分 ----
        geom = np.zeros(N, np.float32)
        geom[ok] = (np.clip(cosn[ok], 0, 1) ** NADIR_P) * \
                   (np.cos(np.clip(rr[ok], 0, 1) * math.pi / 2.0) ** RAD_P)
        tex = _get_tex(i)
        if TEX_W > 0 and tex is not None:
            mapx = (u * 0.125).reshape(th, tw).astype(np.float32)
            mapy = (v * 0.125).reshape(th, tw).astype(np.float32)
            tmap = cv2.remap(tex, mapx, mapy, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0).ravel()
            tmap = tmap.astype(np.float32) * (1.0 / 8.0)
        else:
            tmap = np.zeros(N, np.float32)
        s = geom * (1.0 + TEX_W * np.clip(tmap / TEX_REF, 0, 1.5))
        s[~ok] = -1e9
        # ---- 取色 ----
        img = _get_img(paths[i])
        if img is None:
            continue
        mapx = u.reshape(th, tw).astype(np.float32)
        mapy = v.reshape(th, tw).astype(np.float32)
        col = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        col = col.reshape(-1, 3).astype(np.float32)
        if radio is not None:
            col = col * radio[i][0][None, :] + radio[i][1][None, :]
        elif gains is not None:
            col = col * gains[i][None, :]
        yield i, col, geom, s, ok


def render_views(gx0, gy0, gsd, tw, th, views, cam_p, Rs, Cs, paths, gains,
                 dsm_path, dmeta, z_fallback, radio=None, dsm=None):
    """按指定视图列表渲染一个矩形，返回 {i: (col_uint8, geom_f32, ok_bool)}。

    供接缝线优化（step6d）按需获取任意候选视图的颜色与得分。
    """
    out = {}
    for i, col, geom, s, ok in _project_views(
            gx0, gy0, gsd, tw, th, views, cam_p, Rs, Cs, paths,
            dsm_path, dmeta, z_fallback, radio=radio, gains=gains, dsm=dsm):
        out[i] = (np.clip(col, 0, 255).astype(np.uint8).reshape(th, tw, 3),
                  geom.reshape(th, tw), ok.reshape(th, tw))
    return out


def render_rect(gx0, gy0, gsd, tw, th, views, cam_p, Rs, Cs, paths, gains,
                dsm_path, dmeta, z_fallback, also_avg=False, radio=None, debug=False):
    """硬性单帧选取渲染一个矩形区域。

    返回 (bgr_uint8, view_idx_int16, valid_bool, score1_float32[, avg_bgr])

    radio: (n,2,3) 逐帧仿射辐射校正 [[a_R,a_G,a_B],[b_R,b_G,b_B]]；给了就替代 gains。
    debug=True 时返回 dict，额外含 s2/col1/col2/nvis —— 供接缝实验离线复用中间量。
    """
    N = tw * th
    s1 = np.full(N, -1e9, np.float32)
    s2 = np.full(N, -1e9, np.float32)
    col1 = np.zeros((N, 3), np.float32)
    col2 = np.zeros((N, 3), np.float32)
    best = np.full(N, -1, np.int16)
    nvis = np.zeros(N, np.uint8)
    if also_avg:
        acc = np.zeros((N, 3), np.float64)
        wsum = np.zeros(N, np.float64)
    else:
        acc = wsum = None

    for i, col, geom, s, ok in _project_views(
            gx0, gy0, gsd, tw, th, views, cam_p, Rs, Cs, paths,
            dsm_path, dmeta, z_fallback, radio=radio, gains=gains):
        if also_avg:
            g_ = np.where(ok, geom, 0.0)
            acc += col * g_[:, None]
            wsum += g_
        take = s > s1
        if TRACK2:
            old1 = s1
            old2 = s2
            better2 = (~take) & (s > old2)
            col2 = np.where(take[:, None], col1,
                            np.where(better2[:, None], col, col2))
            s2 = np.where(take, old1, np.where(better2, s, old2))
        col1 = np.where(take[:, None], col, col1)
        s1 = np.where(take, s, s1)
        best = np.where(take, np.int16(i), best)
        nvis[ok] += 1

    valid = best >= 0
    out = np.zeros((N, 3), np.float32)
    out[valid] = col1[valid]
    if TRACK2:
        a = np.clip(0.5 + (s1 - s2) / (2.0 * SEAM_EPS), 0.0, 1.0)
        a[s2 < -1e8] = 1.0
        bl = a[:, None] * col1 + (1.0 - a[:, None]) * col2
        out = np.where(valid[:, None], bl, 0.0)
    bgr = np.clip(out, 0, 255).astype(np.uint8).reshape(th, tw, 3)
    avg = None
    if also_avg:
        a2 = np.zeros((N, 3), np.float32)
        w = wsum > 1e-9
        a2[w] = (acc[w] / wsum[w, None]).astype(np.float32)
        avg = np.clip(a2, 0, 255).astype(np.uint8).reshape(th, tw, 3)
    if debug:
        return dict(bgr=bgr, vidx=best.reshape(th, tw), valid=valid.reshape(th, tw),
                    s1=s1.reshape(th, tw), s2=s2.reshape(th, tw),
                    col1=np.clip(col1, 0, 255).astype(np.uint8).reshape(th, tw, 3),
                    col2=np.clip(col2, 0, 255).astype(np.uint8).reshape(th, tw, 3),
                    nvis=nvis.reshape(th, tw), avg=avg)
    return bgr, best.reshape(th, tw), valid.reshape(th, tw), s1.reshape(th, tw), avg


def _render_tile(args):
    (tx, ty, gx0, gy0, gsd, tw, th, views, cam_p, Rs, Cs, paths, gains,
     dsm_path, dmeta, z_fb, radio) = args
    r = render_rect(gx0, gy0, gsd, tw, th, views, cam_p, Rs, Cs, paths, gains,
                    dsm_path, dmeta, z_fb, also_avg=True, radio=radio)
    return (tx, ty) + r[:3] + (r[4],)


# ---------------- 主流程 ----------------
def main():
    meta = load_json(os.path.join(WORK, "meta.json"))
    cj = load_json(os.path.join(WORK, "camera.json"))
    cam = Camera(cj["w"], cj["h"], *cj["init"])
    cam_p = (cj["w"], cj["h"], *cj["init"])
    paths = [m["path"] for m in meta]
    n = len(meta)
    Rs = np.load(os.path.join(WORK, "ba_R.npy"))
    Cs = np.load(os.path.join(WORK, "ba_C.npy"))
    gains = np.load(os.path.join(WORK, "gains.npy"))
    radio_path = os.path.join(WORK, "radio.npy")
    radio = np.load(radio_path) if os.path.exists(radio_path) else None
    if radio is not None:
        log(f"使用逐帧仿射辐射校正 work/radio.npy  "
            f"a[{radio[:,0,0].min():.2f},{radio[:,0,0].max():.2f}] "
            f"b[{radio[:,1,0].min():.1f},{radio[:,1,0].max():.1f}]  (取代乘性增益)")
    else:
        log("未找到 radio.npy，使用乘性增益 gains.npy")

    dsm_v2 = os.path.join(WORK, "dsm_v2.npy")
    if not os.path.exists(dsm_v2):
        raise SystemExit("缺少 dsm_v2.npy，请先运行 step5c_dsm_refine.py")
    dm2 = load_json(os.path.join(WORK, "dsm_meta_v2.json"))
    gsd_o = float(os.environ.get("ORTHO_GSD", "0.055"))
    gx0, gy0 = dm2["x0"], dm2["y0"]
    W = int(math.ceil(dm2["w"] * dm2["gsd"] / gsd_o))
    H = int(math.ceil(dm2["h"] * dm2["gsd"] / gsd_o))
    log(f"正射网格 v2: {W} x {H} @ {gsd_o*100:.1f}cm  (左上 {gx0:.2f},{gy0:.2f})")
    z_fb = float(dm2["z_median"])
    grid = {"x0": gx0, "y0": gy0, "gsd": gsd_o, "w": W, "h": H}
    save_json(os.path.join(WORK, "grid.json"), grid)

    ortho_ids = [i for i in range(n) if -Rs[i][2, 2] > 0.85]
    log(f"参与渲染的近似正射影像: {len(ortho_ids)} / {n}")

    # ---- 纹理图预计算（仅在启用纹理项时需要）----
    if TEX_W > 0:
        os.makedirs(TEX_DIR, exist_ok=True)
        todo = [(i, paths[i], os.path.join(TEX_DIR, f"{i:03d}.npy"))
                for i in ortho_ids if not os.path.exists(os.path.join(TEX_DIR, f"{i:03d}.npy"))]
        if todo:
            log(f"预计算纹理图 {len(todo)} 张 ...")
            with ProcessPoolExecutor(max_workers=min(12, os.cpu_count())) as ex:
                list(ex.map(_make_tex_task, todo, chunksize=2))
            log("纹理图完成")
    else:
        log("TEX_W=0：跳过纹理图（纯几何评分）")

    # ---- 覆盖 bbox + 分块任务 ----
    zr = (dm2["z_min"] - 2.0, dm2["z_max"] + 2.0)
    bboxes = [image_ortho_bbox(i, Rs, Cs, cam, grid, zr) for i in range(n)]
    np.save(os.path.join(WORK, "ortho_bboxes_v2.npy"), np.array(bboxes, dtype=np.int32))
    tasks = []
    nview_stat = []
    ntiles_x = int(math.ceil(W / TILE))
    ntiles_y = int(math.ceil(H / TILE))
    for ty in range(ntiles_y):
        for tx in range(ntiles_x):
            x0, y0 = tx * TILE, ty * TILE
            tw = min(TILE, W - x0)
            th = min(TILE, H - y0)
            cands = []
            for i in ortho_ids:
                c0, r0, c1, r1 = bboxes[i]
                ox = max(0, min(c1, x0 + tw) - max(c0, x0))
                oy = max(0, min(r1, y0 + th) - max(r0, y0))
                if ox > 0 and oy > 0:
                    frac = (ox * oy) / float(tw * th)
                    cands.append((frac * (0.25 + 0.75 * (-Rs[i][2, 2])), i))
            if not cands:
                continue
            cands.sort(key=lambda t: -t[0])
            views = [i for _, i in cands[:MAX_VIEWS]]
            nview_stat.append(len(cands))
            tasks.append((tx, ty, gx0 + x0 * gsd_o, gy0 - y0 * gsd_o, gsd_o, tw, th,
                          views, cam_p, Rs, Cs, paths, gains, dsm_v2, dm2, z_fb, radio))
    nview_stat = np.array(nview_stat)
    log(f"分块 {len(tasks)} 个; 每块候选视图 均值 {nview_stat.mean():.1f} "
        f"中位 {np.median(nview_stat):.0f} 最大 {nview_stat.max()}")

    out_img = np.zeros((H, W, 3), dtype=np.uint8)
    avg_img = np.zeros((H, W, 3), dtype=np.uint8)
    vidx = np.full((H, W), -1, dtype=np.int16)
    vmask = np.zeros((H, W), dtype=bool)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as ex:
        for tx, ty, blk, bidx, bval, bavg in ex.map(_render_tile, tasks, chunksize=1):
            x0, y0 = tx * TILE, ty * TILE
            th, tw = blk.shape[:2]
            out_img[y0:y0 + th, x0:x0 + tw] = blk
            if bavg is not None:
                avg_img[y0:y0 + th, x0:x0 + tw] = bavg
            vidx[y0:y0 + th, x0:x0 + tw] = bidx
            vmask[y0:y0 + th, x0:x0 + tw] = bval
            done += 1
            if done % 30 == 0 or done == len(tasks):
                log(f"  渲染 {done}/{len(tasks)} 块 ({time.time()-t0:.0f}s), "
                    f"有效率 {bval.mean()*100:.1f}%")
    log(f"渲染完成: 有效率 {vmask.mean()*100:.2f}%, 用时 {time.time()-t0:.0f}s")

    # ---- 视图使用统计 ----
    uu = vidx[vmask]
    uniq, cnt = np.unique(uu, return_counts=True)
    log(f"实际使用影像 {len(uniq)} 张; 占比前5: " +
        ", ".join(f"#{int(u)} {c/len(uu)*100:.1f}%" for u, c in
                  sorted(zip(uniq, cnt), key=lambda t: -t[1])[:5]))
    np.save(os.path.join(OUT, "ortho_rgb.npy"), out_img)
    np.save(os.path.join(OUT, "ortho_avg.npy"), avg_img)
    np.save(os.path.join(OUT, "ortho_valid.npy"), vmask)
    np.save(os.path.join(OUT, "ortho_vidx.npy"), vidx)
    save_json(os.path.join(OUT, "ortho_meta.json"),
              {"x0": gx0, "y0": gy0, "gsd": gsd_o, "w": W, "h": H,
               "method": "single-best-view (hard selection)",
               "radio": bool(radio is not None),
               "seam_eps": SEAM_EPS, "max_views": MAX_VIEWS})
    log("已保存 ortho_rgb.npy / ortho_avg.npy / ortho_valid.npy / ortho_vidx.npy / ortho_meta.json")


if __name__ == "__main__":
    main()
