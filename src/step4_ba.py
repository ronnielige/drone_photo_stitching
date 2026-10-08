"""Step4: 构建 track、三角化、带 POS 先验的鲁棒光束法平差(BA)。"""
import os
import math
import numpy as np
from scipy.optimize import least_squares
from common import (WORK, log, load_json, save_json, Camera,
                    rotvec_to_R, R_to_rotvec)

SIG_POS = 2.0                    # GPS 先验 标准差 (m)
SIG_ROT = math.radians(3.0)      # 姿态先验 (rad)
MAX_DRIFT = 15.0                 # 相对 GPS 的最大允许漂移 (m)
Z_GROUND_MARGIN = 60.0           # 点云高度相对初始地面的容差 (m)


def build_tracks(feats, matches, pairs):
    """并查集构建 track。返回 tracks: list of list[(img, feat_idx)]"""
    n_img = len(feats)
    base = np.zeros(n_img + 1, dtype=np.int64)
    for i in range(n_img):
        base[i + 1] = base[i] + len(feats[i]["des"])
    total = int(base[-1])
    parent = np.arange(total, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for (i, j, _), m in zip(pairs, matches):
        if len(m) == 0:
            continue
        a = base[i] + m[:, 0]
        b = base[j] + m[:, 1]
        for x, y in zip(a, b):
            union(int(x), int(y))

    idx = np.arange(total)
    while True:
        p = parent[idx]
        if np.array_equal(p, idx):
            break
        idx = p
    roots = idx
    order = np.argsort(roots, kind="stable")
    rsorted = roots[order]
    boundaries = np.flatnonzero(np.diff(rsorted)) + 1
    groups = np.split(order, boundaries)
    tracks = []
    for g in groups:
        if len(g) < 2:
            continue
        items = []
        seen_img = set()
        ok = True
        for node in g:
            i = int(np.searchsorted(base, node, side="right") - 1)
            fi = int(node - base[i])
            if i in seen_img:
                ok = False
                break
            seen_img.add(i)
            items.append((i, fi))
        if ok:
            tracks.append(items)
    return tracks


def _triangulate_chunk(tracks, feats, Rs, Cs, cam, min_parallax_deg=1.0):
    M = len(tracks)
    X = np.full((M, 3), np.nan, dtype=np.float64)
    if M == 0:
        return X, np.zeros(0, bool)
    MAXOBS = 10
    maxc = int(max(min(len(t), MAXOBS) for t in tracks))
    obs_nx = np.zeros((M, maxc))
    obs_ny = np.zeros((M, maxc))
    obs_i = np.full((M, maxc), -1, dtype=np.int64)
    for k, t in enumerate(tracks):
        for c, (i, fi) in enumerate(t[:MAXOBS]):
            obs_nx[k, c] = feats[i]["nx"][fi]
            obs_ny[k, c] = feats[i]["ny"][fi]
            obs_i[k, c] = i
    valid = obs_i >= 0
    dirs = np.stack([obs_nx, obs_ny, np.ones_like(obs_nx)], axis=2)
    Rall = np.zeros((M, maxc, 3, 3))
    Call = np.zeros((M, maxc, 3))
    mk = valid
    Rall[mk] = Rs[obs_i[mk]]
    Call[mk] = Cs[obs_i[mk]]
    d = np.einsum("mcij,mcj->mci", Rall, dirs)
    d = d / np.maximum(np.linalg.norm(d, axis=2, keepdims=True), 1e-12)
    d[~mk] = 0.0
    I = np.eye(3)
    P = I - np.einsum("mci,mcj->mcij", d, d)
    P[~mk] = 0.0
    A = P.sum(axis=1)
    b = np.einsum("mcij,mcj->mci", P, Call).sum(axis=1)
    ok = np.abs(np.linalg.det(A)) > 1e-8
    Xs = np.zeros((M, 3))
    if ok.any():
        Xs[ok] = np.linalg.solve(A[ok], b[ok])
    zc = np.einsum("mcij,mcj->mci", Rall.transpose(0, 1, 3, 2), (Xs[:, None, :] - Call))
    front = (zc[:, :, 2] > 1.0)
    front_cnt = (front & mk).sum(axis=1)
    n_obs = valid.sum(axis=1)
    d0 = d[:, 0, :]
    d1 = d[np.arange(M), np.minimum(np.maximum(n_obs - 1, 0), maxc - 1)]
    parl = np.degrees(np.arccos(np.clip((d0 * d1).sum(axis=1), -1, 1)))
    good = ok & (front_cnt >= 2) & (n_obs >= 2) & (parl > min_parallax_deg)
    X[good] = Xs[good]
    return X, good, parl


def triangulate_batch(tracks, feats, Rs, Cs, cam, min_parallax_deg=1.0, chunk=60000):
    n = len(tracks)
    X = np.full((n, 3), np.nan, dtype=np.float64)
    g = np.zeros(n, dtype=bool)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        X[s:e], g[s:e], _ = _triangulate_chunk(tracks[s:e], feats, Rs, Cs, cam,
                                               min_parallax_deg)
    return X, g


def camera_residual(p, Xs, obs, fx, fy, cx, cy, C0, r0, sp, sr):
    r = p[:3]
    C = p[3:6]
    R = rotvec_to_R(r)
    Xc = (Xs - C) @ R
    z = Xc[:, 2]
    denom = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = fx * Xc[:, 0] / denom + cx
    v = fy * Xc[:, 1] / denom + cy
    return np.concatenate([u - obs[:, 0], v - obs[:, 1],
                           (C - C0) / sp, (r - r0) / sr])


def main():
    meta = load_json(os.path.join(WORK, "meta.json"))
    cj = load_json(os.path.join(WORK, "camera.json"))
    cam = Camera(cj["w"], cj["h"], *cj["init"])
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy

    z = np.load(os.path.join(WORK, "feats.npz"), allow_pickle=True)
    feats = [{k: z[f"{i}_{k}"] for k in ("px", "nx", "ny", "scale", "ang", "resp", "des")}
             for i in range(len(meta))]
    zm = np.load(os.path.join(WORK, "matches.npz"), allow_pickle=True)
    keys = sorted(zm.files, key=lambda s: int(s.split("_")[0]))
    matches = [zm[k] for k in keys]
    pairs = load_json(os.path.join(WORK, "pairs.json"))

    Cs = np.load(os.path.join(WORK, "init_C.npy"))
    Rs = np.load(os.path.join(WORK, "init_R.npy"))
    C_gps = Cs.copy()                                        # 固定先验
    r_gps = np.array([R_to_rotvec(Rs[i]) for i in range(len(meta))])
    n = len(meta)
    ground0 = load_json(os.path.join(WORK, "ground0.json"))["z_ground"]

    tc = os.path.join(WORK, "tracks.npz")
    if os.path.exists(tc):
        tz = np.load(tc, allow_pickle=True)
        tr_i, tr_f, tr_off = tz["ti"], tz["tf"], tz["off"]
        tracks = [list(zip(tr_i[tr_off[k]:tr_off[k + 1]].tolist(),
                           tr_f[tr_off[k]:tr_off[k + 1]].tolist()))
                  for k in range(len(tr_off) - 1)]
        log(f"读取缓存 tracks: {len(tracks)}")
    else:
        tracks = build_tracks(feats, matches, pairs)
        tr_i = np.concatenate([np.array([a for a, _ in t], dtype=np.int32) for t in tracks])
        tr_f = np.concatenate([np.array([b for _, b in t], dtype=np.int32) for t in tracks])
        off = np.zeros(len(tracks) + 1, dtype=np.int64)
        off[1:] = np.cumsum([len(t) for t in tracks])
        np.savez_compressed(tc, ti=tr_i, tf=tr_f, off=off)
        log(f"构建 tracks: {len(tracks)}")

    X, good = triangulate_batch(tracks, feats, Rs, Cs, cam, min_parallax_deg=1.0)
    log(f"初始三角化成功: {good.sum()} / {len(tracks)}")

    # 组织按相机的观测
    cam_tr = [[] for _ in range(n)]
    for k, t in enumerate(tracks):
        for (i, fi) in t:
            cam_tr[i].append((k, fi))
    cam_arr = [np.array(v, dtype=np.int64) if v else np.zeros((0, 2), dtype=np.int64)
               for v in cam_tr]
    obs_uv = []
    for i in range(n):
        a = cam_arr[i]
        if len(a) == 0:
            obs_uv.append(np.zeros((0, 2)))
        else:
            obs_uv.append(np.stack([feats[i]["px"][a[:, 1], 0],
                                    feats[i]["px"][a[:, 1], 1]], 1).astype(np.float64))
    keep = good.copy()                      # track 是否有效
    inlier = [np.ones(len(a), dtype=bool) for a in cam_arr]
    # 观测 -> track 的映射
    obs_trk = [a[:, 0] for a in cam_arr]

    def residuals_all():
        out = []
        for i in range(n):
            a = cam_arr[i]
            if len(a) == 0:
                out.append(np.zeros(0))
                continue
            sel = inlier[i] & keep[a[:, 0]]
            Xs = X[a[:, 0]]
            R = Rs[i]
            Xc = (Xs - Cs[i]) @ R
            zz = np.where(np.abs(Xc[:, 2]) < 1e-6, 1e-6, Xc[:, 2])
            u = fx * Xc[:, 0] / zz + cx
            v = fy * Xc[:, 1] / zz + cy
            du = u - obs_uv[i][:, 0]
            dv = v - obs_uv[i][:, 1]
            r = np.sqrt(du ** 2 + dv ** 2)
            r[~sel] = np.nan
            out.append(r)
        return out

    def report(tag=""):
        allr = np.concatenate([r[~np.isnan(r)] for r in residuals_all()])
        if len(allr) == 0:
            return 0.0
        log(f"  {tag}重投影误差 px: p50={np.percentile(allr,50):.3f} "
            f"p90={np.percentile(allr,90):.3f} p99={np.percentile(allr,99):.3f} "
            f"mean={allr.mean():.3f} n={len(allr)}")
        return float(np.percentile(allr, 90))

    report("初始 ")

    N_ITER = int(os.environ.get("BA_ITER", "8"))
    thresh_seq = [30.0, 15.0, 8.0, 5.0, 4.0, 3.0, 3.0, 2.5, 2.5, 2.5]
    for it in range(N_ITER):
        th = thresh_seq[min(it, len(thresh_seq) - 1)]
        # 1) 逐相机优化（仅用内点）
        for i in range(n):
            a = cam_arr[i]
            if len(a) == 0:
                continue
            sel = inlier[i] & keep[a[:, 0]]
            if sel.sum() < 40:
                continue
            Xs = X[a[sel, 0]]
            obs = obs_uv[i][sel]
            p0 = np.concatenate([R_to_rotvec(Rs[i]), Cs[i]])
            sol = least_squares(
                camera_residual, p0, method="trf", loss="huber", f_scale=2.0,
                xtol=1e-9, ftol=1e-9, max_nfev=40,
                args=(Xs, obs, fx, fy, cx, cy, C_gps[i], r_gps[i], SIG_POS, SIG_ROT))
            Rn = rotvec_to_R(sol.x[:3])
            Cn = sol.x[3:6]
            # 漂移限制：不允许位姿偏离 GPS/初始姿态过远
            dv = Cn - C_gps[i]
            nd = np.linalg.norm(dv)
            if nd > MAX_DRIFT:
                Cn = C_gps[i] + dv * (MAX_DRIFT / nd)
            dR = Rn.T @ Rs[i]
            ang = math.degrees(math.acos(max(-1, min(1, (np.trace(dR) - 1) / 2))))
            if ang > 6.0 and it < 4:
                # 早期迭代限制单步旋转幅度
                ax = R_to_rotvec(dR)
                na = np.linalg.norm(ax)
                if na > 1e-9:
                    ax = ax * (math.radians(6.0) / na)
                Rn = Rs[i] @ rotvec_to_R(ax)
            Rs[i] = Rn
            Cs[i] = Cn
        # 2) 剔除大残差观测
        for i in range(n):
            r = residuals_all()[i]
            inlier[i] = (~np.isnan(r)) & (np.nan_to_num(r, nan=1e9) < th)
        # 3) 重新三角化（用内点）
        live = [tracks[k] for k in range(len(tracks)) if keep[k]]
        Xin, gin = triangulate_batch(live, feats, Rs, Cs, cam, min_parallax_deg=0.4)
        idx = np.flatnonzero(keep)
        for j, k in enumerate(idx):
            if gin[j]:
                X[k] = Xin[j]
        # 高度合理性过滤
        bad = keep & ((X[:, 2] < ground0 - 20) | (X[:, 2] > ground0 + Z_GROUND_MARGIN))
        keep &= ~bad
        log(f"迭代 {it+1}/{N_ITER} (阈值 {th}px): 保留 track {keep.sum()}, "
            f"高度异常剔除 {bad.sum()}")
        report("  ")

    np.save(os.path.join(WORK, "ba_R.npy"), Rs)
    np.save(os.path.join(WORK, "ba_C.npy"), Cs)
    np.save(os.path.join(WORK, "ba_X.npy"), X[keep])
    save_json(os.path.join(WORK, "ba_info.json"),
              {"n_tracks": int(len(tracks)), "n_valid": int(keep.sum())})
    log(f"BA 完成，保留 {keep.sum()} 个 3D 点")

    atp = os.path.join(WORK, "at_pose.npy")
    if os.path.exists(atp):
        from common import euler_to_R
        at = np.load(atp, allow_pickle=True)
        d = np.linalg.norm(at[:, 3:6] - Cs, axis=1)
        angs = []
        for i in range(n):
            dR = euler_to_R(*at[i, :3]).T @ Rs[i]
            angs.append(math.degrees(math.acos(max(-1, min(1, (np.trace(dR) - 1) / 2)))))
        angs = np.array(angs)
        log(f"[对比大疆AT] 位置差 mean={d.mean():.2f}m p90={np.percentile(d,90):.2f}m "
            f"max={d.max():.2f}m")
        log(f"[对比大疆AT] 角度差 mean={angs.mean():.3f}deg max={angs.max():.3f}deg")


if __name__ == "__main__":
    main()
