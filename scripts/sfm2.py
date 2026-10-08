import os, sys, json, pickle, math, time
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(__file__))
from config import (load_meta, to_local, CACHE, get_intrinsics_k,
                    K1, K2, K3, P1, P2, camera_from_gimbal)

MAX_TRACK_OBS = 24
# progressive reprojection thresholds for outlier rejection (px)
OBS_THRESH = [40, 20, 12, 8, 6, 5, 4, 3, 3, 3, 3, 3]
HUBER = 2.5
GPS_SIGMA = 3.0

def rodrigues_batch(Rs):
    out = np.zeros((len(Rs), 3))
    for i, R in enumerate(Rs):
        out[i] = cv2.Rodrigues(R)[0].ravel()
    return out

def rot_from_aa(aa):
    return cv2.Rodrigues(aa.reshape(3, 1))[0]

def undist_px(px, K, D):
    cu = cv2.undistortPoints(px[:, None].astype(np.float64), K, D).reshape(-1, 2)
    return np.stack([K[0, 0] * cu[:, 0] + K[0, 2], K[1, 1] * cu[:, 1] + K[1, 2]], 1)

def build_tracks(pair_list, matches, kpcoords):
    key = {}
    parent = []
    obs = []
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for (i, j) in pair_list:
        id_i, id_j = matches[(i, j)]
        for ki, kj in zip(id_i, id_j):
            ci = (i, int(ki)); cj = (j, int(kj))
            for c in (ci, cj):
                if c not in key:
                    key[c] = len(obs)
                    parent.append(len(obs))
                    cam, kpid = c
                    u, v = kpcoords[cam][int(kpid)]
                    obs.append((cam, int(kpid), float(u), float(v)))
            union(key[ci], key[cj])
    from collections import defaultdict
    tracks_map = defaultdict(list)
    for oid in range(len(obs)):
        tracks_map[find(oid)].append(obs[oid])
    return [v for v in tracks_map.values() if 2 <= len(v) <= MAX_TRACK_OBS]

def triangulate(Rs_l, Cs_l, pn):
    A = []
    for R, C, p in zip(Rs_l, Cs_l, pn):
        P = np.hstack([R, (-R @ C)[:, None]])
        r1, r2, r3 = P[0], P[1], P[2]
        A.append(p[0] * r3 - r1)
        A.append(p[1] * r3 - r2)
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]

def main():
    t00 = time.time()
    local, _ = to_local(load_meta())
    N = len(local)
    K = get_intrinsics_k()
    D = np.array([K1, K2, P1, P2, K3], float)

    with open(os.path.join(CACHE, "matches2.json")) as f:
        pair_list = [tuple(p) for p in json.load(f)]
    mz = np.load(os.path.join(CACHE, "matches2.npz"))
    matches = {}
    for (i, j) in pair_list:
        key = f"{i}_{j}"
        if key in mz:
            m = mz[key]
            matches[(i, j)] = (m[:, 0].astype(int), m[:, 1].astype(int))
    with open(os.path.join(CACHE, "feats2.pkl"), "rb") as f:
        kpcoords = pickle.load(f)
    print("pairs:", len(matches), flush=True)

    tracks = build_tracks(pair_list, matches, kpcoords)
    nobs = sum(len(t) for t in tracks)
    print(f"tracks: {len(tracks)}  obs: {nobs}", flush=True)

    # undistort to ideal px
    ku_all = []
    for t in tracks:
        ku_all.append([(c, u, v) for (c, _k, u, v) in t])
    # cameras init from gimbal/GPS
    Cs = np.zeros((N, 3)); Rs = np.zeros((N, 3, 3)); gps_C = np.zeros((N, 3))
    for i, r in enumerate(local):
        C, R = camera_from_gimbal(r)
        Cs[i] = C; Rs[i] = R; gps_C[i] = C

    # ---- flatten obs, sort by track ----
    T = len(tracks)
    obs_cam = np.zeros(nobs, np.int32)
    obs_uv = np.zeros((nobs, 2))
    obs_track = np.zeros(nobs, np.int32)
    o = 0
    for tid, t in enumerate(ku_all):
        for (c, u, v) in t:
            obs_cam[o] = c; obs_uv[o] = (u, v); obs_track[o] = tid
            o += 1
    order = np.argsort(obs_track, kind="stable")
    obs_cam = obs_cam[order]; obs_uv = obs_uv[order]; obs_track = obs_track[order]
    # undistort raw SIFT pixel coords to ideal pinhole pixels (matches pinhole projection below)
    obs_uv = undist_px(obs_uv, K, D)
    seg_starts = np.flatnonzero(np.r_[True, obs_track[1:] != obs_track[:-1]])
    seg_sizes = np.diff(np.r_[seg_starts, nobs])

    # ---- initial triangulation (gate: must reproject <30px on every obs) ----
    Xs = np.zeros((T, 3))
    for tid in range(T):
        s, e = seg_starts[tid], (seg_starts[tid+1] if tid+1 < T else nobs)
        cams = obs_cam[s:e]
        uv = obs_uv[s:e]
        pn = [((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]) for u, v in uv]
        Rs_l = [Rs[c] for c in cams]
        Cs_l = [Cs[c] for c in cams]
        try:
            X = triangulate(Rs_l, Cs_l, pn)
        except Exception:
            continue
        if X is None or not np.all(np.isfinite(X)) or not (0 < X[2] < 600):
            continue
        good = True
        for c, (u, v) in zip(cams, uv):
            vv = Rs[c] @ (X - Cs[c])
            if vv[2] <= 1e-6:
                good = False; break
            pu = K[0, 0] * vv[0] / vv[2] + K[0, 2]
            pv = K[1, 1] * vv[1] / vv[2] + K[1, 2]
            if math.hypot(pu - u, pv - v) > 30.0:
                good = False; break
        if good:
            Xs[tid] = X
    print("init triangulated:", int((np.abs(Xs).sum(1) > 0).sum()), f"({time.time()-t00:.0f}s)", flush=True)

    # ---- PnP bootstrap rounds (pose-only, robust) ----
    from scipy.optimize import least_squares
    for rnd, fscale in enumerate([6.0, 3.0]):
        for c in range(N):
            m = obs_cam == c
            if m.sum() < 20:
                continue
            tids = obs_track[m]; uv = obs_uv[m]
            Xp = Xs[tids]
            ok = np.abs(Xp).sum(1) > 0
            if ok.sum() < 20:
                continue
            def res(p, Xp, uv, ok):
                R = rot_from_aa(p[:3]); C = p[3:6]
                v = (Xp[ok] - C) @ R.T
                z = np.maximum(v[:, 2], 1e-6)
                pu = K[0, 0] * v[:, 0] / z + K[0, 2]
                pv = K[1, 1] * v[:, 1] / z + K[1, 2]
                r = np.concatenate([pu - uv[ok][:, 0], pv - uv[ok][:, 1]])
                gp = (C - gps_C[c]) / np.array([GPS_SIGMA, GPS_SIGMA, GPS_SIGMA * 1.5])
                return np.concatenate([r, gp])
            p0 = np.concatenate([rodrigues_batch(Rs[c:c+1])[0], Cs[c]])
            try:
                sol = least_squares(res, p0, args=(Xp, uv, ok), loss="huber",
                                    f_scale=fscale, max_nfev=200)
            except Exception:
                continue
            Rs[c] = rot_from_aa(sol.x[:3]); Cs[c] = sol.x[3:6]
        # re-triangulate (gate: median reproj <= 8px else invalidate)
        for tid in range(T):
            s, e = seg_starts[tid], (seg_starts[tid+1] if tid+1 < T else nobs)
            cams = obs_cam[s:e]; uv = obs_uv[s:e]
            pn = [((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]) for u, v in uv]
            try:
                X = triangulate([Rs[c] for c in cams], [Cs[c] for c in cams], pn)
            except Exception:
                continue
            if X is None or not np.all(np.isfinite(X)) or not (0 < X[2] < 600):
                Xs[tid] = 0.0
                continue
            reps = []
            for c, (u, v) in zip(cams, uv):
                vv = Rs[c] @ (X - Cs[c])
                if vv[2] <= 1e-6:
                    reps.append(1e9); continue
                pu = K[0, 0] * vv[0] / vv[2] + K[0, 2]
                pv = K[1, 1] * vv[1] / vv[2] + K[1, 2]
                reps.append(math.hypot(pu - u, pv - v))
            if float(np.median(reps)) <= 8.0:
                Xs[tid] = X
            else:
                Xs[tid] = 0.0
        print(f"pnp round {rnd} done ({time.time()-t00:.0f}s)", flush=True)

    # ---- free-net GN with Schur complement ----
    HZ = 20.0  # cam0 anchor weight (large)
    lam_c = 1e-4
    f = K[0, 0]; cx0, cy0 = K[0, 2], K[1, 2]
    prev_rmse = np.inf

    def project_all():
        Xo = Xs[obs_track]
        v = np.einsum("nij,nj->ni", Rs[obs_cam], Xo - Cs[obs_cam])
        z = np.maximum(v[:, 2], 1e-9)
        x = v[:, 0] / z; y = v[:, 1] / z
        pu = f * x + cx0; pv = f * y + cy0
        return v, z, x, y, pu, pv

    for it in range(len(OBS_THRESH)):
        v, z, x, y, pu, pv = project_all()
        r = obs_uv - np.stack([pu, pv], 1)
        rn = np.hypot(r[:, 0], r[:, 1])
        thr = OBS_THRESH[it]
        # re-evaluate live set each iteration (valid tracks & residual gate) so
        # observations excluded during early iterations can revive as poses converge
        live = (np.abs(Xs).sum(1) > 0)[obs_track] & (rn < thr)
        w = np.where(rn <= HUBER, 1.0, HUBER / np.maximum(rn, 1e-9)) * live
        # jacobians
        Jpv = np.zeros((nobs, 2, 3))
        Jpv[:, 0, 0] = f / z; Jpv[:, 0, 2] = -f * x / z
        Jpv[:, 1, 1] = f / z; Jpv[:, 1, 2] = -f * y / z
        vx = np.zeros((nobs, 3, 3))
        vx[:, 0, 1] = -v[:, 2]; vx[:, 0, 2] = v[:, 1]
        vx[:, 1, 0] = v[:, 2];  vx[:, 1, 2] = -v[:, 0]
        vx[:, 2, 0] = -v[:, 1]; vx[:, 2, 1] = v[:, 0]
        Jw = -np.einsum("nai,nij->naj", Jpv, vx)          # (n,2,3) d p / d omega
        Rc = Rs[obs_cam]
        Jc = -np.einsum("nai,nij->naj", Jpv, Rc)           # (n,2,3) d p / d C
        Jcam = np.concatenate([Jw, Jc], axis=2)            # (n,2,6)
        Jpt = np.einsum("nai,nij->naj", Jpv, Rc)           # (n,2,3)
        # blocks  (jacobs are (n,2,X): axis1=residual dim, axis2=param dim)
        B = np.einsum("nia,nib->nab", Jcam, Jpt) * w[:, None, None]   # (n,6,3)
        Hpp = np.add.reduceat(np.einsum("nia,nib->nab", Jpt, Jpt) * w[:, None, None], seg_starts)
        gp = np.add.reduceat(np.einsum("nia,ni->na", Jpt, w[:, None] * r), seg_starts)
        # direct camera terms via bincount
        Aobs = np.einsum("nia,nib->nab", Jcam, Jcam) * w[:, None, None]  # (n,6,6)
        nH = 6 * N
        ci6 = obs_cam * 6
        ridx = (ci6[:, None] + np.arange(6)[None, :])      # (n,6)
        flat_r = np.repeat(ridx, 6, axis=1)                 # (n,36)
        flat_c = np.tile(ci6[:, None] + np.arange(6)[None, :], (1, 6))
        Hflat = np.bincount((flat_r * nH + flat_c).ravel(),
                            weights=Aobs.reshape(nobs, 36).ravel(), minlength=nH * nH)
        gc = np.bincount(ridx.ravel(), weights=np.einsum("nia,ni->na", Jcam, w[:, None] * r).reshape(-1),
                         minlength=nH)
        # schur pairs
        m2 = seg_sizes.astype(np.int64) ** 2
        total_pairs = int(m2.sum())
        pair_track = np.repeat(np.arange(T, dtype=np.int32), m2)
        cums = np.concatenate([[0], np.cumsum(m2)])
        ploc = np.arange(total_pairs, dtype=np.int32) - np.repeat(cums[:-1].astype(np.int32), m2)
        mseg = np.repeat(seg_sizes.astype(np.int32), m2)
        i_loc = ploc // mseg
        j_loc = ploc % mseg
        start_of = seg_starts[pair_track].astype(np.int32)
        ki = start_of + i_loc
        kj = start_of + j_loc
        Hinv = np.linalg.inv(Hpp + 1e-9 * np.eye(3)[None] * np.maximum(np.trace(Hpp, axis1=1, axis2=2)[:, None, None], 1e-6))
        # schur RHS part per obs: -B_k Hinv gp
        sc_rhs = np.einsum("nij,nj->ni", B, np.einsum("tij,tj->ti", Hinv, gp)[obs_track])
        gc -= np.bincount(ridx.ravel(), weights=sc_rhs.reshape(-1), minlength=nH)
        # schur blocks in chunks
        CH = 400000
        for a in range(0, total_pairs, CH):
            b = min(a + CH, total_pairs)
            kia, kja = ki[a:b], kj[a:b]
            Pblk = np.einsum("nij,nkj->nik", np.einsum("nij,njk->nik", B[kia], Hinv[pair_track[a:b]]), B[kja])
            ri = (obs_cam[kia] * 6)[:, None] + np.arange(6)[None, :]
            cj = (obs_cam[kja] * 6)[:, None] + np.arange(6)[None, :]
            fr = np.repeat(ri, 6, axis=1).ravel().astype(np.int64)
            fc = np.tile(cj, (1, 6)).ravel().astype(np.int64)
            Hflat -= np.bincount(fr * nH + fc, weights=Pblk.reshape(-1, 36).ravel(), minlength=nH * nH)
        H = Hflat.reshape(nH, nH)
        # gauge: anchor cam0 (rotation+position)
        H[0:6, 0:6] += HZ * np.eye(6)
        # damping
        diag = np.diag(H).copy()
        H[range(nH), range(nH)] = diag * (1 + lam_c) + 1e-9
        try:
            dc = np.linalg.solve(H, gc)
        except np.linalg.LinAlgError:
            lam_c *= 10
            continue
        # point updates
        dc = dc.reshape(N, 6)
        dc_obs = dc[obs_cam]     # (n,6)
        dX = np.einsum("tij,tj->ti", Hinv, gp - np.add.reduceat(np.einsum("nab,na->nb", B, dc_obs), seg_starts))
        # apply with simple LM step control
        Rn = Rs.copy(); Cn = Cs.copy(); Xn = Xs.copy()
        for c in range(N):
            dw = dc[c, :3]
            th = np.linalg.norm(dw)
            if th < 1e-12:
                Rn[c] = Rs[c]
            else:
                Rn[c] = rot_from_aa(dw) @ Rs[c]
        Cn = Cs + dc[:, 3:6]
        Xn = Xs + dX
        # evaluate
        Rs_try, Cs_try, Xs_try = Rn, Cn, Xn
        save = (Rs, Cs, Xs)
        Rs, Cs, Xs = Rs_try, Cs_try, Xs_try
        _, _, _, _, pu2, pv2 = project_all()
        r2 = obs_uv - np.stack([pu2, pv2], 1)
        rn2 = np.hypot(r2[:, 0], r2[:, 1])
        rmse_new = math.sqrt((rn2[live] ** 2).mean())
        rmse_old = math.sqrt((rn[live] ** 2).mean())
        if rmse_new < rmse_old:
            lam_c = max(lam_c * 0.5, 1e-6)
            prev_rmse = rmse_new
        else:
            Rs, Cs, Xs = save
            lam_c *= 5
            prev_rmse = rmse_old
        print(f"gn it{it}: rmse={min(rmse_new, rmse_old):.3f}px live={int(live.sum())}/{nobs} lam={lam_c:.1e} ({time.time()-t00:.0f}s)", flush=True)

    # ---- similarity align to GPS (free-net -> georeference, Umeyama with scale) ----
    mu_s = Cs.mean(0); mu_d = gps_C.mean(0)
    A_ = Cs - mu_s; B_ = gps_C - mu_d
    H_ = A_.T @ B_ / N
    U, S_, Vt = np.linalg.svd(H_)
    d_ = np.sign(np.linalg.det(Vt.T @ U.T))
    Rg = Vt.T @ np.diag([1, 1, d_]) @ U.T
    var_s = (A_ ** 2).sum() / N
    s_ = float((S_ * np.array([1.0, 1.0, d_])).sum() / max(var_s, 1e-12))
    tt = mu_d - s_ * Rg @ mu_s
    Cs = s_ * (Rg @ Cs.T).T + tt
    Rs = np.einsum("nij,jk->nik", Rs, Rg.T)
    Xs = s_ * (Rg @ Xs.T).T + tt
    print("similarity align done: scale %.5f, rot %.2f deg, shift %.2f m" %
          (s_, np.rad2deg(np.arccos(np.clip((np.trace(Rg) - 1) / 2, -1, 1))),
           float(np.linalg.norm(tt))), flush=True)

    with open(os.path.join(CACHE, "sfm_cam.pkl"), "wb") as fp:
        pickle.dump({"Cs": Cs, "Rs": Rs, "gps_C": gps_C}, fp)
    with open(os.path.join(CACHE, "tracks3.pkl"), "wb") as fp:
        pickle.dump({"tracks": tracks, "Xs": Xs, "obs_track": obs_track,
                     "obs_cam": obs_cam, "seg_starts": seg_starts, "live": live}, fp)

    # final RMSE report
    v, z, x, y, pu, pv = project_all()
    r = obs_uv - np.stack([pu, pv], 1)
    rn = np.hypot(r[:, 0], r[:, 1])
    print("FINAL rmse %.3f px (%d live obs)" % (math.sqrt((rn[live] ** 2).mean()), int(live.sum())), flush=True)
    print("cam shift vs gps: med %.2f m, max %.2f m" %
          (np.median(np.linalg.norm(Cs - gps_C, axis=1)), np.linalg.norm(Cs - gps_C, axis=1).max()), flush=True)
    print("sfm2 done in %.0fs" % (time.time() - t00), flush=True)

if __name__ == "__main__":
    main()