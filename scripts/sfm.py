import os, sys, json, pickle, math
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(__file__))
from config import (load_meta, to_local, CACHE, get_intrinsics_k,
                    K1, K2, K3, P1, P2, camera_from_gimbal, IMG_W, IMG_H)

def rodrigues(R):
    return cv2.Rodrigues(R)[0].ravel()

def rot_vect(aa):
    return cv2.Rodrigues(np.asarray(aa, float).reshape(3, 1))[0]

def undist_px(px, K, D):
    """px: (N,2) distorted pixel coords -> (N,2) undistorted pixel coords."""
    cu = cv2.undistortPoints(px[:, None].astype(np.float64), K, D)
    cu = cu.reshape(-1, 2)
    u = K[0, 0] * cu[:, 0] + K[0, 2]
    v = K[1, 1] * cu[:, 1] + K[1, 2]
    return np.stack([u, v], 1)

def triangulate(Rs, Cs, pts_norm):
    """Linear least-squares triangulation of one point seen by multiple cameras.
    Rs: list (3,3) world->cam. Cs: list (3,) cam centers. pts_norm: list (2,) normalized coords.
    Returns world 3D point."""
    A = []
    for R, C, p in zip(Rs, Cs, pts_norm):
        P = np.hstack([R, (-R @ C)[:, None]])  # 3x4 world->cam
        r1, r2, r3 = P[0], P[1], P[2]
        A.append(p[0] * r3 - r1)
        A.append(p[1] * r3 - r2)
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return X[:3] * 0  # degenerate
    return X[:3] / X[3]

def project_point(R, C, X, K):
    xc = R @ (X - C)
    if xc[2] <= 1e-9:
        return None
    pp = K @ (xc / xc[2])
    return pp

def build_tracks(pair_list, matches, kpcoords):
    """Return tracks: each track = list of (cam, kpt_id, u, v) observations.
    pair_list: list of (i,j). matches: dict (i,j)->(id_i, id_j). kpcoords[i]=(Ni,2,)."""
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
            for c, ki_local in ((ci, ki), (cj, kj)):
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
        r = find(oid)
        tracks_map[r].append(obs[oid])
    tracks = [v for v in tracks_map.values() if len(v) >= 2]
    return tracks

def main():
    subset = int(sys.argv[1]) if len(sys.argv) > 1 else None
    local, _ = to_local(load_meta())
    if subset is not None:
        local = local[:subset]
    N = len(local)
    K = get_intrinsics_k()
    D = np.array([K1, K2, P1, P2, K3], float)  # opencv order [k1,k2,p1,p2,k3]

    # load matches (keypoint index pairs) and feature coords
    with open(os.path.join(CACHE, "matches.json")) as f:
        pair_list = [tuple(p) for p in json.load(f)]
    pair_list = [p for p in pair_list if p[0] < N and p[1] < N]
    mz = np.load(os.path.join(CACHE, "matches.npz"))
    matches = {}
    for (i, j) in pair_list:
        key = f"{i}_{j}"
        if key in mz:
            matches[(i, j)] = (mz[key][:, 0].astype(int), mz[key][:, 1].astype(int))
    with open(os.path.join(CACHE, "feats.pkl"), "rb") as f:
        kpcoords = pickle.load(f)
    kpcoords = [np.asarray(c, float) for c in kpcoords]
    print("valid pairs:", len(matches))

    # undistort all feature coords to ideal pixels
    ku = [undist_px(kpcoords[i], K, D) for i in range(N)]

    # camera init poses
    Cs, Rs = [], []
    for r in local:
        C, R = camera_from_gimbal(r)
        Cs.append(C); Rs.append(R)
    Cs = np.array(Cs); Rs = np.array(Rs)
    gps_C = Cs.copy()

    tracks = build_tracks(pair_list, matches, ku)
    print("tracks:", len(tracks), "obs/track med",
          int(np.median([len(t) for t in tracks])), "max", max(len(t) for t in tracks))
    # cap tracks count for speed? keep all.
    # per-camera obs list: list for each cam of list of (trackid, u, v)
    cam_obs = [[] for _ in range(N)]
    for tid, tr in enumerate(tracks):
        for (cam, _k, u, v) in tr:
            cam_obs[cam].append((tid, float(u), float(v)))

    # Initial triangulation
    Xs = np.zeros((len(tracks), 3))
    valid = np.zeros(len(tracks), bool)
    for tid, tr in enumerate(tracks):
        Rs_l = [Rs[c] for (c, _k, _, _) in tr]
        Cs_l = [Cs[c] for (c, _k, _, _) in tr]
        pn = []
        ok = True
        for (c, _k, u, v) in tr:
            # to normalized
            pn.append([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]])
        try:
            X = triangulate(Rs_l, Cs_l, pn)
        except Exception:
            continue
        # sanity: project back
        rep = 0.0
        good = True
        for (c, _k, u, v) in tr:
            pp = project_point(Rs[c], Cs[c], X, K)
            if pp is None: good = False; break
            dist = math.hypot(pp[0] - u, pp[1] - v)
            if not math.isfinite(dist) or dist > 25:
                good = False; break
            rep += dist
        if good:
            Xs[tid] = X; valid[tid] = True
    print("valid tracks after init tri:", valid.sum())

    # undistorted ideal pixel residuals; collect obs arrays for BA
    # store per camera arrays (track id, u_ideal, v_ideal)
    obs_arr = {c: np.array(cam_obs[c], float) if cam_obs[c] else np.zeros((0, 3)) for c in range(N)}
    pt_pairs = {c: (obs_arr[c][:, 1], obs_arr[c][:, 2]) for c in range(N)}  # u,v
    track_ids = {c: obs_arr[c][:, 0].astype(int) for c in range(N)}

    # camera solve helper
    from scipy.optimize import least_squares
    def camera_residuals(params, tids, uv, Xs_sub, gpsC, K):
        aa = params[:3]; C = params[3:6]
        R = rot_vect(aa)
        # reproject
        dX = Xs_sub - C
        Xc = dX @ R.T          # (M,3)
        Xc2 = Xc[:, 2]
        good = Xc2 > 1e-6
        res = np.empty(len(tids) * 2)
        res.fill(500.0)
        if good.any():
            x = Xc[good, 0] / Xc2[good]; y = Xc[good, 1] / Xc2[good]
            pu = K[0, 0] * x + K[0, 2]; pv = K[1, 1] * y + K[1, 2]
            du = pu - uv[0][good]; dv = pv - uv[1][good]
            g = good.nonzero()[0]
            res[g * 2] = du; res[g * 2 + 1] = dv
        # GPS prior (weak)
        res = res * 1.0
        res = np.concatenate([res, (C - gpsC) / np.array([2.5, 2.5, 3.0])])
        return res

    ITER = 8
    hist = []
    for it in range(ITER):
        # solve each camera pose with current points
        nr_o = 0
        for c in range(N):
            tids = track_ids[c]
            if len(tids) < 6:
                continue
            Xs_sub = Xs[tids][valid[tids]]
            tids_v = tids[valid[tids]]
            uv = (pt_pairs[c][0][valid[tids]], pt_pairs[c][1][valid[tids]])
            if len(tids_v) < 6:
                continue
            aa0 = rodrigues(Rs[c]).ravel()
            p0 = np.concatenate([aa0, Cs[c]])
            try:
                sol = least_squares(camera_residuals, p0, args=(tids_v, uv, Xs_sub, gps_C[c], K),
                                    loss='huber', f_scale=3.0, max_nfev=400)
            except Exception:
                continue
            Rs[c] = rot_vect(sol.x[:3]); Cs[c] = sol.x[3:6]
            nr_o += len(tids_v)
        # re-triangulate all tracks
        rej = 0
        for tid in range(len(tracks)):
            if not valid[tid]:
                continue
            tr = tracks[tid]
            Rs_l = [Rs[c] for (c, _k, _, _) in tr]; Cs_l = [Cs[c] for (c, _k, _, _) in tr]
            pn = [[(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]] for (c, _k, u, v) in tr]
            try:
                X = triangulate(Rs_l, Cs_l, pn)
            except Exception:
                valid[tid] = False; rej += 1; continue
            if not (np.all(np.isfinite(X)) and abs(X[2]) < 2000):
                valid[tid] = False; rej += 1; continue
            # check reproj
            reps = []
            okx = True
            for (c, _k, u, v) in tr:
                pp = project_point(Rs[c], Cs[c], X, K)
                if pp is None: okx = False; break
                reps.append(math.hypot(pp[0] - u, pp[1] - v))
            if not okx or np.median(reps) > 6.0:
                valid[tid] = False; rej += 1
            else:
                Xs[tid] = X
        # RMSE
        errs = []
        for c in range(N):
            vi = valid[track_ids[c]]
            tids_v = track_ids[c][vi]; uv = (pt_pairs[c][0][vi], pt_pairs[c][1][vi])
            for k, tid in enumerate(tids_v):
                pp = project_point(Rs[c], Cs[c], Xs[tid], K)
                if pp is None: continue
                errs.append(math.hypot(pp[0]-uv[0][k], pp[1]-uv[1][k]))
        errs = np.array(errs)
        rmse = math.sqrt((errs**2).mean()) if len(errs) else float('nan')
        hist.append((int(valid.sum()), rmse))
        print(f"iter {it}: valid_pts={valid.sum()} rmse_px={rmse:.3f} n_obs={len(errs)}")

    # save
    np.savez_compressed(os.path.join(CACHE, "sfm.npz"),
                        Cs=Cs, Rs=Rs, Xs=Xs, valid=valid, gps_C=gps_C,
                        track_ids=np.array([t for t in track_ids.values()], object, dtype=object) if False else None)
    with open(os.path.join(CACHE, "sfm_cam.pkl"), "wb") as f:
        pickle.dump({"Cs": Cs, "Rs": Rs, "gps_C": gps_C}, f)
    # save tracks with obs for DSM use: (cam list per track)
    with open(os.path.join(CACHE, "tracks.pkl"), "wb") as f:
        pickle.dump({"tracks": tracks, "Xs": Xs, "valid": valid}, f)
    print("final RMSE px:", hist[-1][1], "valid pts:", hist[-1][0])
    print("campaign hist:", hist)

if __name__ == "__main__":
    main()