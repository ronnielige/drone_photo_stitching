import os, sys, json, pickle, time
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(__file__))
from config import CACHE, SRC, IMG_W, IMG_H, get_intrinsics_k, K1, K2, K3, P1, P2

SCALE = 1.0 / 3.0
CELL = 0.25
MIN_B, MAX_B = 15.0, 45.0
MAX_PAIRS = 300
ND_CAP = 512
OUT_DIR = os.path.join(CACHE, "dense")

def load_gray_s(name):
    img = cv2.imdecode(np.fromfile(os.path.join(SRC, name), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    return cv2.resize(img, (int(w * SCALE), int(h * SCALE)), interpolation=cv2.INTER_AREA)

def main():
    t00 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(CACHE, "sfm_cam.pkl"), "rb") as f:
        d = pickle.load(f)
    Cs, Rs = d["Cs"], d["Rs"]
    N = len(Cs)
    K = get_intrinsics_k()
    D = np.array([K1, K2, P1, P2, K3], float)
    K_s = np.array([[K[0, 0] * SCALE, 0, K[0, 2] * SCALE],
                    [0, K[1, 1] * SCALE, K[1, 2] * SCALE],
                    [0, 0, 1.0]])
    Fs = K_s[0, 0]

    with open(os.path.join(CACHE, "tracks3.pkl"), "rb") as f:
        tr = pickle.load(f)
    Xs = tr["Xs"]
    ok = (np.abs(Xs).sum(1) > 0) & (Xs[:, 2] > 50) & (Xs[:, 2] < 400)
    Pts = Xs[ok]
    h_p1, h_p99 = np.percentile(Pts[:, 2], [1, 99])
    print("terrain band: %.1f .. %.1f m, sparse pts %d" % (h_p1, h_p99, len(Pts)), flush=True)

    # stereo pair selection
    pairs = set()
    for i in range(N):
        dd = np.hypot(Cs[:, 0] - Cs[i, 0], Cs[:, 1] - Cs[i, 1])
        cand = [(dd[j], j) for j in range(N) if MIN_B <= dd[j] <= MAX_B and j != i]
        cand.sort()
        for _, j in cand[:3]:
            pairs.add((min(i, j), max(i, j)))
    pairs = sorted(pairs)
    if len(pairs) > MAX_PAIRS:
        step = len(pairs) / MAX_PAIRS
        pairs = [pairs[int(k * step)] for k in range(MAX_PAIRS)]
    print("stereo pairs:", len(pairs), flush=True)

    # grid definition from sparse points (robust bounds)
    e1, e2 = np.percentile(Pts[:, 0], [0.5, 99.5])
    n1, n2 = np.percentile(Pts[:, 1], [0.5, 99.5])
    E0 = np.floor((e1 - 30) / CELL) * CELL
    N0 = np.floor((n1 - 30) / CELL) * CELL
    Emax = np.ceil((e2 + 30) / CELL) * CELL
    Nmax = np.ceil((n2 + 30) / CELL) * CELL
    cols = int(round((Emax - E0) / CELL)) + 1
    rows = int(round((Nmax - N0) / CELL)) + 1
    print("grid %dx%d cells (%.0f x %.0f m)" % (rows, cols, (Emax - E0), (Nmax - N0)), flush=True)

    done = 0
    for (i, j) in pairs:
        outf = os.path.join(OUT_DIR, "p%03d_%03d.npz" % (i, j))
        if os.path.exists(outf):
            done += 1
            continue
        B = float(np.linalg.norm(Cs[i] - Cs[j]))
        zmin = max(min(Cs[i, 2], Cs[j, 2]) - h_p99 - 10, 40.0)
        dmax = Fs * B / zmin
        nd = int(np.ceil((dmax + 24) / 16)) * 16
        if nd > ND_CAP:
            continue
        try:
            gl = load_gray_s(_name(i)); gr = load_gray_s(_name(j))
        except Exception:
            continue
        h, w = gl.shape
        R_rel = Rs[j] @ Rs[i].T
        T_rel = Rs[j] @ (Cs[i] - Cs[j])
        try:
            R1, R2, P1m, P2m, Q, _, _ = cv2.stereoRectify(K_s, D, K_s, D, (w, h), R_rel, T_rel,
                                                          flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
        except Exception:
            continue
        m1x, m1y = cv2.initUndistortRectifyMap(K_s, D, R1, P1m, (w, h), cv2.CV_32F)
        m2x, m2y = cv2.initUndistortRectifyMap(K_s, D, R2, P2m, (w, h), cv2.CV_32F)
        rl = cv2.remap(gl, m1x, m1y, cv2.INTER_LINEAR)
        rr = cv2.remap(gr, m2x, m2y, cv2.INTER_LINEAR)
        sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=5,
                                     P1=8 * 25, P2=32 * 25,
                                     disp12MaxDiff=1, uniquenessRatio=12,
                                     speckleWindowSize=200, speckleRange=2,
                                     preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        disp = sgbm.compute(rl, rr).astype(np.float32) / 16.0
        valid = (disp > 1.0) & (disp < nd - 1.5)
        if valid.sum() < 50000:
            continue
        # 3D in rectified cam-1 frame via Q
        uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        u = uu[valid]; v = vv[valid]; dd_ = disp[valid]
        X_ = Q[0, 0] * u + Q[0, 1] * v + Q[0, 2] * dd_ + Q[0, 3]
        Y_ = Q[1, 0] * u + Q[1, 1] * v + Q[1, 2] * dd_ + Q[1, 3]
        Z_ = Q[2, 0] * u + Q[2, 1] * v + Q[2, 2] * dd_ + Q[2, 3]
        W_ = Q[3, 0] * u + Q[3, 1] * v + Q[3, 2] * dd_ + Q[3, 3]
        W_ = np.where(np.abs(W_) < 1e-9, 1.0, W_)
        xr = np.stack([X_ / W_, Y_ / W_, Z_ / W_], 1)
        # x_rect -> x_cam_orig = R1^T x_rect ; X_world = R_i^T x_cam + C_i  (verified synthetically)
        xw = xr @ R1
        Xw = xw @ Rs[i] + Cs[i]
        m = (Xw[:, 0] > E0) & (Xw[:, 0] < Emax) & (Xw[:, 1] > N0) & (Xw[:, 1] < Nmax) \
            & (Xw[:, 2] > h_p1 - 15) & (Xw[:, 2] < h_p99 + 25)
        Xw = Xw[m]
        if len(Xw) < 30000:
            continue
        cc = np.floor((Xw[:, 0] - E0) / CELL).astype(np.int64)
        rr_ = np.floor((Xw[:, 1] - N0) / CELL).astype(np.int64)
        cid = rr_ * cols + cc
        keep = (cc >= 0) & (cc < cols) & (rr_ >= 0) & (rr_ < rows)
        np.savez_compressed(outf, cid=cid[keep].astype(np.int32), h=Xw[keep, 2].astype(np.float32))
        done += 1
        if done % 20 == 0:
            el = time.time() - t00
            print("dense pairs %d/%d (%.0fs, eta %.0fs)" % (done, len(pairs), el, el / max(done, 1) * (len(pairs) - done)), flush=True)

    print("dense matching done: %d pairs (%.0fs)" % (done, time.time() - t00), flush=True)
    fuse(rows, cols, E0, N0, Pts, t00)

def _name(i):
    global _names
    return _names[i]

_names = None

def fuse(rows, cols, E0, N0, Pts, t00):
    """Two-pass robust fusion of cached per-pair cell heights."""
    files = [os.path.join(OUT_DIR, f) for f in os.listdir(OUT_DIR) if f.endswith(".npz")]
    print("fusing %d pair caches" % len(files), flush=True)
    ncell = rows * cols
    cnt = np.zeros(ncell, np.int32)
    ssum = np.zeros(ncell, np.float64)
    for fp in files:
        z = np.load(fp)
        np.add.at(cnt, z["cid"], 1)
        np.add.at(ssum, z["cid"], z["h"])
    coarse = np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan)
    cnt2 = np.zeros(ncell, np.int32)
    s2 = np.zeros(ncell, np.float64)
    sq2 = np.zeros(ncell, np.float64)
    for fp in files:
        z = np.load(fp)
        ref = coarse[z["cid"]]
        m = np.abs(z["h"] - ref) < 2.5
        c2 = z["cid"][m]; h2 = z["h"][m]
        np.add.at(cnt2, c2, 1)
        np.add.at(s2, c2, h2)
        np.add.at(sq2, c2, h2.astype(np.float64) ** 2)
    mean2 = np.where(cnt2 >= 2, s2 / np.maximum(cnt2, 1), coarse)
    grid = mean2.reshape(rows, cols).astype(np.float32)
    # fill from sparse points where dense failed
    from scipy.ndimage import median_filter, distance_transform_edt
    grid = median_filter(grid, size=3)
    hole = ~np.isfinite(grid)
    if hole.any() and len(Pts):
        E, N, H = Pts[:, 0], Pts[:, 1], Pts[:, 2]
        cc = np.floor((E - E0) / CELL).astype(int); rr = np.floor((N - N0) / CELL).astype(int)
        m = (cc >= 0) & (cc < cols) & (rr >= 0) & (rr < rows)
        sc, sr, sh = cc[m], rr[m], H[m]
        if len(sc):
            order = np.lexsort((sh, sc, sr))
            sc, sr, sh = sc[order], sr[order], sh[order]
            newg = np.r_[True, (sc[1:] != sc[:-1]) | (sr[1:] != sr[:-1])]
            starts = np.flatnonzero(newg)
            sz = np.diff(np.r_[starts, len(sh)])
            take = starts + (sz - 1) // 2
            grid[sr[starts], sc[starts]] = sh[take]
    hole = ~np.isfinite(grid)
    if hole.any():
        # limited nearest fill (max 8 m)
        idx = distance_transform_edt(hole, return_distances=True, return_indices=True)
        dist, (yi, xi) = idx
        near = hole & (dist * CELL < 8.0)
        grid[near] = grid[yi[near], xi[near]]
    print("holes left: %d cells" % int((~np.isfinite(grid)).sum()), flush=True)
    np.save(os.path.join(CACHE, "dsm_dense.npy"), grid)
    with open(os.path.join(CACHE, "dsm_dense_meta.json"), "w") as f:
        json.dump({"E0": float(E0), "N0": float(N0), "cell": CELL}, f)
    gsafe = np.where(np.isfinite(grid), grid, np.nanmedian(grid))
    gy, gx = np.gradient(gsafe)
    print("DSM dense: %.1f..%.1f m, mean|grad| %.3f" %
          (np.nanmin(grid), np.nanmax(grid), np.nanmean(np.hypot(gx, gy))), flush=True)
    # hillshade preview
    az, el = np.deg2rad(315), np.deg2rad(45)
    nx, ny, nz = -gx, gy, np.ones_like(grid)
    nn = np.sqrt(nx**2 + ny**2 + nz**2)
    shade = np.clip((-nx * np.sin(az) * np.cos(el) - ny * np.cos(az) * np.cos(el) + nz * np.sin(el)) / nn, 0, 1)
    prev = (shade * 255).astype(np.uint8)
    prev = np.flipud(prev)
    small = cv2.resize(prev, (min(1400, cols), min(1100, rows)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(r"D:\lr\project\无人机航拍拼接\mysolution\out", "dsm_dense_hillshade.png"), small)
    print("dense_dsm done (%.0fs)" % (time.time() - t00), flush=True)

if __name__ == "__main__":
    with open(os.path.join(CACHE, "sfm_cam.pkl"), "rb") as _f:
        pass
    import json as _json, os as _os
    sys.path.insert(0, _os.path.dirname(__file__))
    from config import load_meta, to_local
    _local, _ = to_local(load_meta())
    _names = [r["name"] for r in _local]
    main()