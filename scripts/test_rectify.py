import os, sys, json, pickle
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(__file__))
from config import (CACHE, get_intrinsics_k, K1, K2, K3, P1, P2, IMG_W, IMG_H, gimbal_rot2)

K = get_intrinsics_k()
D = np.array([K1, K2, P1, P2, K3], float)

def rectify_pts(uv1, uv2, P1m, P2m):
    """raw distorted px -> rectified px, via undistort then P[:, :3] @ ray."""
    r1 = cv2.undistortPoints(uv1.astype(np.float64), K, D).reshape(-1, 2)
    r2 = cv2.undistortPoints(uv2.astype(np.float64), K, D).reshape(-1, 2)
    h1 = np.hstack([r1, np.ones((len(r1), 1))]) @ P1m[:, :3].T
    h2 = np.hstack([r2, np.ones((len(r2), 1))]) @ P2m[:, :3].T
    return h1[:, 0] / h1[:, 2], h1[:, 1] / h1[:, 2], h2[:, 0] / h2[:, 2], h2[:, 1] / h2[:, 2]

def rect(uv1, uv2, Rr, Tr):
    R1o, R2o, P1m, P2m, Q, _, _ = cv2.stereoRectify(
        K, D, K, D, (IMG_W, IMG_H), Rr, Tr,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
    return rectify_pts(uv1, uv2, P1m, P2m)

def test_synth():
    print("=== SYNTHETIC (nadir cams, yaw diff 12deg, B~31m) ===", flush=True)
    rng = np.random.default_rng(7)
    R1 = gimbal_rot2(0.0, -90.0, 0.0)
    C1 = np.array([0., 0., 286.])
    R2 = gimbal_rot2(12.0, -88.5, 1.5)
    C2 = np.array([30., 8., 286.6])
    n = 3000
    X = np.stack([rng.uniform(-140, 140, n), rng.uniform(-140, 140, n),
                  rng.uniform(40, 160, n)], 1)
    def proj(R, C):
        xc = (X - C) @ R.T
        m = xc[:, 2] > 0.5
        u = K[0, 0] * xc[:, 0] / xc[:, 2] + K[0, 2]
        v = K[1, 1] * xc[:, 1] / xc[:, 2] + K[1, 2]
        m &= (u > 3) & (u < IMG_W - 4) & (v > 3) & (v < IMG_H - 4)
        return np.stack([u, v], 1), m
    uv1, m1 = proj(R1, C1)
    uv2, m2 = proj(R2, C2)
    m = m1 & m2
    uv1, uv2 = uv1[m], uv2[m]
    print("visible pts:", len(uv1))
    for name, Rr, Tr in [("A x2=R*x1+T", R2 @ R1.T, R2 @ (C1 - C2)),
                         ("B x1=R*x2+T", R1 @ R2.T, R1 @ (C2 - C1))]:
        u1r, v1r, u2r, v2r = rect(uv1, uv2, Rr, Tr)
        dv = v1r - v2r
        du = u1r - u2r
        print("%-16s med|dv|=%8.4f  med_du=%9.3f  mad_dv=%8.4f" %
              (name, np.median(np.abs(dv)), np.median(du),
               np.median(np.abs(dv - np.median(dv)))), flush=True)

    # reversed baseline: cam2 to the LEFT of cam1 -> check disparity sign handling
    C2b = np.array([-30., -8., 285.4])
    uv2b, m2b = proj(R2, C2b)
    mb = m1 & m2b
    print("--- reversed baseline (cam2 left of cam1) ---")
    for name, Rr, Tr in [("A x2=R*x1+T", R2 @ R1.T, R2 @ (C1 - C2b)),
                         ("B x1=R*x2+T", R1 @ R2.T, R1 @ (C2b - C1))]:
        u1r, v1r, u2r, v2r = rect(uv1[mb], uv2b[mb], Rr, Tr)
        dv = v1r - v2r
        du = u1r - u2r
        print("%-16s med|dv|=%8.4f  med_du=%9.3f" %
              (name, np.median(np.abs(dv)), np.median(du)), flush=True)

def test_real():
    print("=== REAL DATA ===", flush=True)
    with open(os.path.join(CACHE, "sfm_cam.pkl"), "rb") as f:
        d = pickle.load(f)
    Cs, Rs = d["Cs"], d["Rs"]
    N = len(Cs)
    print("cams:", N, " alt: %.1f..%.1f" % (Cs[:, 2].min(), Cs[:, 2].max()))
    dd = []
    for i in range(N):
        for j in range(i + 1, N):
            dd.append((float(np.hypot(Cs[i, 0] - Cs[j, 0], Cs[i, 1] - Cs[j, 1])), i, j))
    dd.sort()
    nn = np.zeros((N, 8))
    for i in range(N):
        ds = np.sort(np.hypot(Cs[:, 0] - Cs[i, 0], Cs[:, 1] - Cs[i, 1]))
        nn[i] = ds[1:9]
    print("NN dist med (1st..8th):", " ".join("%.1f" % v for v in np.median(nn, 0)))
    print("NN dist p90 (1st..4th):", " ".join("%.1f" % v for v in np.percentile(nn[:, :4], 90, 0)))
    hist, edges = np.histogram([b for b, _, _ in dd if b < 120], bins=12, range=(0, 120))
    print("pair distance hist 0-120m:")
    for h, e in zip(hist, edges):
        print("  %5.0f-%5.0f m: %d" % (e, e + 10, h))

    mz = np.load(os.path.join(CACHE, "matches2.npz"))
    with open(os.path.join(CACHE, "feats2.pkl"), "rb") as f:
        kp = pickle.load(f)
    cands = [(b, i, j) for b, i, j in dd if 15 <= b <= 45]
    step = max(1, len(cands) // 8)
    for b, i, j in cands[::step][:8]:
        key = "%d_%d" % (i, j)
        if key not in mz:
            key2 = "%d_%d" % (j, i)
            if key2 in mz:
                mm = mz[key2]
                uv1 = kp[j][mm[:, 0].astype(int)]
                uv2 = kp[i][mm[:, 1].astype(int)]
                i, j = j, i
            else:
                continue
        else:
            mm = mz[key]
            uv1 = kp[i][mm[:, 0].astype(int)]
            uv2 = kp[j][mm[:, 1].astype(int)]
        for name, Rr, Tr in [("A", Rs[j] @ Rs[i].T, Rs[j] @ (Cs[i] - Cs[j])),
                             ("B", Rs[i] @ Rs[j].T, Rs[i] @ (Cs[j] - Cs[i]))]:
            u1r, v1r, u2r, v2r = rect(uv1, uv2, Rr, Tr)
            dv = v1r - v2r
            du = u1r - u2r
            print("pair %3d_%-3d B=%5.1f n=%4d %s med|dv|=%7.3f med_du=%8.2f" %
                  (i, j, b, len(uv1), name, np.median(np.abs(dv)), np.median(du)), flush=True)

if __name__ == "__main__":
    test_synth()
    test_real()
