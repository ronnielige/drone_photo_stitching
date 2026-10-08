import os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(__file__))
from config import get_intrinsics_k, IMG_W, IMG_H, gimbal_rot2

print("cv2", cv2.__version__)
K = get_intrinsics_k()
D = np.zeros(5)

def rectify_pts(uv1, uv2, P1m, P2m):
    r1 = cv2.undistortPoints(uv1.astype(np.float64), K, D).reshape(-1, 2)
    r2 = cv2.undistortPoints(uv2.astype(np.float64), K, D).reshape(-1, 2)
    h1 = np.hstack([r1, np.ones((len(r1), 1))]) @ P1m[:, :3].T
    h2 = np.hstack([r2, np.ones((len(r2), 1))]) @ P2m[:, :3].T
    return h1[:, 0] / h1[:, 2], h1[:, 1] / h1[:, 2], h2[:, 0] / h2[:, 2], h2[:, 1] / h2[:, 2]

# ---------- 1) identity pair sanity ----------
print("=== identity pair: R=I, T=[20,0,0] ===")
Rr = np.eye(3)
Tr = np.array([[20.], [0.], [0.]])
R1o, R2o, P1m, P2m, Q, _, _ = cv2.stereoRectify(K, D, K, D, (IMG_W, IMG_H), Rr, Tr,
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
print("R1o:\n", np.round(R1o, 6))
print("P1:\n", np.round(P1m, 3))
print("P2:\n", np.round(P2m, 3))
print("Q:\n", np.round(Q, 6))
# world point (0,0,100) in cam1 frame -> cam2 frame: x2 = x1 + T = (-20,0,100)
def px(xc):
    return np.array([K[0, 0] * xc[0] / xc[2] + K[0, 2], K[1, 1] * xc[1] / xc[2] + K[1, 2]])
u1, v1 = px(np.array([0., 0., 100.]))
u2, v2 = px(np.array([-20., 0., 100.]))
u1r, v1r, u2r, v2r = rectify_pts(np.array([[u1, v1]]), np.array([[u2, v2]]), P1m, P2m)
print("rect: dv=%.4f  du=%.2f (expect dv~0, du~%.0f)" % (v1r[0] - v2r[0], u1r[0] - u2r[0], K[0, 0] * 0.2))

# ---------- 2) nadir synthetic A vs B, print P diff ----------
print("=== nadir A vs B ===")
R1 = gimbal_rot2(0.0, -90.0, 0.0)
C1 = np.array([0., 0., 286.])
R2 = gimbal_rot2(12.0, -88.5, 1.5)
C2 = np.array([30., 8., 286.6])
rng = np.random.default_rng(7)
n = 3000
X = np.stack([rng.uniform(-140, 140, n), rng.uniform(-140, 140, n), rng.uniform(40, 160, n)], 1)
def proj(R, C):
    xc = (X - C) @ R.T
    u = K[0, 0] * xc[:, 0] / xc[:, 2] + K[0, 2]
    v = K[1, 1] * xc[:, 1] / xc[:, 2] + K[1, 2]
    m = (xc[:, 2] > 0.5) & (u > 3) & (u < IMG_W - 4) & (v > 3) & (v < IMG_H - 4)
    return np.stack([u, v], 1), m
uv1, m1 = proj(R1, C1)
uv2, m2 = proj(R2, C2)
m = m1 & m2
uv1, uv2 = uv1[m], uv2[m]

for name, Rr, Tr in [("A", R2 @ R1.T, R2 @ (C1 - C2)),
                     ("B", R1 @ R2.T, R1 @ (C2 - C1))]:
    Tr = Tr.reshape(3, 1)
    print(name, "R:\n", np.round(Rr, 4), "\nT:", Tr.ravel())
    R1o, R2o, P1m, P2m, Q, _, _ = cv2.stereoRectify(K, D, K, D, (IMG_W, IMG_H), Rr, Tr,
                                                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
    print(" R1o:\n", np.round(R1o, 4), "\n P1[:, 3]:", np.round(P1m[:, 3], 1),
          " P2[:, 3]:", np.round(P2m[:, 3], 1))
    u1r, v1r, u2r, v2r = rectify_pts(uv1, uv2, P1m, P2m)
    dv = v1r - v2r
    du = u1r - u2r
    print(" med|dv|=%.4f med_du=%.2f" % (np.median(np.abs(dv)), np.median(du)))

# ---------- 3) same but D=real distortion ----------
Dreal = np.array([-0.10893391, 0.00334538, -0.00062201, -0.00026050, -0.02004984])
print("=== with real distortion ===")
def rectify_pts2(uv1, uv2, P1m, P2m):
    r1 = cv2.undistortPoints(uv1.astype(np.float64), K, Dreal).reshape(-1, 2)
    r2 = cv2.undistortPoints(uv2.astype(np.float64), K, Dreal).reshape(-1, 2)
    h1 = np.hstack([r1, np.ones((len(r1), 1))]) @ P1m[:, :3].T
    h2 = np.hstack([r2, np.ones((len(r2), 1))]) @ P2m[:, :3].T
    return h1[:, 0] / h1[:, 2], h1[:, 1] / h1[:, 2], h2[:, 0] / h2[:, 2], h2[:, 1] / h2[:, 2]

def proj_d(R, C):
    xc = (X - C) @ R.T
    x = xc[:, 0] / xc[:, 2]; y = xc[:, 1] / xc[:, 2]
    r2 = x * x + y * y
    rd = 1 + Dreal[0] * r2 + Dreal[1] * r2 ** 2 + Dreal[4] * r2 ** 3
    xd = x * rd + 2 * Dreal[2] * x * y + Dreal[3] * (r2 + 2 * x * x)
    yd = y * rd + 2 * Dreal[3] * x * y + Dreal[2] * (r2 + 2 * y * y)
    u = K[0, 0] * xd + K[0, 2]; v = K[1, 1] * yd + K[1, 2]
    m = (xc[:, 2] > 0.5) & (u > 3) & (u < IMG_W - 4) & (v > 3) & (v < IMG_H - 4)
    return np.stack([u, v], 1), m

uv1, m1 = proj_d(R1, C1)
uv2, m2 = proj_d(R2, C2)
m = m1 & m2
uv1, uv2 = uv1[m], uv2[m]
for name, Rr, Tr in [("A", R2 @ R1.T, R2 @ (C1 - C2)),
                     ("B", R1 @ R2.T, R1 @ (C2 - C1))]:
    R1o, R2o, P1m, P2m, Q, _, _ = cv2.stereoRectify(K, Dreal, K, Dreal, (IMG_W, IMG_H),
                                                    Rr, Tr.reshape(3, 1),
                                                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
    u1r, v1r, u2r, v2r = rectify_pts2(uv1, uv2, P1m, P2m)
    dv = v1r - v2r
    du = u1r - u2r
    print("%s med|dv|=%.4f med_du=%.2f  (R1o[0,:2]=%s)" %
          (name, np.median(np.abs(dv)), np.median(du), np.round(R1o[0, :2], 4)))
