import os, json, glob
import numpy as np
from pyproj import Transformer

BASE = r"D:\lr\project\无人机航拍拼接"
SRC = os.path.join(BASE, "xiangjia1_186m")
SC_OUT = os.path.join(BASE, "mysolution", "out")
SCRIPTS = os.path.join(BASE, "mysolution", "scripts")
CACHE = os.path.join(BASE, "mysolution", "cache")
for d in (SC_OUT, SCRIPTS, CACHE):
    os.makedirs(d, exist_ok=True)

# DJI Terra calibrated intrinsic for the M3E wide camera (pixels at 5280x3956)
IMG_W, IMG_H = 5280, 3956
F = 3720.60
CX, CY = 2664.36, 1952.64
K1, K2, K3, P1, P2 = -0.10893391, 0.00334538, -0.02004984, -0.00062201, -0.00026050

wgs2utm = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)  # UTM 50N

def load_meta():
    rows = json.load(open(os.path.join(SC_OUT, "metadata.json")))
    return rows

def to_local(rows):
    """Return arrays E,N,H (m, UTM50N) with origin shifted to block center; plus per-row metadata."""
    Es, Ns, Hs = [], [], []
    for r in rows:
        E, N = wgs2utm.transform(r["lon"], r["lat"])
        Es.append(E); Ns.append(N); Hs.append(r["alt"])
    E0 = np.mean(Es); N0 = np.mean(Ns)
    out = []
    for r, E, N, H in zip(rows, Es, Ns, Hs):
        d = dict(r)
        d["E"] = E - E0; d["N"] = N - N0; d["H"] = H
        out.append(d)
    return out, (E0, N0)

def get_intrinsics_k():
    return np.array([[F, 0, CX], [0, F, CY], [0, 0, 1]], float)

def gimbal_rot2(yaw_deg, pitch_deg, roll_deg):
    """World ENU (X=E,Y=N,Z=up). Build world->cam rotation for nadir drone camera.
    yaw: gimbal yaw deg (0=north, clockwise+). pitch: gimbal pitch  (=-90 nadir).
    roll: gimbal roll deg.
    """
    yaw = np.deg2rad(yaw_deg); pitch = np.deg2rad(pitch_deg); roll = np.deg2rad(roll_deg)
    # heading rotation about world Z, clockwise by yaw (so forward north-> yaw toward east)
    ch, sh = np.cos(-yaw), np.sin(-yaw)
    Rh = np.array([[ch, -sh, 0], [sh, ch, 0], [0, 0, 1]], float)
    # base nadir cam axes
    cxd = Rh @ np.array([1.0, 0, 0]); cyd = Rh @ np.array([0.0, -1, 0]); czd = Rh @ np.array([0, 0, -1.0])
    R0 = np.vstack([cxd, cyd, czd])  # world->cam rows
    def Rx(a):
        ca, sa = np.cos(a), np.sin(a)
        return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], float)
    def Ry(a):
        ca, sa = np.cos(a), np.sin(a)
        return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], float)
    # gimbal pitch: camera pitched looking further; pitch = -90 at nadir.
    # deviation off nadir: pitch_off = pitch - (-90) in degrees
    pitch_off = np.deg2rad(pitch_deg + 90.0)
    # Apply roll about camera x, then pitch rotation about camera y, to the base
    R = Rx(roll) @ R0
    R = Ry(pitch_off) @ R
    return R

def camera_from_gimbal(meta):
    """C: world position (E,N,H). R: world->cam."""
    C = np.array([meta["E"], meta["N"], meta["H"]], float)
    mk = meta["maker"]
    R = gimbal_rot2(float(mk["dji:GimbalYawDegree"]),
                    float(mk["dji:GimbalPitchDegree"]),
                    float(mk["dji:GimbalRollDegree"]))
    return C, R

if __name__ == "__main__":
    rows = load_meta()
    local, (E0, N0) = to_local(rows)
    # sanity: center GPS
    import numpy as np
    Es = np.array([r["E"] for r in local]); Ns = np.array([r["N"] for r in local])
    print("block span E:", Es.min(), Es.max(), "N:", Ns.min(), Ns.max(),
          "size m:", Es.max()-Es.min(), "x", Ns.max()-Ns.min())
    # footprint radius estimate (half diagonal) via intrinsics+alt
    r0 = rows[0]; C0, R0 = camera_from_gimbal(local[0])
    H0 = C0[2]
    half_w = 0.5*IMG_W/ F * H0
    half_h = 0.5*IMG_H/ F * H0
    half_d = np.hypot(half_w, half_h)
    print("sample alt", H0, "foot halfW", half_w, "halfH", half_h, "halfD", half_d)
    # validate: project image center (optical axis) to ground -> should ~ GPS point
    K = get_intrinsics_k()
    # optical axis direction in cam = z_c=(0,0,1); ray in world = R.T@(0,0,1)
    ray = R0.T @ np.array([0,0,1.0]); ray = ray/np.linalg.norm(ray[:2])
    # camera looks down so ray.z<0
    print("ray", ray, "sign z", ray[2])
    # expected GPS center == C0 (nadir). Check orientation of foot: project image corners to ground at z=0
    # to see footprint alignment with heading
    corners = np.array([[0,0,1],[IMG_W,0,1],[IMG_W,IMG_H,1],[0,IMG_H,1]], float)
    Foot=[]
    for p in corners:
        # cam ray dir
        d = R0.T @ (np.linalg.inv(K) @ p)  # dir in world, unbounded (fine)
        d = d / np.linalg.norm(d)
        if d[2] >= -1e-9: Foot.append(None); continue
        t = (0 - C0[2]) / d[2]
        X = C0 + t*d
        Foot.append(X)
    print("footprint corner EN (should form box):")
    for X in Foot:
        if X is not None:
            print("   E%+.1f N%+.1f" % (X[0], X[1]))
    print("heading should orient long dimension ~ angle", np.rad2deg(np.arctan2(ray[0], ray[1])))