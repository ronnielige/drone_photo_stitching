"""公共工具：路径、坐标变换、旋转表示、日志。"""
import os
import sys
import time
import math
import numpy as np

# ---- 路径 / 投影配置：换数据集只需设环境变量，无需改代码 ----
#   DRONE_DATA_DIR  数据根目录（下面三项若未单独设置，则从这个目录派生）
#   DRONE_IMG_DIR   航拍原始图片目录（必改：换成新采集目录）
#   DRONE_DJI_DIR   大疆智图工程目录（仅用于对照评估，可不设）
#   DRONE_WORK      中间产物目录
#   DRONE_OUT       最终输出目录
#   DRONE_UTM_ZONE  UTM 带号，默认 50（东经 114~120°）。换地区务必确认！
#   DRONE_SOUTH     南半球设 1（默认 0 = 北半球）
#   DRONE_EPSG      直接指定 EPSG（设了就忽略上面两项）
# 例（Windows cmd）：
#   set DRONE_IMG_DIR=D:\data\new_flight\raw
#   set DRONE_OUT=D:\data\new_flight\out
#   set DRONE_UTM_ZONE=51
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)                       # .../<工程>/mysolution

_DATA_ROOT = os.environ.get("DRONE_DATA_DIR", os.path.dirname(_PROJ))

ROOT = _DATA_ROOT
IMG_DIR = os.environ.get("DRONE_IMG_DIR", os.path.join(_DATA_ROOT, "xiangjia1_186m"))
DJI_DIR = os.environ.get("DRONE_DJI_DIR", os.path.join(_DATA_ROOT, "xiangjia1"))
WORK = os.environ.get("DRONE_WORK", os.path.join(_PROJ, "work"))
OUT = os.environ.get("DRONE_OUT", os.path.join(_PROJ, "out"))
SRC = _HERE

for d in (WORK, OUT):
    os.makedirs(d, exist_ok=True)

# UTM：默认 50N（对应 EPSG:32650）。zone*6-183 = 中央经线
UTM_ZONE = int(os.environ.get("DRONE_UTM_ZONE", "50"))
UTM_SOUTH = os.environ.get("DRONE_SOUTH", "0").strip() in ("1", "true", "True")
EPSG = os.environ.get("DRONE_EPSG") or f"EPSG:{32700 + UTM_ZONE if UTM_SOUTH else 32600 + UTM_ZONE}"

_T0 = time.time()


def log(msg):
    el = time.time() - _T0
    print(f"[{el:7.1f}s] {msg}", flush=True)


# ---------------- 旋转工具 ----------------
def euler_to_R(omega_deg, phi_deg, kappa_deg):
    """大疆/Pix4D 约定的 Omega-Phi-Kappa (deg) -> 相机到世界旋转矩阵 R。"""
    o = math.radians(omega_deg)
    p = math.radians(phi_deg)
    k = math.radians(kappa_deg)
    co, so = math.cos(o), math.sin(o)
    cp, sp = math.cos(p), math.sin(p)
    ck, sk = math.cos(k), math.sin(k)
    # R = Rx(o) * Ry(p) * Rz(k)  (Pix4D/BlocksExchange 标准)
    Rx = np.array([[1, 0, 0], [0, co, -so], [0, so, co]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[ck, -sk, 0], [sk, ck, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def R_to_euler(R):
    """R = Rx(o)Ry(p)Rz(k) 的反解。"""
    p = math.asin(-R[2, 0])
    o = math.atan2(R[2, 1], R[2, 2])
    k = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(o), math.degrees(p), math.degrees(k)


def rotvec_to_R(r):
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def R_to_rotvec(R):
    import cv2
    r, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return r.ravel()


def ypr_to_R_camera_to_world(yaw_deg, pitch_deg, roll_deg):
    """由 DJI 云台 YPR 构造 相机->世界 旋转。
    DJI 定义：yaw 为机头绕 z 轴(向下为正?) ，此处用实测数据标定：
      GimbalYaw=-91, Pitch=-90, Roll=0 时相机朝下且影像上方向对应航线方向。
    采用 R = Rz(yaw+90) * Ry(pitch) * Rx(roll) 的常用近似，再由 AT 位姿做整体校正。
    """
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    return Rz @ Ry @ Rx


# ---------------- 相机模型 ----------------
class Camera:
    """针孔 + Brown 畸变(k1,k2,k3,p1,p2)，像素单位。"""

    def __init__(self, w, h, fx, fy, cx, cy, k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0):
        self.w, self.h = int(w), int(h)
        self.fx, self.fy = float(fx), float(fy)
        self.cx, self.cy = float(cx), float(cy)
        self.k1, self.k2, self.k3 = float(k1), float(k2), float(k3)
        self.p1, self.p2 = float(p1), float(p2)

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def distort(self, xn, yn):
        """归一化平面 -> 畸变后归一化平面"""
        r2 = xn * xn + yn * yn
        radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2
        xd = xn * radial + 2 * self.p1 * xn * yn + self.p2 * (r2 + 2 * xn * xn)
        yd = yn * radial + self.p1 * (r2 + 2 * yn * yn) + 2 * self.p2 * xn * yn
        return xd, yd

    def project(self, Xc):
        """相机坐标 (N,3) -> 像素 (N,2)"""
        Xc = np.asarray(Xc, dtype=np.float64)
        xn = Xc[:, 0] / Xc[:, 2]
        yn = Xc[:, 1] / Xc[:, 2]
        xd, yd = self.distort(xn, yn)
        u = self.fx * xd + self.cx
        v = self.fy * yd + self.cy
        return np.stack([u, v], axis=1)

    def undistort_map(self):
        import cv2
        K = self.K
        D = np.array([self.k1, self.k2, self.p1, self.p2, self.k3])
        return cv2.initUndistortRectifyMap(K, D, None, K, (self.w, self.h), cv2.CV_32FC1)


def imread_any(path, gray=False, color=None):
    """支持中文路径的图像读取"""
    import cv2
    flag = cv2.IMREAD_GRAYSCALE if gray else (
        cv2.IMREAD_UNCHANGED if color == "raw" else cv2.IMREAD_COLOR)
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, flag)
    except Exception:
        return None


def imwrite_any(path, img, params=None):
    """支持中文路径的图像写出"""
    import cv2
    ext = os.path.splitext(path)[1].lower()
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(path)
    return True


def save_json(path, obj):
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def load_json(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def np_save(path, arr):
    np.save(path, arr)


def np_load(path):
    return np.load(path, allow_pickle=True)
